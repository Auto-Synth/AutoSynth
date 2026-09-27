import argparse
import json
import math
import random
import time
from contextlib import nullcontext

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .audio import _load_stereo_audio, _mel_from_stereo
from .checkpoints import MetricCheckpointStore
from .synth.helm_state import ROUTE_AMOUNT_ENCODING
from .clap import ClapMetric
from .common import build_model, load_config, read_checkpoint, read_manifest, setup
from .data import HelmH5Dataset, helm_collate_batch
from .metrics import WmfccMetric
from .paths import resolve_project_path
from .policy import group_advantages, grpo_loss, sample_sequences, token_log_probs
from .render import SubprocessHelmRendererPool
from .rewards import RewardEvaluator, acoustic_metrics


def render_sequences(renderer, model, sequences, duration):
    jobs = [
        {"decoded": model.tokenizer.decode(sequence), "duration_seconds": duration}
        for sequence in sequences
    ]
    audio = renderer.render_jobs_resilient(jobs)
    if any(item is None for item in audio):
        raise RuntimeError("Helm rendering failed")
    return audio


def load_audio_row(row, cfg):
    sr = cfg["audio"]["sample_rate"]
    stereo, _ = _load_stereo_audio(
        resolve_project_path(row["audio"]),
        sample_rate=sr,
        target_samples=round(sr * cfg["audio"]["duration"]),
    )
    return _mel_from_stereo(stereo, sample_rate=sr, metadata=cfg["audio"]), stereo


def teacher_loss(model, batch, temperature):
    logits = model(
        batch["mel_spec"],
        batch["token_value_ids"],
        batch["token_kind_ids"],
        batch["token_field_ids"],
        batch["token_mask"],
    )
    return model.loss(
        logits,
        batch["token_value_ids"],
        batch["token_kind_ids"],
        batch["token_field_ids"],
        batch["loss_mask"],
        temperature=temperature,
        return_components=True,
    )


def to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def mean_records(records):
    return {
        key: float(np.mean([r[key] for r in records if key in r]))
        for key in sorted(set().union(*records))
    }


def source_at(position, mix, seed):
    cycle = ["stable_audio"] * mix["stable_audio"] + ["in_domain"] * mix["in_domain"]
    block, offset = divmod(position, len(cycle))
    random.Random(seed + block).shuffle(cycle)
    return cycle[offset]


class Validator:
    def __init__(self, cfg, renderer, reward=None):
        self.cfg, self.renderer, self.reward = cfg, renderer, reward
        self.helm = HelmH5Dataset(
            cfg["data"]["helm"], "val", cfg["data"]["order"], read_audio=True
        )
        count = cfg["validation"]["samples_per_set"]
        self.indices = sorted(
            random.Random(cfg["seed"]).sample(range(len(self.helm)), count)
        )
        self.rows = {}
        for name in ("stable_audio", "nsynth"):
            manifest = cfg["data"][name + "_val"]
            if manifest is not None:
                rows = read_manifest(manifest)
                if len(rows) < count:
                    raise ValueError(f"Insufficient {name} validation samples")
                self.rows[name] = random.Random(cfg["seed"]).sample(rows, count)
        self.clap = (
            reward.clap
            if reward
            else ClapMetric(
                cfg["metrics"]["clap_checkpoint"],
                cfg["device"],
                cfg["metrics"]["batch_size"],
            )
        )
        self.wmfcc = WmfccMetric(sample_rate=cfg["audio"]["sample_rate"])

    @torch.no_grad()
    def __call__(self, model):
        model.eval()
        cfg = self.cfg
        device, sr = cfg["device"], cfg["audio"]["sample_rate"]
        size = cfg["validation"]["batch_size"]
        result = {}
        for domain in ["in_domain", *self.rows]:
            scores = []
            items = self.indices if domain == "in_domain" else self.rows[domain]
            for offset in range(0, len(items), size):
                chosen = items[offset : offset + size]
                if domain == "in_domain":
                    batch = helm_collate_batch([self.helm[i] for i in chosen])
                    mel = batch["mel_spec"].to(device)
                    targets = list(batch["audio"].numpy())
                    prompts = [None] * len(chosen)
                    loss, param_loss, midi_loss = teacher_loss(
                        model, to_device(batch, device), cfg["sft"]["temperature"]
                    )
                    losses = dict(
                        loss=float(loss),
                        param_loss=float(param_loss),
                        midi_loss=float(midi_loss),
                    )
                else:
                    loaded = [load_audio_row(row, cfg) for row in chosen]
                    mel = torch.stack([x[0] for x in loaded]).to(device)
                    targets = [x[1] for x in loaded]
                    prompts = (
                        [row["prompt"] for row in chosen]
                        if domain == "stable_audio"
                        else [None] * len(chosen)
                    )
                    losses = {}
                sequences = sample_sequences(
                    model,
                    model.tokenizer,
                    mel,
                    **cfg["sampling"],
                    greedy=True,
                    generator=torch.Generator(device=device).manual_seed(cfg["seed"]),
                )
                predictions = render_sequences(
                    self.renderer, model, sequences, cfg["audio"]["duration"]
                )
                if self.reward:
                    batch_scores = self.reward(predictions, targets, prompts, 1)
                else:
                    targets = [x.mean(axis=0) for x in targets]
                    predictions = [x.mean(axis=0) for x in predictions]
                    features = self.clap.audio(targets + predictions, sr)
                    a, b = features.chunk(2)
                    distances = self.clap.distance(a, b).tolist()
                    batch_scores = [
                        {
                            "clap": distance,
                            **acoustic_metrics(pred, target, sr, self.wmfcc),
                        }
                        for pred, target, distance in zip(
                            predictions, targets, distances
                        )
                    ]
                scores.extend([{**entry, **losses} for entry in batch_scores])
            result.update(
                {
                    f"val_{domain}/{key}": value
                    for key, value in mean_records(scores).items()
                }
            )
        return result

    def close(self):
        self.helm.close()


class Run:
    def __init__(self, output, cfg, stage, resume):
        self.output = resolve_project_path(output)
        if not resume and (self.output / "metrics.jsonl").exists():
            raise FileExistsError(
                "Run already exists; use --resume or a new output directory"
            )
        self.output.mkdir(parents=True, exist_ok=True)
        self.cfg, self.stage = cfg, stage
        self.metrics = (self.output / "metrics.jsonl").open(
            "a", encoding="utf-8", buffering=1
        )
        self.wandb = None
        if cfg["logging"]["wandb"]:
            import wandb

            self.wandb = wandb.init(
                project=cfg["logging"]["project"],
                name=self.output.name,
                config=cfg,
                dir=str(self.output),
            )

    def log(self, step, metrics):
        self.metrics.write(
            json.dumps({"step": step, **metrics}, allow_nan=False) + "\n"
        )
        if self.wandb:
            self.wandb.log(metrics, step=step)
        if step % self.cfg["logging"]["every"] == 0 or any(
            k.startswith("val_") for k in metrics
        ):
            print(json.dumps({"step": step, **metrics}), flush=True)

    def save(self, step, model, optimizer, scheduler, validation):
        def save(path):
            payload = dict(
                format="autosynth-v1",
                route_amount_encoding=ROUTE_AMOUNT_ENCODING,
                stage=self.stage,
                step=step,
                config=self.cfg,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict() if scheduler else None,
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else [],
            )
            temporary = path.with_suffix(".tmp")
            torch.save(payload, temporary)
            temporary.replace(path)

        last = self.output / "last.pt"
        save(last)
        if validation:
            maximize = {"rms", "reward", "audiobox_ce", "audiobox_cu"}
            modes = {
                name: "max" if name.split("/")[-1] in maximize else "min"
                for name in validation
            }
            store = MetricCheckpointStore(
                self.output / "checkpoints",
                modes=modes,
                top_k=self.cfg["validation"]["checkpoint_top_k"],
            )
            store.save(
                step=step, metrics=validation, save_checkpoint=save, save_last=False
            )

    def close(self):
        self.metrics.close()
        if self.wandb:
            self.wandb.finish()


def resume_state(path, stage, cfg, model, optimizer, scheduler):
    saved = read_checkpoint(path)
    if saved["stage"] != stage or saved["config"] != cfg:
        raise ValueError("Resume requires the original stage and config")
    model.load_state_dict(saved["model"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    if scheduler:
        scheduler.load_state_dict(saved["scheduler"])
    torch.set_rng_state(saved["torch_rng"])
    if saved["cuda_rng"]:
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
    return saved["step"]


def sft(cfg, output, resume=None):
    setup(cfg)
    settings = cfg["sft"]
    data = HelmH5Dataset(cfg["data"]["helm"], "train", cfg["data"]["order"])
    model = build_model(cfg).to(cfg["device"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"]
    )
    warmup, updates = settings["warmup_steps"], settings["updates"]

    def factor(step):
        if step < warmup:
            return (
                settings["start_factor"]
                + (1 - settings["start_factor"]) * step / warmup
            )
        progress = min(1.0, (step - warmup) / max(1, updates - warmup))
        floor = settings["min_lr"] / settings["lr"]
        return floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    renderer = SubprocessHelmRendererPool(**cfg["renderer"])
    validation = Validator(cfg, renderer)
    run = Run(output, cfg, "sft", resume)
    start = (
        resume_state(resume, "sft", cfg, model, optimizer, scheduler) if resume else 0
    )
    steps_per_epoch = math.ceil(len(data) / settings["batch_size"])
    iterator, loader = None, None
    try:
        for step in range(start + 1, updates + 1):
            started = time.monotonic()
            epoch, batch_index = divmod(step - 1, steps_per_epoch)
            if iterator is None or batch_index == 0:
                indices = torch.randperm(
                    len(data),
                    generator=torch.Generator().manual_seed(cfg["seed"] + epoch),
                ).tolist()
                indices = indices[batch_index * settings["batch_size"] :]
                loader = DataLoader(
                    Subset(data, indices),
                    batch_size=settings["batch_size"],
                    num_workers=settings["workers"],
                    collate_fn=helm_collate_batch,
                    pin_memory=str(cfg["device"]).startswith("cuda"),
                    generator=torch.Generator().manual_seed(cfg["seed"] + epoch),
                )
                iterator = iter(loader)
            model.train()
            batch = to_device(next(iterator), cfg["device"])
            optimizer.zero_grad(set_to_none=True)
            loss, param_loss, midi_loss = teacher_loss(
                model, batch, settings["temperature"]
            )
            if not torch.isfinite(loss):
                raise ValueError("Non-finite SFT loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), settings["grad_clip"], error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
            run.log(
                step,
                dict(
                    train_loss=float(loss.detach()),
                    param_loss=float(param_loss.detach()),
                    midi_loss=float(midi_loss.detach()),
                    lr=scheduler.get_last_lr()[0],
                    seconds=time.monotonic() - started,
                ),
            )
            val = (
                validation(model)
                if step % settings["eval_every"] == 0 or step == updates
                else {}
            )
            if val:
                run.log(step, val)
            if val or step % settings["save_every"] == 0 or step == updates:
                run.save(step, model, optimizer, scheduler, val)
    finally:
        validation.close()
        data.close()
        renderer.close()
        run.close()


def grpo(cfg, output, sft_checkpoint, resume=None):
    setup(cfg)
    settings = cfg["grpo"]
    model = build_model(cfg).to(cfg["device"])
    if not resume:
        saved = read_checkpoint(sft_checkpoint)
        if (
            saved["stage"] != "sft"
            or saved["config"]["model"] != cfg["model"]
            or saved["config"]["audio"] != cfg["audio"]
        ):
            raise ValueError("GRPO requires a matching SFT checkpoint")
        model.load_state_dict(saved["model"], strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"]
    )
    model.eval()
    data = HelmH5Dataset(
        cfg["data"]["helm"], "train", cfg["data"]["order"], read_audio=True
    )
    targets = read_manifest(cfg["data"]["stable_audio_train"])
    if any(not row.get("prompt", "").strip() for row in targets):
        raise ValueError("Stable Audio targets require their original text prompts")
    renderer = SubprocessHelmRendererPool(**cfg["renderer"])
    evaluator = RewardEvaluator(cfg)
    validator = Validator(cfg, renderer, evaluator)
    run = Run(output, cfg, "grpo", resume)
    start = resume_state(resume, "grpo", cfg, model, optimizer, None) if resume else 0
    size, group = settings["batch_size"], settings["group_size"]

    def autocast():
        return (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if settings["bf16"] and str(cfg["device"]).startswith("cuda")
            else nullcontext()
        )

    try:
        for step in range(start + 1, settings["updates"] + 1):
            started = time.monotonic()
            mels, audio, prompts = [], [], []
            for position in range((step - 1) * size, step * size):
                source = source_at(position, settings["mix"], cfg["seed"] + 4000000)
                rng = random.Random(cfg["seed"] + 5000000 + position)
                if source == "stable_audio":
                    row = targets[rng.randrange(len(targets))]
                    mel, waveform = load_audio_row(row, cfg)
                    prompt = row["prompt"]
                else:
                    sample = data[rng.randrange(len(data))]
                    mel, waveform, prompt = (
                        sample["mel_spec"],
                        sample["audio"].numpy(),
                        None,
                    )
                mels.append(mel)
                audio.append(waveform)
                prompts.append(prompt)
            inputs = torch.stack(mels).to(cfg["device"]).repeat_interleave(group, dim=0)
            generator = torch.Generator(device=cfg["device"]).manual_seed(
                cfg["seed"] + step
            )
            with autocast():
                sequences = sample_sequences(
                    model,
                    model.tokenizer,
                    inputs,
                    **cfg["sampling"],
                    generator=generator,
                )
            predictions = render_sequences(
                renderer, model, sequences, cfg["audio"]["duration"]
            )
            metrics = evaluator(predictions, audio, prompts, group)
            rewards = torch.tensor(
                [r["reward"] for r in metrics], device=cfg["device"]
            ).reshape(size, group)
            advantages = group_advantages(rewards)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                scores, mask = token_log_probs(
                    model, model.tokenizer, inputs, sequences, **cfg["sampling"]
                )
                loss = grpo_loss(scores, mask, advantages)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite GRPO loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), settings["grad_clip"], error_if_nonfinite=True
            )
            optimizer.step()
            run.log(
                step,
                {
                    "train_loss": float(loss.detach()),
                    "seconds": time.monotonic() - started,
                    **{"train/" + k: v for k, v in mean_records(metrics).items()},
                },
            )
            val = (
                validator(model)
                if step % settings["eval_every"] == 0 or step == settings["updates"]
                else {}
            )
            if val:
                run.log(step, val)
            if val or step % settings["save_every"] == 0 or step == settings["updates"]:
                run.save(step, model, optimizer, None, val)
    finally:
        validator.close()
        data.close()
        renderer.close()
        run.close()


def main(stage):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    if stage == "grpo":
        parser.add_argument("--sft-checkpoint")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if stage == "sft":
        sft(cfg, args.output, args.resume)
    else:
        if not args.resume and not args.sft_checkpoint:
            parser.error("GRPO requires --sft-checkpoint or --resume")
        grpo(cfg, args.output, args.sft_checkpoint, args.resume)
