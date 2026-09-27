from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from .clap import ClapMetric
from .metrics import (
    AudioBoxAestheticsScorer,
    BasicPitchEmbeddingDistance,
    WmfccMetric,
    compute_mss,
    compute_rms_env_cosine,
    compute_sot,
)
from .paths import resolve_project_path


def acoustic_metrics(prediction, target, sample_rate, wmfcc):
    p = torch.from_numpy(prediction).unsqueeze(0)
    t = torch.from_numpy(target).unsqueeze(0)
    return {
        "wmfcc": float(wmfcc(p, t)),
        "sot": compute_sot(target, prediction, sample_rate),
        "mss": compute_mss(target, prediction, sample_rate),
        "rms": compute_rms_env_cosine(target, prediction, sample_rate),
    }


def combine_reward(metrics, weights, score_min=1.0, score_max=10.0):
    values = dict(metrics)
    for name in ("audiobox_ce", "audiobox_cu"):
        values[name] = float(
            np.clip((values[name] - score_min) / (score_max - score_min), 0, 1)
        )
    positive = {"audiobox_ce", "audiobox_cu", "rms"}
    return sum(
        weights[name] * values[name] * (1 if name in positive else -1)
        for name in weights
    )


class RewardEvaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.sr = cfg["audio"]["sample_rate"]
        self.clap = ClapMetric(
            cfg["metrics"]["clap_checkpoint"],
            cfg["device"],
            cfg["metrics"]["batch_size"],
        )
        self.pitch = BasicPitchEmbeddingDistance(**cfg["reward"]["basic_pitch"])
        self.aesthetics = AudioBoxAestheticsScorer(
            resolve_project_path(cfg["metrics"]["audiobox_checkpoint"]),
            axes=("CE", "CU"),
            device=cfg["device"],
        )
        self.wmfcc = WmfccMetric(sample_rate=self.sr)

    def __call__(self, predictions, targets, prompts, group_size):
        if len(predictions) != len(targets) * group_size or len(prompts) != len(
            targets
        ):
            raise ValueError("Reward target and group dimensions differ")
        if any(p is None for p in predictions):
            raise RuntimeError("Helm render failed; refusing to score missing audio")
        targets = [
            np.asarray(t, dtype=np.float32).mean(axis=0) if t.ndim == 2 else t
            for t in targets
        ]
        predictions = [
            np.asarray(p, dtype=np.float32).mean(axis=0) if p.ndim == 2 else p
            for p in predictions
        ]
        all_audio = targets + predictions
        embeddings = self.clap.audio(all_audio, self.sr)
        target_ids = torch.arange(len(targets)).repeat_interleave(group_size)
        pred_embeddings = embeddings[len(targets) :]
        audio_distances = self.clap.distance(
            pred_embeddings, embeddings[target_ids]
        ).tolist()
        text_ids = [i for i, text in enumerate(prompts) if text is not None]
        text_features = (
            dict(zip(text_ids, self.clap.text([prompts[i] for i in text_ids])))
            if text_ids
            else {}
        )
        pitch = self.pitch.extract_batch(all_audio, sample_rate=self.sr)
        aesthetics = self.aesthetics.score_batch(predictions, sample_rate=self.sr)

        def score(i):
            target_id = i // group_size
            aa = audio_distances[i]
            ta = (
                float(self.clap.distance(pred_embeddings[i], text_features[target_id]))
                if target_id in text_features
                else None
            )
            metrics = {
                "clap_audio": aa,
                "clap": aa if ta is None else (aa + ta) / 2,
                "basic_pitch": self.pitch.distance(
                    pitch[len(targets) + i], pitch[target_id]
                ),
                "audiobox_ce": aesthetics[i]["ce"],
                "audiobox_cu": aesthetics[i]["cu"],
                **acoustic_metrics(
                    predictions[i], targets[target_id], self.sr, self.wmfcc
                ),
            }
            if ta is not None:
                metrics["clap_text"] = ta
            reward = self.cfg["reward"]
            metrics["reward"] = combine_reward(
                metrics,
                reward["weights"],
                reward["audiobox_min"],
                reward["audiobox_max"],
            )
            if not all(np.isfinite(v) for v in metrics.values()):
                raise ValueError("Non-finite reward component")
            return metrics

        with ThreadPoolExecutor(max_workers=self.cfg["threads"]) as pool:
            return list(pool.map(score, range(len(predictions))))
