import argparse
from dataclasses import asdict
import json
import subprocess
import sys

import numpy as np
import soundfile as sf
import torch

from .audio import _load_stereo_audio, _mel_from_stereo
from .common import load_config, load_model, read_manifest, setup, write_json
from .paths import ROOT, project_relative_string, resolve_project_path
from .policy import sample_sequences
from .render import SubprocessHelmRendererPool
from .synth.helm_midi import write_standard_midi
from .synth.helm_parameter_spec import DELAY_PARAMETER_NAMES
from .synth.helm_state import payload_with_routes, ROUTE_AMOUNT_ENCODING


def export_program(decoded, sequence, tokenizer, output, lookup):
    output.mkdir(parents=True, exist_ok=True)
    payload = payload_with_routes(
        lookup["init_payload"], decoded.routes, patch_name=output.name
    )
    payload["author"] = "AutoSynth"
    for parameter, value in zip(
        tokenizer.space.parameters, decoded.parameters, strict=True
    ):
        if parameter.name in DELAY_PARAMETER_NAMES:
            native = lookup["delay_off_native"][parameter.name]
        else:
            values = lookup["parameters"][parameter.name]
            native = values[round(float(value) * (len(values) - 1))]
        payload["settings"][parameter.name] = native
    preset = {
        "autosynth_midi": {
            "tempo_bpm": 120,
            "time_unit": "seconds",
            **asdict(decoded.performance),
        },
        **payload,
    }
    write_json(output / "preset.helm", preset)
    write_json(
        output / "program.json",
        {
            "midi": asdict(decoded.performance),
            "route_amount_encoding": ROUTE_AMOUNT_ENCODING,
            "parameters_normalized": dict(
                zip(tokenizer.space.names, map(float, decoded.parameters))
            ),
            "routes": [asdict(route) for route in decoded.routes],
            "tokens": asdict(sequence),
        },
    )
    write_standard_midi(decoded.performance, output / "performance.mid")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", choices=("inversion", "text"), default="inversion")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--manifest")
    inputs.add_argument("--audio")
    inputs.add_argument("--text")
    parser.add_argument("--output", required=True)
    parser.add_argument("--intermediate-manifest")
    parser.add_argument("--text-python", default=sys.executable)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.task == "text" and args.audio or args.task == "inversion" and args.text:
        parser.error("--audio requires inversion; --text requires text")
    cfg = load_config(args.config)
    setup(cfg)
    out = resolve_project_path(args.output)
    if (out / "manifest.jsonl").exists():
        raise FileExistsError("Inference output already exists; select a new directory")
    out.mkdir(parents=True, exist_ok=True)
    if args.manifest:
        rows = read_manifest(args.manifest)
    else:
        rows = [
            {
                "id": "sample",
                "dataset": "input",
                **({"audio": args.audio} if args.audio else {"prompt": args.text}),
            }
        ]
    if args.task == "text":
        if not args.intermediate_manifest:
            source = out / "prompts.jsonl"
            source.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            subprocess.run(
                [
                    args.text_python,
                    str(ROOT / "generate_audio.py"),
                    "--config",
                    str(resolve_project_path(args.config)),
                    "--manifest",
                    str(source),
                    "--output",
                    str(out / "intermediate"),
                ],
                cwd=ROOT,
                check=True,
            )
            args.intermediate_manifest = out / "intermediate/manifest.jsonl"
        intermediate = {
            row["id"]: row for row in read_manifest(args.intermediate_manifest)
        }
        if set(intermediate) != {row["id"] for row in rows}:
            raise ValueError("Intermediate audio IDs do not match the test manifest")
        for row in rows:
            item = intermediate[row["id"]]
            if item["prompt"] != row["prompt"]:
                raise ValueError("Intermediate audio prompt mismatch")
            row["conditioning_audio"] = item["audio"]
    model, checkpoint = load_model(args.checkpoint, cfg["device"])
    audio_cfg = checkpoint["config"]["audio"]
    if audio_cfg != cfg["audio"]:
        raise ValueError("Inference audio settings must match training")
    lookup = json.loads((ROOT / "assets/helm_native_values.json").read_text())
    renderer = SubprocessHelmRendererPool(**cfg["renderer"]) if args.render else None
    sr = audio_cfg["sample_rate"]
    try:
        with (out / "manifest.jsonl").open("w", encoding="utf-8") as stream:
            for start in range(0, len(rows), args.batch_size):
                batch = rows[start : start + args.batch_size]
                audio = [
                    _load_stereo_audio(
                        resolve_project_path(
                            row["conditioning_audio"]
                            if args.task == "text"
                            else row["audio"]
                        ),
                        sample_rate=sr,
                        target_samples=round(sr * audio_cfg["duration"]),
                    )[0]
                    for row in batch
                ]
                mel = torch.stack(
                    [
                        _mel_from_stereo(waveform, sample_rate=sr, metadata=audio_cfg)
                        for waveform in audio
                    ]
                ).to(cfg["device"])
                sequences = sample_sequences(
                    model,
                    model.tokenizer,
                    mel,
                    **cfg["sampling"],
                    greedy=True,
                    generator=torch.Generator(device=cfg["device"]).manual_seed(
                        cfg["seed"]
                    ),
                )
                decoded = [model.tokenizer.decode(sequence) for sequence in sequences]
                rendered = (
                    renderer.render_jobs_resilient(
                        [
                            {"decoded": item, "duration_seconds": audio_cfg["duration"]}
                            for item in decoded
                        ]
                    )
                    if renderer
                    else [None] * len(batch)
                )
                if renderer and any(item is None for item in rendered):
                    raise RuntimeError("Helm rendering failed")
                for row, original, sequence, program, prediction in zip(
                    batch, audio, sequences, decoded, rendered, strict=True
                ):
                    folder = out / row["id"]
                    export_program(program, sequence, model.tokenizer, folder, lookup)
                    record = {
                        **row,
                        "task": args.task,
                        "program": project_relative_string(folder / "program.json"),
                        "preset": project_relative_string(folder / "preset.helm"),
                    }
                    sf.write(
                        folder / "conditioning.wav", original.T, sr, subtype="PCM_24"
                    )
                    record["conditioning_audio"] = project_relative_string(
                        folder / "conditioning.wav"
                    )
                    if args.task == "inversion":
                        record["reference"] = record["conditioning_audio"]
                    elif row.get("audio"):
                        record["reference"] = row["audio"]
                    if prediction is not None:
                        if not np.isfinite(prediction).all():
                            raise ValueError("Non-finite rendered audio")
                        sf.write(
                            folder / "output.wav", prediction.T, sr, subtype="PCM_24"
                        )
                        record["output"] = project_relative_string(
                            folder / "output.wav"
                        )
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                print(
                    f"Generated {min(start + len(batch), len(rows))}/{len(rows)}",
                    flush=True,
                )
    finally:
        if renderer:
            renderer.close()
