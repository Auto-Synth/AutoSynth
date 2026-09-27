import argparse
import csv
import json

import numpy as np

from .audio import _load_stereo_audio
from .clap import ClapMetric
from .common import load_config, read_manifest, setup, write_json
from .metrics import AudioBoxAestheticsScorer, WmfccMetric
from .paths import resolve_project_path
from .rewards import acoustic_metrics


def load_waveform(path, cfg):
    sr = cfg["audio"]["sample_rate"]
    return _load_stereo_audio(
        resolve_project_path(path),
        sample_rate=sr,
        target_samples=round(sr * cfg["audio"]["duration"]),
    )[1]


def report(rows, output, metric_names):
    if not rows:
        raise ValueError("No evaluation rows")
    output = resolve_project_path(output)
    output.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for dataset in sorted({r.get("dataset", "test") for r in rows}):
        selected = [r for r in rows if r.get("dataset", "test") == dataset]
        summary[dataset] = {
            "count": len(selected),
            **{
                name: float(np.mean([r[name] for r in selected]))
                for name in metric_names
            },
        }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", choices=("inversion", "text"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    setup(cfg)
    rows = read_manifest(args.manifest)
    if any(row.get("task", args.task) != args.task for row in rows):
        raise ValueError("Manifest task mismatch")
    clap = ClapMetric(
        cfg["metrics"]["clap_checkpoint"], cfg["device"], cfg["metrics"]["batch_size"]
    )
    wmfcc = (
        WmfccMetric(sample_rate=cfg["audio"]["sample_rate"])
        if args.task == "inversion"
        else None
    )
    aesthetics = (
        AudioBoxAestheticsScorer(
            resolve_project_path(cfg["metrics"]["audiobox_checkpoint"]),
            axes=("CE", "CU", "PQ"),
            device=cfg["device"],
        )
        if args.task == "text"
        else None
    )
    scored = []
    size = cfg["metrics"]["batch_size"]
    for start in range(0, len(rows), size):
        batch = rows[start : start + size]
        predictions = [load_waveform(row["output"], cfg) for row in batch]
        if args.task == "inversion":
            targets = [load_waveform(row["reference"], cfg) for row in batch]
            a, b = clap.audio(targets + predictions, cfg["audio"]["sample_rate"]).chunk(
                2
            )
            auxiliary = [
                acoustic_metrics(p, t, cfg["audio"]["sample_rate"], wmfcc)
                for p, t in zip(predictions, targets)
            ]
        else:
            a = clap.text([row["prompt"] for row in batch])
            b = clap.audio(predictions, cfg["audio"]["sample_rate"])
            auxiliary = aesthetics.score_batch(
                predictions, sample_rate=cfg["audio"]["sample_rate"]
            )
        for row, distance, extra in zip(batch, clap.distance(a, b).tolist(), auxiliary):
            metrics = {"clap": distance, **extra}
            if not all(np.isfinite(value) for value in metrics.values()):
                raise ValueError("Non-finite evaluation metric")
            scored.append({**row, **metrics})
        print(f"Scored {len(scored)}/{len(rows)}", flush=True)
    names = (
        ["clap", "wmfcc", "sot", "mss", "rms"]
        if args.task == "inversion"
        else ["clap", "ce", "cu", "pq"]
    )
    report(scored, args.output, names)
