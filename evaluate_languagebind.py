import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F

from autosynth.common import load_config, read_manifest, setup
from autosynth.evaluation import load_waveform, report
from autosynth.paths import resolve_project_path


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", choices=("inversion", "text"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    setup(cfg)
    sys.path.insert(0, str(resolve_project_path(cfg["metrics"]["languagebind_code"])))
    from languagebind import LanguageBindAudio, LanguageBindAudioTokenizer
    from languagebind.audio.processing_audio import get_audio_transform

    model_path = str(resolve_project_path(cfg["metrics"]["languagebind_checkpoint"]))
    model, info = LanguageBindAudio.from_pretrained(
        model_path, local_files_only=True, output_loading_info=True
    )
    if info["missing_keys"] or info["mismatched_keys"] or info["error_msgs"]:
        raise ValueError(f"Incomplete LanguageBind weights: {info}")
    model = model.to(cfg["device"]).float().eval().requires_grad_(False)
    transform = get_audio_transform(model.config)
    tokenizer = (
        LanguageBindAudioTokenizer.from_pretrained(model_path, local_files_only=True)
        if args.task == "text"
        else None
    )
    rows = read_manifest(args.manifest)

    def audio_features(paths):
        batch = torch.stack(
            [
                transform(
                    (
                        torch.from_numpy(load_waveform(path, cfg)).unsqueeze(0),
                        cfg["audio"]["sample_rate"],
                    )
                )
                for path in paths
            ]
        ).to(cfg["device"])
        return F.normalize(model.get_image_features(pixel_values=batch).float(), dim=-1)

    scored = []
    for start in range(0, len(rows), cfg["metrics"]["batch_size"]):
        batch = rows[start : start + cfg["metrics"]["batch_size"]]
        generated = audio_features([row["output"] for row in batch])
        if args.task == "inversion":
            reference = audio_features([row["reference"] for row in batch])
        else:
            tokens = tokenizer(
                [row["prompt"] for row in batch],
                max_length=77,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            reference = F.normalize(
                model.get_text_features(
                    **{key: value.to(cfg["device"]) for key, value in tokens.items()}
                ).float(),
                dim=-1,
            )
        a, b = reference.double(), generated.double()
        distance = 1 - (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1))
        for row, value in zip(batch, distance.clamp(0, 2).cpu().tolist()):
            if not np.isfinite(value):
                raise ValueError("Non-finite LanguageBind distance")
            scored.append({**row, "languagebind": value})
        print(f"Scored {len(scored)}/{len(rows)}", flush=True)
    report(scored, args.output, ["languagebind"])


if __name__ == "__main__":
    main()
