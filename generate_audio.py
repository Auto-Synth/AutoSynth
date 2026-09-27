import argparse
import json
import os

from autosynth.common import load_config, read_manifest, setup
from autosynth.paths import project_relative_string, resolve_project_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds-per-prompt", type=int, default=1)
    args = parser.parse_args()
    if args.seeds_per_prompt < 1:
        parser.error("--seeds-per-prompt must be positive")
    cfg = load_config(args.config)
    setup(cfg)
    os.environ["ENABLE_TORCH_COMPILE"] = "0"
    from stable_audio_3.loading_utils import load_diffusion_cond
    from stable_audio_3.model import StableAudioModel
    import soundfile as sf
    import torch

    settings = cfg["text_generation"]
    directory = resolve_project_path(settings["checkpoint"])
    config = json.loads((directory / "model_config.json").read_text())
    for condition in config["model"]["conditioning"]["configs"]:
        if condition["type"] == "t5gemma":
            condition["config"]["model_path"] = str(directory / "t5gemma-b-b-ul2")
            condition["config"].pop("repo_id", None)
            condition["config"].pop("subfolder", None)
    network = load_diffusion_cond(
        config, str(directory / "model.safetensors"), device="cpu", model_half=True
    )
    network.to(cfg["device"]).eval().requires_grad_(False)
    network.use_lora, network.lora_names = False, []
    model = StableAudioModel(network, config, cfg["device"], True)
    out = resolve_project_path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(args.manifest)
    with (
        (out / "manifest.jsonl").open("x", encoding="utf-8") as stream,
        torch.inference_mode(),
    ):
        for index, row in enumerate(rows):
            for seed_index in range(args.seeds_per_prompt):
                sid = (
                    row["id"]
                    if args.seeds_per_prompt == 1
                    else f"{row['id']}_seed{seed_index}"
                )
                seed = (
                    row.get("seed", cfg["seed"] + index * args.seeds_per_prompt)
                    + seed_index
                )
                audio = model.generate(
                    prompt=row["prompt"],
                    duration=settings["duration"],
                    steps=settings["steps"],
                    cfg_scale=settings["cfg_scale"],
                    seed=seed,
                    batch_size=1,
                    chunked_decode=False,
                )
                path = out / f"{sid}.wav"
                sf.write(
                    path,
                    audio[0].float().cpu().numpy().T,
                    int(network.sample_rate),
                    subtype="PCM_24",
                )
                stream.write(
                    json.dumps(
                        {
                            "id": sid,
                            "audio": project_relative_string(path),
                            "prompt": row["prompt"],
                            "seed": seed,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                stream.flush()
            print(f"Generated {index + 1}/{len(rows)} prompts", flush=True)


if __name__ == "__main__":
    main()
