import argparse
import json

import numpy as np
import resampy
import torch

from autosynth.common import load_config, read_manifest, setup, write_json
from autosynth.evaluation import load_waveform, report
from autosynth.paths import resolve_project_path


def frechet_distance(reference, generated):
    a, b = (
        np.asarray(reference, dtype=np.float64),
        np.asarray(generated, dtype=np.float64),
    )
    ma, mb = a.mean(0), b.mean(0)
    ca, cb = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    eig, vec = np.linalg.eigh(ca)
    root_a = (vec * np.sqrt(np.maximum(eig, 0))) @ vec.T
    eig, vec = np.linalg.eigh(cb)
    root_b = (vec * np.sqrt(np.maximum(eig, 0))) @ vec.T
    cross = np.linalg.svd(root_a @ root_b, compute_uv=False).sum()
    value = (ma - mb) @ (ma - mb) + np.trace(ca) + np.trace(cb) - 2 * cross
    if not np.isfinite(value) or value < -1e-5:
        raise ValueError(f"Invalid FAD: {value}")
    return float(max(value, 0))


def paired_kl(reference, generated):
    a = reference.double().log_softmax(-1)
    b = generated.double().log_softmax(-1)
    return (a.exp() * (a - b)).sum(-1)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    setup(cfg)
    from torchvggish.vggish import VGGish
    from hear21passt.models.passt import get_model
    from hear21passt.models.preprocess import AugmentMelSTFT
    from hear21passt.wrapper import PasstBasicWrapper

    vggish = VGGish(urls={}, pretrained=False, postprocess=False, device=cfg["device"])
    vggish.load_state_dict(
        torch.load(
            resolve_project_path(cfg["metrics"]["vggish_checkpoint"]),
            map_location="cpu",
            weights_only=True,
        ),
        strict=True,
    )
    vggish.embeddings = torch.nn.Sequential(*list(vggish.embeddings.children())[:-1])
    vggish.to(cfg["device"]).eval().requires_grad_(False)
    net = get_model(arch="passt_s_swa_p16_128_ap476", pretrained=False)
    net.load_state_dict(
        torch.load(
            resolve_project_path(cfg["metrics"]["passt_checkpoint"]),
            map_location="cpu",
            weights_only=True,
        ),
        strict=True,
    )
    mel = AugmentMelSTFT(
        n_mels=128,
        sr=32000,
        win_length=800,
        hopsize=320,
        n_fft=1024,
        freqm=48,
        timem=192,
        htk=False,
        fmin=0.0,
        fmax=None,
        norm=1,
        fmin_aug_range=10,
        fmax_aug_range=2000,
    )
    passt = (
        PasstBasicWrapper(mel=mel, net=net, mode="logits")
        .to(cfg["device"])
        .eval()
        .requires_grad_(False)
    )

    def features(path):
        waveform = load_waveform(path, cfg)
        embedding = vggish(waveform, cfg["audio"]["sample_rate"]).cpu().numpy()
        wave32 = resampy.resample(waveform, cfg["audio"]["sample_rate"], 32000)
        logits = passt(torch.from_numpy(wave32).unsqueeze(0).to(cfg["device"])).cpu()
        return embedding, logits

    rows = read_manifest(args.manifest)
    scored, distributions = [], {}
    for index, row in enumerate(rows):
        a, la = features(row["reference"])
        b, lb = features(row["output"])
        dataset = row.get("dataset", "test")
        record = distributions.setdefault(
            dataset, {"reference": [], "generated": [], "kl": []}
        )
        kl = float(paired_kl(la, lb).mean())
        record["reference"].append(a)
        record["generated"].append(b)
        record["kl"].append(kl)
        scored.append({**row, "kl": kl})
        print(f"Scored {index + 1}/{len(rows)}", flush=True)
    report(scored, args.output, ["kl"])
    summaries = {
        name: {
            "count": len(values["kl"]),
            "fad": frechet_distance(
                np.concatenate(values["reference"]), np.concatenate(values["generated"])
            ),
            "kl": float(np.mean(values["kl"])),
        }
        for name, values in distributions.items()
    }
    write_json(resolve_project_path(args.output) / "summary.json", summaries)
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
