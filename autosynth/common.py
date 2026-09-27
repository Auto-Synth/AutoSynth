import json
import os
import random

import numpy as np
import torch
import yaml

from .paths import ROOT, resolve_project_path
from .synth.helm_state import ROUTE_AMOUNT_ENCODING


def load_config(path):
    with resolve_project_path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def setup(cfg):
    os.environ.setdefault("HF_HOME", str(ROOT / "models/huggingface"))
    torch.set_num_threads(cfg["threads"])
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if str(cfg["device"]).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select device: cpu explicitly")
    if str(cfg["device"]).startswith("cuda"):
        torch.cuda.set_device(torch.device(cfg["device"]))


def read_manifest(path):
    with resolve_project_path(path).open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError("Manifest is empty")
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Manifest sample IDs must be unique")
    for sid in ids:
        if (
            not isinstance(sid, str)
            or not sid
            or "/" in sid
            or "\\" in sid
            or sid in {".", ".."}
        ):
            raise ValueError(f"Invalid sample ID: {sid}")
    return rows


def write_json(path, value):
    path = resolve_project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_model(cfg):
    from .model import HelmAutoregressiveTransformer
    from .synth.helm_parameter_spec import HelmParameterSpace
    from .tokens import HelmAutoregressiveTokenizer, load_helm_parameter_order

    tokenizer = HelmAutoregressiveTokenizer(
        HelmParameterSpace(cfg["data"]["schema"]),
        parameter_order=load_helm_parameter_order(cfg["data"]["order"]),
        target_duration_seconds=cfg["audio"]["duration"],
    )
    return HelmAutoregressiveTransformer(tokenizer=tokenizer, **cfg["model"])


def read_checkpoint(path):
    payload = torch.load(
        resolve_project_path(path), map_location="cpu", weights_only=True
    )
    if payload["format"] != "autosynth-v1":
        raise ValueError("Expected an AutoSynth release checkpoint")
    if payload.get("route_amount_encoding") != ROUTE_AMOUNT_ENCODING:
        raise ValueError(
            "Checkpoint route representation is incompatible; train from regenerated "
            "destination-normalized data. Relabeling existing weights is not a conversion."
        )
    return payload


def load_model(path, device):
    payload = read_checkpoint(path)
    model = build_model(payload["config"])
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval(), payload
