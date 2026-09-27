from __future__ import annotations
import copy
import json
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
from .helm_parameter_spec import MODULATION_DESTINATIONS, MODULATION_SOURCES
from autosynth.paths import resolve_project_path


ROUTE_AMOUNT_ENCODING = "helm-destination-span-v1"


@lru_cache(maxsize=1)
def _route_amount_scales() -> dict[str, float]:
    path = resolve_project_path("assets/helm_modulation_ranges.json")
    ranges = json.loads(path.read_text(encoding="utf-8"))["parameter_ranges"]
    if set(ranges) != set(MODULATION_DESTINATIONS):
        raise ValueError("Helm modulation ranges must cover every destination")
    scales = {name: float(hi) - float(lo) for name, (lo, hi) in ranges.items()}
    if any(not np.isfinite(value) or value <= 0 for value in scales.values()):
        raise ValueError("Helm modulation destination spans must be finite and positive")
    return scales


def route_amount_scale(destination: str) -> float:
    return _route_amount_scales()[destination]


def normalize_route_amount(destination: str, native_amount: float) -> float:
    value = float(native_amount) / route_amount_scale(destination)
    if not np.isfinite(value) or abs(value) > 1.0 + 1e-6:
        raise ValueError(
            f"Native route amount outside {destination}'s modulation range: {native_amount}"
        )

    return min(1.0, max(-1.0, value))


def native_route_amount(destination: str, normalized_amount: float) -> float:
    value = float(normalized_amount)
    if not np.isfinite(value) or not -1.0 <= value <= 1.0:
        raise ValueError(f"Normalized route amount must be finite in [-1,1]: {value}")
    return value * route_amount_scale(destination)


@dataclass(frozen=True, order=True)
class HelmRoute:
    source: str
    destination: str
    amount: float

    def __post_init__(self) -> None:
        if self.source not in MODULATION_SOURCES:
            raise ValueError(f"Unknown Helm modulation source: {self.source}")
        if self.destination not in MODULATION_DESTINATIONS:
            raise ValueError(f"Unknown Helm modulation destination: {self.destination}")
        if not np.isfinite(self.amount) or not -1.0 <= float(self.amount) <= 1.0:
            raise ValueError(
                f"Helm route amount must be finite in [-1,1], got {self.amount}"
            )


@dataclass(frozen=True)
class HelmPresetSeed:
    preset_id: str
    name: str
    category: str
    state_payload: Mapping[str, Any]
    parameters: np.ndarray
    routes: tuple[HelmRoute, ...]
    source_path: str | None = None


def read_helm_preset(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Helm preset must contain a JSON object: {path}")
    return payload


def parse_state_blob(blob: bytes) -> tuple[bytes, dict[str, Any]]:
    search_from = 4
    while True:
        json_start = blob.find(b"{", search_from)
        if json_start < 0:
            break
        declared_size = struct.unpack(">I", blob[json_start - 4 : json_start])[0]
        json_size = max(declared_size - 1, 0)
        json_end = json_start + json_size
        if 0 < json_size and json_end <= len(blob):
            try:
                payload = json.loads(blob[json_start:json_end].decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                pass
            else:
                if isinstance(payload, dict):
                    return (blob[: json_start - 4], payload)
        search_from = json_start + 1
    raise ValueError("Could not locate a valid Helm JSON chunk in VST state blob")


def build_state_blob(prefix: bytes, payload: Mapping[str, Any]) -> bytes:
    raw_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return prefix + struct.pack(">I", len(raw_json) + 1) + raw_json + b"\x00"


def routes_from_payload(payload: Mapping[str, Any]) -> tuple[HelmRoute, ...]:
    settings = payload.get("settings", {})
    raw_routes = (
        settings.get("modulations", []) if isinstance(settings, Mapping) else []
    )
    by_pair: dict[tuple[str, str], float] = {}
    for raw in raw_routes:
        if not isinstance(raw, Mapping):
            continue
        source = str(raw.get("source", ""))
        destination = str(raw.get("destination", ""))
        if (
            source not in MODULATION_SOURCES
            or destination not in MODULATION_DESTINATIONS
        ):
            continue
        native_amount = float(raw.get("amount", 0.0))
        amount = normalize_route_amount(destination, native_amount)
        if abs(native_amount) >= 1e-06:
            by_pair[source, destination] = amount
    return tuple(
        (
            HelmRoute(source, destination, amount)
            for (source, destination), amount in sorted(by_pair.items())
        )
    )


def payload_with_routes(
    payload: Mapping[str, Any],
    routes: Sequence[HelmRoute],
    *,
    patch_name: str | None = None,
) -> dict[str, Any]:
    result = copy.deepcopy(dict(payload))
    settings = result.setdefault("settings", {})
    if not isinstance(settings, dict):
        raise ValueError("Helm state payload has non-object `settings`")
    unique: dict[tuple[str, str], HelmRoute] = {}
    for route in routes:
        if abs(native_route_amount(route.destination, route.amount)) >= 1e-06:
            unique[route.source, route.destination] = route
    settings["modulations"] = [
        {
            "source": route.source,
            "destination": route.destination,
            "amount": native_route_amount(route.destination, route.amount),
        }
        for route in sorted(unique.values(), key=lambda x: (x.source, x.destination))
    ]
    if patch_name is not None:
        result["patch_name"] = patch_name
    return result
