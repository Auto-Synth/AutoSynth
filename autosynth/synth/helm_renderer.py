from __future__ import annotations
import hashlib
import gc
import json
import os
import select
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence
import numpy as np
from autosynth.paths import resolve_project_path
from .helm_midi import MidiPerformance, write_standard_midi
from .helm_parameter_spec import DELAY_PARAMETER_NAMES, HelmParameterSpace
from .helm_state import (
    HelmPresetSeed,
    HelmRoute,
    build_state_blob,
    parse_state_blob,
    payload_with_routes,
    read_helm_preset,
    routes_from_payload,
)

CATALOG_CACHE_VERSION = 3


class HelmDawRenderer:
    def __init__(
        self,
        plugin_path: str | Path = "plugins/helm.so",
        *,
        parameter_schema_path: str | Path = "assets/helm_parameter_space.json",
        sample_rate: int = 44100,
        block_size: int = 512,
        state_load_flush_seconds: float = 0.02,
        tail_flush_seconds: float = 2.0,
        fadeout_seconds: float = 0.05,
        auto_start_xvfb: bool = True,
        xvfb_path: str = "Xvfb",
    ) -> None:
        self.plugin_path = resolve_project_path(plugin_path)
        if not self.plugin_path.exists():
            raise FileNotFoundError(f"Helm plugin not found: {self.plugin_path}")
        self.space = HelmParameterSpace(parameter_schema_path)
        self.sample_rate = int(sample_rate)
        self.block_size = int(block_size)
        self.state_load_flush_seconds = float(state_load_flush_seconds)
        self.tail_flush_seconds = float(tail_flush_seconds)
        self.fadeout_samples = max(
            0, int(round(float(fadeout_seconds) * self.sample_rate))
        )
        self._xvfb_process: subprocess.Popen | None = None
        if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
            if not auto_start_xvfb:
                raise RuntimeError(
                    "Helm requires an X display; set DISPLAY or enable auto_start_xvfb"
                )
            self._start_xvfb(xvfb_path)
        try:
            import dawdreamer as daw
        except ImportError as exc:
            self.close()
            raise ImportError(
                "HelmDawRenderer requires `dawdreamer` in the active environment"
            ) from exc
        self._daw = daw
        self._create_plugin_instance()
        self._state_prefix, self._init_payload = self._capture_current_state()
        self._init_parameters = self._read_parameters()

    def _create_plugin_instance(self) -> None:
        self.engine = self._daw.RenderEngine(self.sample_rate, self.block_size)
        self.synth = self.engine.make_plugin_processor("helm", str(self.plugin_path))
        self.engine.load_graph([(self.synth, [])])
        if int(self.synth.get_plugin_parameter_size()) != 139:
            raise RuntimeError(
                f"Expected Helm to expose 139 fixed parameters, got {self.synth.get_plugin_parameter_size()}"
            )
        runtime_names = tuple(
            (str(self.synth.get_parameter_name(index)) for index in range(139))
        )
        if runtime_names != self.space.names:
            mismatch = next(
                (
                    index
                    for index, pair in enumerate(zip(runtime_names, self.space.names))
                    if pair[0] != pair[1]
                ),
                None,
            )
            raise RuntimeError(
                f"Helm parameter schema does not match plugin at index {mismatch}"
            )

    def recreate_plugin_instance(self) -> None:
        self.release_plugin_instance()
        self._create_plugin_instance()

    def init_seed(self) -> HelmPresetSeed:
        payload = dict(self._init_payload)
        payload["patch_name"] = "Init / Basic Saw"
        payload["folder_name"] = "Init"
        payload.setdefault("settings", {})["modulations"] = []
        return HelmPresetSeed(
            preset_id="__helm_init__",
            name="Init / Basic Saw",
            category="Init",
            state_payload=payload,
            parameters=self._init_parameters.copy(),
            routes=(),
            source_path=None,
        )

    def build_or_load_catalog(
        self, preset_root: str | Path, cache_path: str | Path, *, rebuild: bool = False
    ) -> tuple[HelmPresetSeed, list[HelmPresetSeed]]:
        preset_root_path = resolve_project_path(preset_root)
        cache_path_obj = resolve_project_path(cache_path)
        preset_paths = sorted(preset_root_path.rglob("*.helm"))
        if not preset_paths:
            raise FileNotFoundError(f"No .helm presets found below {preset_root_path}")
        fingerprint = self._catalog_fingerprint(preset_root_path, preset_paths)
        if cache_path_obj.exists() and (not rebuild):
            cached = json.loads(cache_path_obj.read_text(encoding="utf-8"))
            if (
                cached.get("cache_version") == CATALOG_CACHE_VERSION
                and cached.get("fingerprint") == fingerprint
                and (cached.get("parameter_names") == list(self.space.names))
            ):
                init = self._seed_from_json(cached["init"])
                factory = [
                    self._seed_from_json(item) for item in cached["factory_presets"]
                ]
                return (init, factory)
        init = self.init_seed()
        factory: list[HelmPresetSeed] = []
        for path in preset_paths:
            raw_payload = read_helm_preset(path)
            upgraded_payload, parameters = self._upgrade_payload(raw_payload)
            relative = path.relative_to(preset_root_path)
            category = relative.parent.name if relative.parent.name else "Uncategorized"
            preset_id = relative.with_suffix("").as_posix()
            name = str(upgraded_payload.get("patch_name") or path.stem)
            factory.append(
                HelmPresetSeed(
                    preset_id=preset_id,
                    name=name,
                    category=category,
                    state_payload=upgraded_payload,
                    parameters=parameters,
                    routes=routes_from_payload(upgraded_payload),
                    source_path=relative.as_posix(),
                )
            )
        if len(factory) != 274:
            raise RuntimeError(
                f"Expected 274 Helm factory presets, found {len(factory)}"
            )
        cache_payload = {
            "cache_version": CATALOG_CACHE_VERSION,
            "fingerprint": fingerprint,
            "parameter_names": list(self.space.names),
            "factory_count": len(factory),
            "total_seed_count_including_init": len(factory) + 1,
            "init": self._seed_to_json(init),
            "factory_presets": [self._seed_to_json(seed) for seed in factory],
        }
        cache_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=cache_path_obj.parent,
            prefix="helm_catalog_",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(cache_payload, handle, ensure_ascii=False, separators=(",", ":"))
            temporary_path = Path(handle.name)
        os.replace(temporary_path, cache_path_obj)
        return (init, factory)

    def render(
        self,
        *,
        base_state_payload: dict | None,
        parameters: np.ndarray,
        routes: Sequence[HelmRoute],
        performance: MidiPerformance,
        duration_seconds: float,
    ) -> np.ndarray:
        values = np.asarray(parameters, dtype=np.float32)
        if values.shape != (139,) or not np.isfinite(values).all():
            raise ValueError(
                f"Helm parameters must be a finite (139,) vector, got {values.shape}"
            )
        values = self.space.canonicalize_delay_off(values)
        routes = tuple(
            (
                route
                for route in routes
                if route.destination not in DELAY_PARAMETER_NAMES
            )
        )
        payload = payload_with_routes(
            self._init_payload if base_state_payload is None else base_state_payload,
            routes,
            patch_name="AutoSynth"
            if base_state_payload is None
            else "AutoSynth augmented",
        )
        self._load_payload(payload)
        if self.state_load_flush_seconds > 0.0:
            self.engine.render(self.state_load_flush_seconds)
        self.synth.clear_midi()
        if self.tail_flush_seconds > 0.0:
            self.engine.render(self.tail_flush_seconds)
        self.synth.clear_midi()
        for index, value in enumerate(values):
            self.synth.set_parameter(index, float(np.clip(value, 0.0, 1.0)))
        midi_path = self._temporary_path(".mid")
        try:
            write_standard_midi(performance, midi_path)
            loaded = self.synth.load_midi(
                str(midi_path), clear_previous=True, beats=False, all_events=True
            )
            if not loaded:
                raise RuntimeError("DawDreamer failed to load generated MIDI")
            self.engine.render(max(float(duration_seconds), 0.01))
            audio = np.asarray(self.engine.get_audio(), dtype=np.float32)
        finally:
            self.synth.clear_midi()
            midi_path.unlink(missing_ok=True)
        if audio.ndim != 2 or audio.shape[0] != 2:
            raise RuntimeError(
                f"Expected DawDreamer stereo output shaped (2, samples), got {audio.shape}"
            )
        if self.fadeout_samples > 0 and audio.shape[-1] >= self.fadeout_samples:
            audio[..., -self.fadeout_samples :] *= np.linspace(
                1.0, 0.0, self.fadeout_samples, dtype=np.float32
            )
        return audio

    def verify_route_roundtrip(self, route: HelmRoute) -> float:
        payload = payload_with_routes(
            self._init_payload, [route], patch_name="route roundtrip"
        )
        self._load_payload(payload)
        self.engine.render(max(self.state_load_flush_seconds, 0.02))
        _prefix, saved = self._capture_current_state()
        routes = routes_from_payload(saved)
        if (
            len(routes) != 1
            or routes[0].source != route.source
            or routes[0].destination != route.destination
        ):
            raise RuntimeError(f"Helm route did not round-trip: {route}")
        return float(routes[0].amount)

    def _upgrade_payload(self, raw_payload: dict) -> tuple[dict, np.ndarray]:
        self._load_payload(raw_payload)
        self.engine.render(max(self.state_load_flush_seconds, 0.02))
        parameters = self._read_parameters()
        _prefix, upgraded = self._capture_current_state()
        return (upgraded, parameters)

    def _read_parameters(self) -> np.ndarray:
        return np.asarray(
            [float(self.synth.get_parameter(index)) for index in range(139)],
            dtype=np.float32,
        )

    def _capture_current_state(self) -> tuple[bytes, dict]:
        path = self._temporary_path(".state")
        try:
            self.synth.save_state(str(path))
            return parse_state_blob(path.read_bytes())
        finally:
            path.unlink(missing_ok=True)

    def _load_payload(self, payload: dict) -> None:
        path = self._temporary_path(".state")
        try:
            path.write_bytes(build_state_blob(self._state_prefix, payload))
            self.synth.load_state(str(path))
        finally:
            path.unlink(missing_ok=True)

    @staticmethod
    def _temporary_path(suffix: str) -> Path:
        descriptor, name = tempfile.mkstemp(prefix="ddsynth_helm_", suffix=suffix)
        os.close(descriptor)
        return Path(name)

    def _catalog_fingerprint(
        self, preset_root: Path, preset_paths: Sequence[Path]
    ) -> str:
        digest = hashlib.sha256()
        digest.update(f"catalog-v{CATALOG_CACHE_VERSION}".encode())
        for path in (self.plugin_path, self.space.schema_path):
            digest.update(path.read_bytes())
        for path in preset_paths:
            digest.update(path.relative_to(preset_root).as_posix().encode("utf-8"))
            digest.update(path.read_bytes())
        return digest.hexdigest()

    @staticmethod
    def _seed_to_json(seed: HelmPresetSeed) -> dict:
        return {
            "preset_id": seed.preset_id,
            "name": seed.name,
            "category": seed.category,
            "state_payload": seed.state_payload,
            "parameters": np.asarray(seed.parameters, dtype=np.float32).tolist(),
            "routes": [
                {
                    "source": route.source,
                    "destination": route.destination,
                    "amount": float(route.amount),
                }
                for route in seed.routes
            ],
            "source_path": seed.source_path,
        }

    @staticmethod
    def _seed_from_json(payload: dict) -> HelmPresetSeed:
        return HelmPresetSeed(
            preset_id=str(payload["preset_id"]),
            name=str(payload["name"]),
            category=str(payload["category"]),
            state_payload=payload["state_payload"],
            parameters=np.asarray(payload["parameters"], dtype=np.float32),
            routes=tuple(
                (
                    HelmRoute(
                        str(item["source"]),
                        str(item["destination"]),
                        float(item["amount"]),
                    )
                    for item in payload["routes"]
                )
            ),
            source_path=payload.get("source_path"),
        )

    def _start_xvfb(self, configured_path: str) -> None:
        executable = (
            configured_path if Path(configured_path).exists() else shutil.which("Xvfb")
        )
        if not executable:
            raise RuntimeError(
                "Helm requires Xvfb on a headless host, but no Xvfb executable was found"
            )
        read_fd, write_fd = os.pipe()
        process = None
        try:
            process = subprocess.Popen(
                [
                    str(executable),
                    "-displayfd",
                    str(write_fd),
                    "-screen",
                    "0",
                    "1280x1024x24",
                    "-nolisten",
                    "tcp",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                pass_fds=(write_fd,),
            )
            os.close(write_fd)
            write_fd = None
            if not select.select([read_fd], [], [], 10.0)[0]:
                raise RuntimeError("Timed out waiting for a private Xvfb display")
            display_number = os.read(read_fd, 64).strip()
            if not display_number.isdigit() or process.poll() is not None:
                raise RuntimeError("Xvfb exited without allocating a private display")
            os.environ["DISPLAY"] = f":{display_number.decode('ascii')}"
            self._xvfb_process = process
        except BaseException:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
        finally:
            os.close(read_fd)
            if write_fd is not None:
                os.close(write_fd)

    def release_plugin_instance(self) -> None:
        try:
            if getattr(self, "synth", None) is not None:
                self.synth.clear_midi()
        except Exception:
            pass
        self.synth = None
        self.engine = None
        gc.collect()

    def close(self) -> None:
        try:
            if getattr(self, "synth", None) is not None:
                self.synth.clear_midi()
        except Exception:
            pass
        self.synth = None
        self.engine = None
        if self._xvfb_process is not None and self._xvfb_process.poll() is None:
            self._xvfb_process.terminate()
            try:
                self._xvfb_process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._xvfb_process.kill()
        self._xvfb_process = None

    def __enter__(self) -> "HelmDawRenderer":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
