from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np
from autosynth.paths import resolve_project_path

MODULATION_SOURCES: tuple[str, ...] = (
    "amp_envelope",
    "aftertouch",
    "fil_envelope",
    "mod_envelope",
    "mod_wheel",
    "mono_lfo_1",
    "mono_lfo_2",
    "note",
    "pitch_wheel",
    "poly_lfo",
    "random",
    "step_sequencer",
    "velocity",
)
MODULATION_DESTINATIONS: tuple[str, ...] = (
    "amp_attack",
    "amp_decay",
    "amp_release",
    "amp_sustain",
    "arp_frequency",
    "arp_gate",
    "arp_octaves",
    "arp_pattern",
    "arp_tempo",
    "beats_per_minute",
    "cross_modulation",
    "cutoff",
    "delay_dry_wet",
    "delay_feedback",
    "delay_frequency",
    "delay_tempo",
    "distortion_drive",
    "distortion_mix",
    "fil_attack",
    "fil_decay",
    "fil_env_depth",
    "fil_release",
    "fil_sustain",
    "filter_blend",
    "filter_drive",
    "formant_x",
    "formant_y",
    "keytrack",
    "mod_attack",
    "mod_decay",
    "mod_release",
    "mod_sustain",
    "mono_lfo_1_amplitude",
    "mono_lfo_1_frequency",
    "mono_lfo_1_tempo",
    "mono_lfo_1_waveform",
    "mono_lfo_2_amplitude",
    "mono_lfo_2_frequency",
    "mono_lfo_2_tempo",
    "mono_lfo_2_waveform",
    "noise_volume",
    "num_steps",
    "osc_1_transpose",
    "osc_1_tune",
    "osc_1_unison_detune",
    "osc_1_unison_voices",
    "osc_1_volume",
    "osc_1_waveform",
    "osc_2_transpose",
    "osc_2_tune",
    "osc_2_unison_detune",
    "osc_2_unison_voices",
    "osc_2_volume",
    "osc_2_waveform",
    "osc_feedback_amount",
    "osc_feedback_transpose",
    "osc_feedback_tune",
    "pitch_bend_range",
    "poly_lfo_amplitude",
    "poly_lfo_frequency",
    "poly_lfo_tempo",
    "poly_lfo_waveform",
    "polyphony",
    "portamento",
    "resonance",
    "reverb_damping",
    "reverb_dry_wet",
    "reverb_feedback",
    "step_frequency",
    "step_sequencer_tempo",
    "step_smoothing",
    "stutter_frequency",
    "stutter_resample_frequency",
    "stutter_resample_tempo",
    "stutter_softness",
    "stutter_tempo",
    "sub_shuffle",
    "sub_volume",
    "sub_waveform",
    "velocity_track",
    "volume",
)
DELAY_PARAMETER_NAMES: tuple[str, ...] = (
    "delay_dry_wet",
    "delay_feedback",
    "delay_frequency",
    "delay_on",
    "delay_sync",
    "delay_tempo",
)
NON_DELAY_MODULATION_DESTINATIONS: tuple[str, ...] = tuple(
    (name for name in MODULATION_DESTINATIONS if name not in DELAY_PARAMETER_NAMES)
)
DISCRETE_CARDINALITIES: dict[str, int] = {
    "arp_octaves": 4,
    "arp_on": 2,
    "arp_pattern": 5,
    "arp_sync": 4,
    "arp_tempo": 12,
    "delay_on": 2,
    "delay_sync": 4,
    "delay_tempo": 12,
    "distortion_on": 2,
    "distortion_type": 4,
    "filter_on": 2,
    "filter_shelf": 3,
    "filter_style": 3,
    "formant_on": 2,
    "legato": 2,
    "mono_lfo_1_retrigger": 3,
    "mono_lfo_1_sync": 4,
    "mono_lfo_1_tempo": 12,
    "mono_lfo_1_waveform": 13,
    "mono_lfo_2_retrigger": 3,
    "mono_lfo_2_sync": 4,
    "mono_lfo_2_tempo": 12,
    "mono_lfo_2_waveform": 13,
    "num_steps": 32,
    "osc_1_transpose": 97,
    "osc_1_unison_voices": 8,
    "osc_1_waveform": 11,
    "osc_2_transpose": 97,
    "osc_2_unison_voices": 8,
    "osc_2_waveform": 11,
    "osc_feedback_transpose": 49,
    "pitch_bend_range": 49,
    "poly_lfo_sync": 4,
    "poly_lfo_tempo": 12,
    "poly_lfo_waveform": 13,
    "polyphony": 32,
    "portamento_type": 3,
    "reverb_on": 2,
    "step_sequencer_retrigger": 3,
    "step_sequencer_sync": 4,
    "step_sequencer_tempo": 12,
    "stutter_on": 2,
    "stutter_resample_sync": 4,
    "stutter_resample_tempo": 12,
    "stutter_sync": 4,
    "stutter_tempo": 6,
    "sub_octave": 2,
    "sub_waveform": 11,
    "unison_1_harmonize": 2,
    "unison_2_harmonize": 2,
}
BOOLEAN_PARAMETERS = frozenset(
    (name for name, cardinality in DISCRETE_CARDINALITIES.items() if cardinality == 2)
)
NOMINAL_PARAMETERS = frozenset(
    {
        "arp_pattern",
        "distortion_type",
        "filter_shelf",
        "filter_style",
        "mono_lfo_1_retrigger",
        "mono_lfo_1_sync",
        "mono_lfo_1_waveform",
        "mono_lfo_2_retrigger",
        "mono_lfo_2_sync",
        "mono_lfo_2_waveform",
        "poly_lfo_sync",
        "poly_lfo_waveform",
        "portamento_type",
        "step_sequencer_retrigger",
        "step_sequencer_sync",
        "stutter_resample_sync",
        "stutter_sync",
        "sub_waveform",
        "osc_1_waveform",
        "osc_2_waveform",
    }
)
MODULE_GATES: dict[str, tuple[str, ...]] = {
    "arp_on": ("arp_",),
    "delay_on": ("delay_",),
    "distortion_on": ("distortion_",),
    "filter_on": ("filter_", "cutoff", "resonance", "keytrack", "fil_"),
    "formant_on": ("formant_",),
    "reverb_on": ("reverb_",),
    "stutter_on": ("stutter_",),
}


@dataclass(frozen=True)
class ParameterInfo:
    index: int
    name: str
    kind: str
    cardinality: int | None
    default: float


class HelmParameterSpace:
    def __init__(
        self, schema_path: str | Path = "assets/helm_parameter_space.json"
    ) -> None:
        self.schema_path = resolve_project_path(schema_path)
        payload = json.loads(self.schema_path.read_text(encoding="utf-8"))
        raw_parameters = payload.get("parameters", [])
        self.names = tuple((str(item["name"]) for item in raw_parameters))
        if len(self.names) != 139 or len(set(self.names)) != 139:
            raise ValueError(
                f"Expected 139 unique Helm parameters in {self.schema_path}, got {len(self.names)}"
            )
        if set(DISCRETE_CARDINALITIES) - set(self.names):
            missing = sorted(set(DISCRETE_CARDINALITIES) - set(self.names))
            raise ValueError(
                f"Discrete metadata references missing Helm parameters: {missing}"
            )
        self.name_to_index = {name: index for index, name in enumerate(self.names)}
        self.defaults = np.asarray(
            [
                float(item.get("currentValue", item.get("defaultValue", 0.0)))
                for item in raw_parameters
            ],
            dtype=np.float32,
        )
        self.parameters = tuple(
            (
                ParameterInfo(
                    index=index,
                    name=name,
                    kind=self.kind(name),
                    cardinality=DISCRETE_CARDINALITIES.get(name),
                    default=float(self.defaults[index]),
                )
                for index, name in enumerate(self.names)
            )
        )
        self.route_count = len(MODULATION_SOURCES) * len(MODULATION_DESTINATIONS)
        if self.route_count != 1053:
            raise AssertionError(
                f"Helm route topology changed unexpectedly: {self.route_count}"
            )

    @staticmethod
    def kind(name: str) -> str:
        if name in BOOLEAN_PARAMETERS:
            return "boolean"
        if name in NOMINAL_PARAMETERS:
            return "nominal"
        if name in DISCRETE_CARDINALITIES:
            return "ordered"
        return "continuous"

    @staticmethod
    def quantize(name: str, value: float) -> float:
        cardinality = DISCRETE_CARDINALITIES.get(name)
        clipped = float(np.clip(value, 0.0, 1.0))
        if cardinality is None:
            return clipped
        if cardinality <= 1:
            return 0.0
        return float(round(clipped * (cardinality - 1)) / (cardinality - 1))

    @staticmethod
    def class_index(name: str, value: float) -> int:
        cardinality = DISCRETE_CARDINALITIES[name]
        return int(np.clip(round(float(value) * (cardinality - 1)), 0, cardinality - 1))

    @staticmethod
    def class_value(name: str, class_index: int) -> float:
        cardinality = DISCRETE_CARDINALITIES[name]
        return float(np.clip(class_index, 0, cardinality - 1) / (cardinality - 1))

    def canonicalize_delay_off(self, parameters: np.ndarray) -> np.ndarray:
        values = np.asarray(parameters, dtype=np.float32)
        if values.shape != (139,):
            raise ValueError(
                f"Helm parameters must have shape (139,), got {values.shape}"
            )
        result = values.copy()
        for name in DELAY_PARAMETER_NAMES:
            index = self.name_to_index[name]
            result[index] = self.defaults[index]
        result[self.name_to_index["delay_on"]] = 0.0
        return result

    @staticmethod
    def route_pair_id(source: str, destination: str) -> int:
        try:
            source_index = MODULATION_SOURCES.index(source)
            destination_index = MODULATION_DESTINATIONS.index(destination)
        except ValueError as exc:
            raise KeyError(
                f"Invalid Helm modulation pair: {source} -> {destination}"
            ) from exc
        return source_index * len(MODULATION_DESTINATIONS) + destination_index

    @staticmethod
    def route_pair(pair_id: int) -> tuple[str, str]:
        if not 0 <= int(pair_id) < 1053:
            raise IndexError(f"Helm route pair id must be in [0, 1052], got {pair_id}")
        source_index, destination_index = divmod(
            int(pair_id), len(MODULATION_DESTINATIONS)
        )
        return (
            MODULATION_SOURCES[source_index],
            MODULATION_DESTINATIONS[destination_index],
        )

    def indices(self, names: Iterable[str]) -> list[int]:
        return [self.name_to_index[name] for name in names]
