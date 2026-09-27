from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from .helm_parameter_spec import (
    DELAY_PARAMETER_NAMES,
    DISCRETE_CARDINALITIES,
    MODULE_GATES,
    MODULATION_SOURCES,
    NON_DELAY_MODULATION_DESTINATIONS,
    HelmParameterSpace,
)
from .helm_state import HelmPresetSeed, HelmRoute


@dataclass(frozen=True)
class AugmentedPatch:
    parameters: np.ndarray
    routes: tuple[HelmRoute, ...]
    base_state_payload: dict
    source_type: str
    source_preset_id: str
    source_name: str
    category: str
    role: str
    changed_parameter_mask: np.ndarray
    operations: tuple[str, ...]


class HelmAugmenter:
    def __init__(
        self,
        parameter_space: HelmParameterSpace,
        *,
        preset_probability: float = 0.9,
        edit_geometric_p: float = 0.16,
        max_parameter_edits: int = 32,
        route_edit_probability: float = 0.55,
        coverage_probability: float = 0.2,
        route_zero_threshold: float = 0.02,
        wild_parameter_strength_multiplier: float = 1.0,
    ) -> None:
        self.space = parameter_space
        self.preset_probability = float(preset_probability)
        self.edit_geometric_p = float(edit_geometric_p)
        self.max_parameter_edits = int(max_parameter_edits)
        self.route_edit_probability = float(route_edit_probability)
        self.coverage_probability = float(coverage_probability)
        self.route_zero_threshold = float(route_zero_threshold)
        self.wild_parameter_strength_multiplier = float(
            wild_parameter_strength_multiplier
        )
        self.delay_parameter_indices = tuple(
            (self.space.name_to_index[name] for name in DELAY_PARAMETER_NAMES)
        )
        self.active_modulation_destinations = NON_DELAY_MODULATION_DESTINATIONS
        if not 0.0 <= self.preset_probability <= 1.0:
            raise ValueError("preset_probability must be in [0,1]")
        if not 0.0 < self.edit_geometric_p <= 1.0:
            raise ValueError("edit_geometric_p must be in (0,1]")
        if not 1 <= self.max_parameter_edits <= 139:
            raise ValueError("max_parameter_edits must be in [1,139]")
        if not 0.0 <= self.route_edit_probability <= 1.0:
            raise ValueError("route_edit_probability must be in [0,1]")
        if self.wild_parameter_strength_multiplier < 1.0:
            raise ValueError("wild_parameter_strength_multiplier must be at least 1")

    def sample(
        self,
        rng: np.random.Generator,
        *,
        preset_seeds: Sequence[HelmPresetSeed],
        init_seed: HelmPresetSeed,
    ) -> AugmentedPatch:
        if preset_seeds and rng.random() < self.preset_probability:
            seed = preset_seeds[int(rng.integers(0, len(preset_seeds)))]
            return self.augment_preset(rng, seed)
        return self.random_patch(rng, init_seed)

    def augment_preset(
        self, rng: np.random.Generator, seed: HelmPresetSeed
    ) -> AugmentedPatch:
        parameters = np.asarray(seed.parameters, dtype=np.float32).copy()
        if parameters.shape != (139,):
            raise ValueError(
                f"Helm seed parameters must have shape (139,), got {parameters.shape}"
            )
        changed = np.zeros(139, dtype=np.uint8)
        operations: list[str] = []
        parameters, base_routes, delay_changed = self.canonicalize_delay(
            parameters, seed.routes
        )
        changed[list(delay_changed)] = 1
        if delay_changed or len(base_routes) != len(seed.routes):
            operations.append("delay:canonicalized_off")
        edit_budget = min(
            int(rng.geometric(self.edit_geometric_p)), self.max_parameter_edits
        )
        weights = self._parameter_edit_weights(parameters)
        selected = rng.choice(
            139,
            size=min(edit_budget, int(np.count_nonzero(weights))),
            replace=False,
            p=weights / weights.sum(),
        )
        step_indices = [
            self.space.name_to_index[f"step_seq_{index:02d}"] for index in range(32)
        ]
        selected_set = {int(index) for index in selected}
        if selected_set.intersection(step_indices):
            touched, operation = self._mutate_step_sequence(rng, parameters)
            changed[touched] = 1
            operations.append(operation)
            selected_set.difference_update(step_indices)
        strength_name, sigma = self._sample_strength(rng)
        for index in sorted(selected_set):
            name = self.space.names[index]
            old_value = float(parameters[index])
            parameter_sigma = self._parameter_sigma(name, sigma)
            parameters[index] = self._mutate_parameter(
                rng, name, old_value, parameter_sigma
            )
            if abs(float(parameters[index]) - old_value) > 1e-07:
                changed[index] = 1
                profile = "base" if parameter_sigma == sigma else "wild"
                operations.append(f"parameter:{name}:{strength_name}_{profile}")
            if name in MODULE_GATES and old_value < 0.5 <= float(parameters[index]):
                dependent = self._module_parameter_indices(name)
                for extra_index in rng.choice(
                    dependent,
                    size=min(len(dependent), int(rng.integers(1, 3))),
                    replace=False,
                ):
                    extra_index = int(extra_index)
                    extra_name = self.space.names[extra_index]
                    old_extra = float(parameters[extra_index])
                    extra_sigma = self._parameter_sigma(extra_name, max(sigma, 0.045))
                    parameters[extra_index] = self._mutate_parameter(
                        rng, extra_name, old_extra, extra_sigma
                    )
                    changed[extra_index] = int(
                        abs(float(parameters[extra_index]) - old_extra) > 1e-07
                    )
        routes, route_operations = self._mutate_routes(
            rng, base_routes, preset_mode=True
        )
        operations.extend(route_operations)
        return AugmentedPatch(
            parameters=self._sanitize(parameters),
            routes=routes,
            base_state_payload=dict(seed.state_payload),
            source_type="preset_augmented",
            source_preset_id=seed.preset_id,
            source_name=seed.name,
            category=seed.category,
            role="preset",
            changed_parameter_mask=changed,
            operations=tuple(operations),
        )

    def random_patch(
        self, rng: np.random.Generator, init_seed: HelmPresetSeed
    ) -> AugmentedPatch:
        parameters = np.asarray(init_seed.parameters, dtype=np.float32).copy()
        changed = np.zeros(139, dtype=np.uint8)
        coverage = bool(rng.random() < self.coverage_probability)
        role = str(
            rng.choice(
                ("bass", "lead", "pad", "keys", "pluck", "percussion", "sfx"),
                p=(0.17, 0.18, 0.17, 0.16, 0.14, 0.1, 0.08),
            )
        )
        operations = [
            f"random_role:{role}",
            f"random_mode:{('coverage' if coverage else 'musical')}",
        ]

        def set_value(name: str, value: float) -> None:
            index = self.space.name_to_index[name]
            value = self.space.quantize(name, value)
            if abs(float(parameters[index]) - value) > 1e-07:
                parameters[index] = value
                changed[index] = 1

        def set_class(name: str, class_index: int) -> None:
            set_value(name, self.space.class_value(name, class_index))

        common_waveforms = np.asarray((0, 2, 4, 5, 7), dtype=np.int64)
        waveform_1 = int(
            rng.integers(0, 11) if coverage else rng.choice(common_waveforms)
        )
        waveform_2 = int(
            rng.integers(0, 11) if coverage else rng.choice(common_waveforms)
        )
        set_class("osc_1_waveform", waveform_1)
        set_class("osc_2_waveform", waveform_2)
        set_value("osc_1_volume", float(rng.uniform(0.42, 0.82)))
        set_value(
            "osc_2_volume",
            float(rng.uniform(0.0, 0.68) if rng.random() < 0.78 else 0.0),
        )
        set_value(
            "noise_volume",
            float(rng.beta(1.2, 7.0)) if role in {"percussion", "sfx"} else 0.0,
        )
        set_value(
            "sub_volume",
            float(rng.uniform(0.12, 0.58))
            if role == "bass"
            else float(rng.uniform(0.0, 0.2)),
        )
        set_class(
            "sub_waveform",
            int(rng.integers(0, 11) if coverage else rng.choice((0, 2, 4))),
        )
        interval_choices = {
            "bass": (-12, 0, 12),
            "lead": (-12, 0, 7, 12),
            "pad": (-12, 0, 7, 12, 19),
            "keys": (-12, 0, 7, 12),
            "pluck": (0, 7, 12),
            "percussion": (-24, -12, 0, 12),
            "sfx": tuple(range(-24, 25)),
        }
        transpose = int(rng.choice(interval_choices[role]))
        set_class("osc_2_transpose", transpose + 48)
        set_value(
            "osc_2_tune",
            float(np.clip(rng.normal(0.5, 0.018 if not coverage else 0.08), 0.0, 1.0)),
        )
        envelope_ranges = {
            "bass": ((0.0, 0.1), (0.05, 0.35), (0.45, 0.9), (0.03, 0.25)),
            "lead": ((0.0, 0.16), (0.08, 0.42), (0.5, 1.0), (0.05, 0.35)),
            "pad": ((0.22, 0.62), (0.18, 0.58), (0.55, 1.0), (0.25, 0.7)),
            "keys": ((0.0, 0.12), (0.12, 0.52), (0.18, 0.72), (0.1, 0.45)),
            "pluck": ((0.0, 0.05), (0.03, 0.22), (0.0, 0.3), (0.02, 0.18)),
            "percussion": ((0.0, 0.03), (0.01, 0.15), (0.0, 0.15), (0.01, 0.12)),
            "sfx": ((0.0, 0.55), (0.02, 0.75), (0.0, 1.0), (0.02, 0.7)),
        }
        for name, bounds in zip(
            ("amp_attack", "amp_decay", "amp_sustain", "amp_release"),
            envelope_ranges[role],
        ):
            set_value(name, float(rng.uniform(*bounds)))
        polyphony = (
            1
            if role in {"bass", "lead", "pluck", "percussion"} and rng.random() < 0.58
            else int(rng.integers(4, 13))
        )
        set_class("polyphony", polyphony - 1)
        set_value("legato", float(role in {"bass", "lead"} and rng.random() < 0.35))
        set_value("velocity_track", float(np.clip(rng.normal(0.68, 0.16), 0.0, 1.0)))
        set_value("volume", float(rng.uniform(0.38, 0.62)))
        filter_probability = 0.88 if role not in {"sfx", "percussion"} else 0.58
        filter_on = bool(rng.random() < filter_probability)
        set_value("filter_on", float(filter_on))
        if filter_on:
            set_class(
                "filter_style",
                int(rng.integers(0, 3) if coverage else rng.choice((0, 0, 0, 1, 2))),
            )
            set_value("cutoff", float(rng.beta(2.6, 1.9)))
            set_value("resonance", float(rng.beta(1.5, 3.5)))
            set_value("keytrack", float(np.clip(rng.normal(0.66, 0.18), 0.0, 1.0)))
            if rng.random() < 0.55:
                set_value(
                    "fil_env_depth", float(np.clip(rng.normal(0.56, 0.16), 0.0, 1.0))
                )
        self._sample_effect_module(
            rng, set_value, "reverb", 0.45 if role in {"pad", "keys"} else 0.18
        )
        for index in self.delay_parameter_indices:
            set_value(self.space.names[index], float(self.space.defaults[index]))
        self._sample_effect_module(
            rng,
            set_value,
            "distortion",
            0.34 if role in {"bass", "lead", "percussion", "sfx"} else 0.12,
        )
        self._sample_effect_module(
            rng, set_value, "formant", 0.2 if role in {"lead", "sfx"} else 0.04
        )
        self._sample_effect_module(
            rng, set_value, "stutter", 0.15 if role in {"percussion", "sfx"} else 0.03
        )
        arp_probability = 0.18 if role in {"keys", "pluck"} else 0.04
        set_value("arp_on", float(rng.random() < arp_probability))
        if parameters[self.space.name_to_index["arp_on"]] >= 0.5:
            set_class("arp_pattern", int(rng.integers(0, 5)))
            set_class("arp_octaves", int(rng.integers(0, 3)))
            set_value("arp_gate", float(rng.uniform(0.25, 0.88)))
        common_destinations = (
            "cutoff",
            "resonance",
            "fil_env_depth",
            "osc_1_tune",
            "osc_2_tune",
            "osc_1_volume",
            "osc_2_volume",
            "cross_modulation",
            "formant_x",
            "formant_y",
            "reverb_dry_wet",
            "volume",
        )
        route_count = int(
            np.clip(rng.poisson(3.0 if coverage else 1.3), 0, 9 if coverage else 5)
        )
        routes: list[HelmRoute] = []
        used: set[tuple[str, str]] = set()
        for _ in range(route_count):
            source = str(
                rng.choice(
                    MODULATION_SOURCES
                    if coverage
                    else (
                        "fil_envelope",
                        "mod_envelope",
                        "mono_lfo_1",
                        "mono_lfo_2",
                        "poly_lfo",
                        "velocity",
                        "note",
                        "random",
                        "step_sequencer",
                    )
                )
            )
            destination = str(
                rng.choice(
                    self.active_modulation_destinations
                    if coverage
                    else common_destinations
                )
            )
            if (source, destination) in used:
                continue
            used.add((source, destination))
            magnitude = float(rng.beta(1.4, 3.0) * (1.0 if coverage else 0.65) + 0.04)
            routes.append(
                HelmRoute(
                    source,
                    destination,
                    float(np.clip(magnitude * rng.choice((-1.0, 1.0)), -1.0, 1.0)),
                )
            )
        return AugmentedPatch(
            parameters=self._sanitize(parameters),
            routes=tuple(
                sorted(routes, key=lambda route: (route.source, route.destination))
            ),
            base_state_payload=dict(init_seed.state_payload),
            source_type="random_constrained",
            source_preset_id=init_seed.preset_id,
            source_name="Init / Basic Saw",
            category=role.capitalize(),
            role=role,
            changed_parameter_mask=changed,
            operations=tuple(operations),
        )

    def _parameter_edit_weights(self, parameters: np.ndarray) -> np.ndarray:
        weights = np.ones(139, dtype=np.float64)
        for gate, prefixes in MODULE_GATES.items():
            gate_index = self.space.name_to_index[gate]
            weights[gate_index] *= 0.42
            if float(parameters[gate_index]) < 0.5:
                for index, name in enumerate(self.space.names):
                    if name != gate and any(
                        (
                            name == prefix or name.startswith(prefix)
                            for prefix in prefixes
                        )
                    ):
                        weights[index] *= 0.16
        for index, name in enumerate(self.space.names):
            if name in DELAY_PARAMETER_NAMES:
                weights[index] = 0.0
            elif name.startswith("step_seq_"):
                weights[index] *= 0.11
            elif name in {"pitch_bend_range", "beats_per_minute"}:
                weights[index] *= 0.3
        return weights

    def _module_parameter_indices(self, gate: str) -> list[int]:
        prefixes = MODULE_GATES[gate]
        return [
            index
            for index, name in enumerate(self.space.names)
            if name != gate
            and any((name == prefix or name.startswith(prefix) for prefix in prefixes))
        ]

    @staticmethod
    def _sample_strength(rng: np.random.Generator) -> tuple[str, float]:
        name = str(rng.choice(("local", "medium", "strong"), p=(0.6, 0.3, 0.1)))
        return (name, {"local": 0.018, "medium": 0.055, "strong": 0.14}[name])

    def _parameter_sigma(self, name: str, sigma: float) -> float:
        protected = name.startswith(("mono_lfo_", "poly_lfo_", "distortion_"))
        if protected or self.space.kind(name) not in {"continuous", "ordered"}:
            return sigma
        return sigma * self.wild_parameter_strength_multiplier

    def canonicalize_delay(
        self, parameters: np.ndarray, routes: Sequence[HelmRoute]
    ) -> tuple[np.ndarray, tuple[HelmRoute, ...], tuple[int, ...]]:
        result = np.asarray(parameters, dtype=np.float32).copy()
        canonical = self.space.canonicalize_delay_off(result)
        changed: list[int] = []
        for index in self.delay_parameter_indices:
            if abs(float(result[index]) - float(canonical[index])) > 1e-07:
                changed.append(index)
        result = canonical
        filtered_routes = tuple(
            sorted(
                (
                    route
                    for route in routes
                    if route.destination not in DELAY_PARAMETER_NAMES
                ),
                key=lambda route: (route.source, route.destination),
            )
        )
        return (result, filtered_routes, tuple(changed))

    def _mutate_parameter(
        self, rng: np.random.Generator, name: str, value: float, sigma: float
    ) -> float:
        kind = self.space.kind(name)
        if kind == "continuous":
            return self._reflect(float(value) + float(rng.normal(0.0, sigma)))
        if kind == "boolean":
            return 0.0 if float(value) >= 0.5 else 1.0
        cardinality = DISCRETE_CARDINALITIES[name]
        current = self.space.class_index(name, value)
        if kind == "ordered":
            class_sigma = max(1.0, sigma * cardinality * 1.8)
            candidate = int(
                np.clip(round(rng.normal(current, class_sigma)), 0, cardinality - 1)
            )
            if candidate == current:
                candidate = int(
                    np.clip(current + rng.choice((-1, 1)), 0, cardinality - 1)
                )
        else:
            candidates = np.delete(np.arange(cardinality, dtype=np.int64), current)
            candidate = int(rng.choice(candidates)) if candidates.size else current
        return self.space.class_value(name, candidate)

    def _mutate_step_sequence(
        self, rng: np.random.Generator, parameters: np.ndarray
    ) -> tuple[list[int], str]:
        indices = [
            self.space.name_to_index[f"step_seq_{index:02d}"] for index in range(32)
        ]
        num_steps = (
            self.space.class_index(
                "num_steps", parameters[self.space.name_to_index["num_steps"]]
            )
            + 1
        )
        active = np.asarray(
            [float(parameters[index]) for index in indices[:num_steps]],
            dtype=np.float32,
        )
        operation = str(
            rng.choice(
                ("offset", "scale", "rotate", "partial_regenerate"),
                p=(0.3, 0.27, 0.23, 0.2),
            )
        )
        if operation == "offset":
            active = np.asarray(
                [
                    self._reflect(float(value) + float(rng.normal(0.0, 0.08)))
                    for value in active
                ]
            )
        elif operation == "scale":
            active = np.clip(
                0.5 + (active - 0.5) * float(rng.uniform(0.65, 1.45)), 0.0, 1.0
            )
        elif operation == "rotate" and num_steps > 1:
            active = np.roll(active, int(rng.integers(1, num_steps)))
        else:
            count = max(1, int(np.ceil(num_steps * rng.uniform(0.12, 0.35))))
            selected = rng.choice(num_steps, size=count, replace=False)
            active[selected] = np.clip(rng.normal(0.5, 0.28, size=count), 0.0, 1.0)
        for index, value in zip(indices[:num_steps], active):
            parameters[index] = float(value)
        return (indices[:num_steps], f"step_sequence:{operation}")

    def _mutate_routes(
        self,
        rng: np.random.Generator,
        routes: Sequence[HelmRoute],
        *,
        preset_mode: bool,
    ) -> tuple[tuple[HelmRoute, ...], list[str]]:
        route_map = {
            (route.source, route.destination): float(route.amount)
            for route in routes
            if route.destination not in DELAY_PARAMETER_NAMES
        }
        operations: list[str] = []
        if rng.random() >= self.route_edit_probability:
            return (
                tuple(
                    sorted(routes, key=lambda route: (route.source, route.destination))
                ),
                operations,
            )
        edit_count = min(int(rng.geometric(0.55)), 5)
        for _ in range(edit_count):
            available = list(route_map)
            if not available:
                operation = "add"
            else:
                operation = str(
                    rng.choice(
                        ("tweak", "add", "delete", "rewire"), p=(0.46, 0.27, 0.12, 0.15)
                    )
                )
            if operation == "tweak":
                pair = available[int(rng.integers(0, len(available)))]
                amount = float(
                    np.clip(route_map[pair] + rng.normal(0.0, 0.1), -1.0, 1.0)
                )
                if abs(amount) < self.route_zero_threshold:
                    route_map.pop(pair)
                else:
                    route_map[pair] = amount
            elif operation == "delete":
                pair = available[int(rng.integers(0, len(available)))]
                route_map.pop(pair)
            elif operation == "rewire":
                old_pair = available[int(rng.integers(0, len(available)))]
                amount = route_map.pop(old_pair)
                source = (
                    old_pair[0]
                    if rng.random() < 0.5
                    else str(rng.choice(MODULATION_SOURCES))
                )
                destination = (
                    old_pair[1]
                    if source != old_pair[0]
                    else str(rng.choice(self.active_modulation_destinations))
                )
                route_map[source, destination] = amount
            else:
                source = str(rng.choice(MODULATION_SOURCES))
                destination = str(rng.choice(self.active_modulation_destinations))
                magnitude = float(
                    rng.beta(1.5, 3.5) * (0.7 if preset_mode else 1.0) + 0.04
                )
                route_map[source, destination] = float(
                    np.clip(magnitude * rng.choice((-1.0, 1.0)), -1.0, 1.0)
                )
            operations.append(f"route:{operation}")
        result = tuple(
            (
                HelmRoute(source, destination, amount)
                for (source, destination), amount in sorted(route_map.items())
                if abs(amount) >= self.route_zero_threshold
            )
        )
        return (result, operations)

    def _sample_effect_module(
        self, rng, set_value, module: str, probability: float
    ) -> None:
        enabled = bool(rng.random() < probability)
        set_value(f"{module}_on", float(enabled))
        if not enabled:
            return
        if module == "reverb":
            set_value("reverb_dry_wet", float(rng.beta(1.5, 4.5)))
            set_value("reverb_feedback", float(rng.uniform(0.15, 0.78)))
            set_value("reverb_damping", float(rng.uniform(0.2, 0.85)))
        elif module == "distortion":
            set_value("distortion_mix", float(rng.beta(2.0, 2.5)))
            set_value("distortion_drive", float(rng.uniform(0.35, 0.75)))
        elif module == "formant":
            set_value("formant_x", float(rng.uniform(0.08, 0.92)))
            set_value("formant_y", float(rng.uniform(0.08, 0.92)))
        elif module == "stutter":
            set_value("stutter_softness", float(rng.uniform(0.05, 0.72)))

    def _sanitize(self, parameters: np.ndarray) -> np.ndarray:
        result = np.clip(np.asarray(parameters, dtype=np.float32), 0.0, 1.0)
        for name in DISCRETE_CARDINALITIES:
            index = self.space.name_to_index[name]
            result[index] = self.space.quantize(name, float(result[index]))
        if not np.isfinite(result).all():
            raise ValueError("Generated Helm parameters contain non-finite values")
        return result

    @staticmethod
    def _reflect(value: float) -> float:
        value = float(value)
        while value < 0.0 or value > 1.0:
            value = -value if value < 0.0 else 2.0 - value
        return float(value)
