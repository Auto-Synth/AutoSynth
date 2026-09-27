from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, IntEnum
from pathlib import Path
from typing import Sequence
import numpy as np
import yaml
from autosynth.synth.helm_midi import MidiControl, MidiNote, MidiPerformance
from autosynth.synth.helm_parameter_spec import (
    DELAY_PARAMETER_NAMES,
    MODULATION_SOURCES,
    NON_DELAY_MODULATION_DESTINATIONS,
    HelmParameterSpace,
)
from autosynth.synth.helm_state import HelmRoute
from autosynth.paths import resolve_project_path

DEFAULT_HELM_PARAMETER_ORDER_PATH = Path("assets/helm_order.yaml")


def load_helm_parameter_order(
    path: str | Path = DEFAULT_HELM_PARAMETER_ORDER_PATH,
) -> tuple[str, ...]:
    resolved_path = resolve_project_path(path)
    data = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(
            f"Helm parameter order must contain a YAML list: {resolved_path}"
        )
    order = tuple((str(name).strip() for name in data))
    if any((not name for name in order)):
        raise ValueError(
            f"Helm parameter order contains an empty name: {resolved_path}"
        )
    return order


class HelmTokenKind(IntEnum):
    BOS = 0
    NOTE_OR_END = 1
    MIDI_PITCH = 2
    MIDI_VELOCITY = 3
    MIDI_ONSET = 4
    MIDI_DURATION = 5
    CONTROL_OR_END = 6
    CONTROL_VALUE = 7
    CONTROL_TIME = 8
    PARAMETER = 9
    ROUTE_SOURCE_OR_EOS = 10
    ROUTE_DESTINATION = 11
    ROUTE_AMOUNT = 12


class HelmLossKind(str, Enum):
    HARD = "hard"
    GAUSSIAN = "gaussian"


@dataclass(frozen=True)
class HelmTokenTargetSpec:
    cardinality: int
    loss_kind: HelmLossKind


@dataclass(frozen=True)
class HelmNextTokenSpec:
    kind: HelmTokenKind
    field_id: int
    allowed_value_ids: tuple[int, ...]


@dataclass
class HelmGenerationState:
    phase: str = "bos"
    note_count: int = 0
    control_count: int = 0
    parameter_position: int = 0
    route_count: int = 0
    previous_route_pair_id: int = -1
    pending_route_source: int = -1


@dataclass(frozen=True)
class HelmTokenSequence:
    kind_ids: tuple[int, ...]
    value_ids: tuple[int, ...]
    field_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        kinds = tuple((int(value) for value in self.kind_ids))
        values = tuple((int(value) for value in self.value_ids))
        fields = tuple((int(value) for value in self.field_ids))
        if not kinds:
            raise ValueError("A Helm token sequence cannot be empty")
        if not len(kinds) == len(values) == len(fields):
            raise ValueError(
                f"kind_ids, value_ids, and field_ids must have identical lengths, got {len(kinds)}, {len(values)}, and {len(fields)}"
            )
        object.__setattr__(self, "kind_ids", kinds)
        object.__setattr__(self, "value_ids", values)
        object.__setattr__(self, "field_ids", fields)

    def __len__(self) -> int:
        return len(self.kind_ids)

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.asarray(self.kind_ids, dtype=np.int16),
            np.asarray(self.value_ids, dtype=np.int16),
            np.asarray(self.field_ids, dtype=np.int16),
        )


@dataclass(frozen=True)
class DecodedHelmSequence:
    parameters: np.ndarray
    routes: tuple[HelmRoute, ...]
    performance: MidiPerformance


class HelmAutoregressiveTokenizer:
    CONTINUOUS_CARDINALITY = 128
    NOTE_END = 0
    NOTE = 1
    CONTROL_END = 0
    MOD_WHEEL = 1
    AFTERTOUCH = 2
    ROUTE_EOS = len(MODULATION_SOURCES)
    NO_FIELD = -1
    MAX_NOTES = 4
    MAX_CONTROLS = 4
    MAX_ROUTES = 13
    MAX_SEQUENCE_LENGTH = (
        1 + MAX_NOTES * 5 + 1 + MAX_CONTROLS * 3 + 1 + 133 + MAX_ROUTES * 3 + 1
    )
    _CONTROL_KIND_TO_ID = {"mod_wheel": MOD_WHEEL, "aftertouch": AFTERTOUCH}
    _CONTROL_ID_TO_KIND = {MOD_WHEEL: "mod_wheel", AFTERTOUCH: "aftertouch"}
    _STATIC_TARGET_SPECS = {
        HelmTokenKind.BOS: HelmTokenTargetSpec(1, HelmLossKind.HARD),
        HelmTokenKind.NOTE_OR_END: HelmTokenTargetSpec(2, HelmLossKind.HARD),
        HelmTokenKind.MIDI_PITCH: HelmTokenTargetSpec(128, HelmLossKind.GAUSSIAN),
        HelmTokenKind.MIDI_VELOCITY: HelmTokenTargetSpec(128, HelmLossKind.GAUSSIAN),
        HelmTokenKind.MIDI_ONSET: HelmTokenTargetSpec(128, HelmLossKind.GAUSSIAN),
        HelmTokenKind.MIDI_DURATION: HelmTokenTargetSpec(128, HelmLossKind.GAUSSIAN),
        HelmTokenKind.CONTROL_OR_END: HelmTokenTargetSpec(3, HelmLossKind.HARD),
        HelmTokenKind.CONTROL_VALUE: HelmTokenTargetSpec(128, HelmLossKind.GAUSSIAN),
        HelmTokenKind.CONTROL_TIME: HelmTokenTargetSpec(128, HelmLossKind.GAUSSIAN),
        HelmTokenKind.ROUTE_SOURCE_OR_EOS: HelmTokenTargetSpec(
            len(MODULATION_SOURCES) + 1, HelmLossKind.HARD
        ),
        HelmTokenKind.ROUTE_DESTINATION: HelmTokenTargetSpec(
            len(NON_DELAY_MODULATION_DESTINATIONS), HelmLossKind.HARD
        ),
        HelmTokenKind.ROUTE_AMOUNT: HelmTokenTargetSpec(
            CONTINUOUS_CARDINALITY, HelmLossKind.GAUSSIAN
        ),
    }

    def __init__(
        self,
        parameter_space: HelmParameterSpace | None = None,
        *,
        parameter_order: Sequence[str] | None = None,
        target_duration_seconds: float = 5.0,
    ) -> None:
        self.space = parameter_space or HelmParameterSpace()
        self.target_duration_seconds = float(target_duration_seconds)
        if (
            not np.isfinite(self.target_duration_seconds)
            or self.target_duration_seconds <= 0.0
        ):
            raise ValueError("target_duration_seconds must be finite and positive")
        if len(self.space.parameters) != 139:
            raise ValueError(
                f"Expected 139 fixed Helm parameters, got {len(self.space.parameters)}"
            )
        self.predicted_parameter_names = tuple(
            (name for name in self.space.names if name not in DELAY_PARAMETER_NAMES)
        )
        self.predicted_parameter_indices = frozenset(
            (self.space.name_to_index[name] for name in self.predicted_parameter_names)
        )
        if len(self.predicted_parameter_names) != 133:
            raise ValueError(
                f"Expected 133 non-Delay Helm parameters, got {len(self.predicted_parameter_names)}"
            )
        self.parameter_order = self._validate_parameter_order(
            load_helm_parameter_order() if parameter_order is None else parameter_order
        )
        self.parameters_in_order = tuple(
            (
                self.space.parameters[self.space.name_to_index[name]]
                for name in self.parameter_order
            )
        )

    def target_spec(
        self, kind: HelmTokenKind | int, field_id: int = NO_FIELD
    ) -> HelmTokenTargetSpec:
        token_kind = self._coerce_kind(kind)
        field_id = int(field_id)
        if token_kind == HelmTokenKind.PARAMETER:
            if not 0 <= field_id < len(self.space.parameters):
                raise ValueError(
                    f"Parameter token requires field_id in [0,138], got {field_id}"
                )
            if field_id not in self.predicted_parameter_indices:
                parameter_name = self.space.parameters[field_id].name
                raise ValueError(f"{parameter_name} is not a predicted Helm parameter")
            parameter = self.space.parameters[field_id]
            cardinality = (
                self.CONTINUOUS_CARDINALITY
                if parameter.kind == "continuous"
                else int(parameter.cardinality or 0)
            )
            loss_kind = (
                HelmLossKind.GAUSSIAN
                if parameter.kind in {"continuous", "ordered"}
                else HelmLossKind.HARD
            )
            return HelmTokenTargetSpec(cardinality=cardinality, loss_kind=loss_kind)
        if field_id != self.NO_FIELD:
            raise ValueError(
                f"{token_kind.name} token must use field_id={self.NO_FIELD}, got {field_id}"
            )
        return self._STATIC_TARGET_SPECS[token_kind]

    def encode(
        self,
        parameters: np.ndarray | Sequence[float],
        routes: Sequence[HelmRoute],
        performance: MidiPerformance,
    ) -> HelmTokenSequence:
        values = np.asarray(parameters, dtype=np.float64)
        if values.shape != (139,):
            raise ValueError(
                f"Helm parameters must have shape (139,), got {values.shape}"
            )
        if (
            not np.isfinite(values).all()
            or np.any(values < -1e-06)
            or np.any(values > 1.0 + 1e-06)
        ):
            raise ValueError(
                "Helm parameters must be finite normalized values in [0,1]"
            )
        values = np.clip(values, 0.0, 1.0)
        kind_ids: list[int] = []
        value_ids: list[int] = []
        field_ids: list[int] = []

        def append(
            kind: HelmTokenKind, value: int, field_id: int = self.NO_FIELD
        ) -> None:
            spec = self.target_spec(kind, field_id)
            value = int(value)
            if not 0 <= value < spec.cardinality:
                raise ValueError(
                    f"{kind.name} value must be in [0,{spec.cardinality - 1}], got {value}"
                )
            kind_ids.append(int(kind))
            value_ids.append(value)
            field_ids.append(int(field_id))

        append(HelmTokenKind.BOS, 0)

        def append_parameters() -> None:
            for parameter in self.parameters_in_order:
                value = float(values[parameter.index])
                class_id = (
                    self._quantize_unit(value, parameter.name)
                    if parameter.kind == "continuous"
                    else self.space.class_index(parameter.name, value)
                )
                append(HelmTokenKind.PARAMETER, class_id, parameter.index)

        notes = tuple(
            sorted(
                performance.notes,
                key=lambda note: (
                    float(note.onset),
                    int(note.pitch),
                    int(note.velocity),
                    float(note.duration),
                ),
            )
        )
        if not notes:
            raise ValueError("A Helm performance must contain at least one MIDI note")
        if len(notes) > self.MAX_NOTES:
            raise ValueError(
                f"Helm AR supports at most {self.MAX_NOTES} simultaneous notes, got {len(notes)}"
            )
        for note in notes:
            append(HelmTokenKind.NOTE_OR_END, self.NOTE)
            append(HelmTokenKind.MIDI_PITCH, self._midi_pitch(note.pitch))
            append(HelmTokenKind.MIDI_VELOCITY, self._midi_velocity(note.velocity))
            append(
                HelmTokenKind.MIDI_ONSET, self._quantize_time(note.onset, "MIDI onset")
            )
            if not np.isfinite(note.duration) or float(note.duration) <= 0.0:
                raise ValueError(
                    f"MIDI duration must be finite and positive, got {note.duration}"
                )
            append(
                HelmTokenKind.MIDI_DURATION,
                self._quantize_time(note.duration, "MIDI duration"),
            )
        append(HelmTokenKind.NOTE_OR_END, self.NOTE_END)
        controls: list[MidiControl] = []
        for control in performance.controls:
            if control.kind == "pitch_bend":
                if int(control.value) != 8192 or abs(float(control.time)) > 1e-06:
                    raise ValueError(
                        "The Helm AR format omits pitch bend only when it is neutral (8192 at t=0)"
                    )
                continue
            if control.kind not in self._CONTROL_KIND_TO_ID:
                raise ValueError(f"Unsupported Helm MIDI control kind: {control.kind}")
            controls.append(control)
        controls.sort(
            key=lambda control: (
                float(control.time),
                str(control.kind),
                int(control.value),
            )
        )
        if len(controls) > self.MAX_CONTROLS:
            raise ValueError(
                f"Helm AR supports at most {self.MAX_CONTROLS} learned controls, got {len(controls)}"
            )
        for control in controls:
            append(HelmTokenKind.CONTROL_OR_END, self._CONTROL_KIND_TO_ID[control.kind])
            append(HelmTokenKind.CONTROL_VALUE, self._midi_control_value(control.value))
            append(
                HelmTokenKind.CONTROL_TIME,
                self._quantize_time(control.time, "MIDI control time"),
            )
        append(HelmTokenKind.CONTROL_OR_END, self.CONTROL_END)
        append_parameters()
        for route in routes:
            if route.destination not in NON_DELAY_MODULATION_DESTINATIONS:
                if route.destination in DELAY_PARAMETER_NAMES:
                    raise ValueError(
                        f"Delay modulation routes are not part of the Helm model target: {route.source} -> {route.destination}"
                    )
                raise ValueError(
                    f"Invalid Helm modulation destination: {route.destination}"
                )
            if route.source not in MODULATION_SOURCES:
                raise ValueError(f"Invalid Helm modulation source: {route.source}")
        canonical_routes = sorted(
            routes,
            key=lambda route: (
                MODULATION_SOURCES.index(route.source),
                NON_DELAY_MODULATION_DESTINATIONS.index(route.destination),
            ),
        )
        if len(canonical_routes) > self.MAX_ROUTES:
            raise ValueError(
                f"Helm AR supports at most {self.MAX_ROUTES} routes, got {len(canonical_routes)}"
            )
        seen_pair_ids: set[int] = set()
        for route in canonical_routes:
            source_id = MODULATION_SOURCES.index(route.source)
            destination_id = NON_DELAY_MODULATION_DESTINATIONS.index(route.destination)
            pair_id = (
                source_id * len(NON_DELAY_MODULATION_DESTINATIONS) + destination_id
            )
            if pair_id in seen_pair_ids:
                raise ValueError(
                    f"Duplicate Helm route: {route.source} -> {route.destination}"
                )
            seen_pair_ids.add(pair_id)
            append(HelmTokenKind.ROUTE_SOURCE_OR_EOS, source_id)
            append(HelmTokenKind.ROUTE_DESTINATION, destination_id)
            append(HelmTokenKind.ROUTE_AMOUNT, self._quantize_amount(route.amount))
        append(HelmTokenKind.ROUTE_SOURCE_OR_EOS, self.ROUTE_EOS)
        return HelmTokenSequence(tuple(kind_ids), tuple(value_ids), tuple(field_ids))

    def initial_generation_state(self) -> HelmGenerationState:
        return HelmGenerationState()

    def next_token_spec(self, state: HelmGenerationState) -> HelmNextTokenSpec | None:
        phase = state.phase
        kind: HelmTokenKind
        field_id = self.NO_FIELD
        allowed: tuple[int, ...]
        if phase == "complete":
            return None
        if phase == "bos":
            kind, allowed = (HelmTokenKind.BOS, (0,))
        elif phase == "note_marker":
            kind = HelmTokenKind.NOTE_OR_END
            if state.note_count == 0:
                allowed = (self.NOTE,)
            elif state.note_count >= self.MAX_NOTES:
                allowed = (self.NOTE_END,)
            else:
                allowed = (self.NOTE_END, self.NOTE)
        elif phase == "note_pitch":
            kind, allowed = (HelmTokenKind.MIDI_PITCH, tuple(range(128)))
        elif phase == "note_velocity":
            kind, allowed = (HelmTokenKind.MIDI_VELOCITY, tuple(range(1, 128)))
        elif phase == "note_onset":
            kind, allowed = (HelmTokenKind.MIDI_ONSET, tuple(range(128)))
        elif phase == "note_duration":
            kind = HelmTokenKind.MIDI_DURATION
            allowed = tuple(range(1, 128))
        elif phase == "control_marker":
            kind = HelmTokenKind.CONTROL_OR_END
            allowed = (
                (self.CONTROL_END,)
                if state.control_count >= self.MAX_CONTROLS
                else (self.CONTROL_END, self.MOD_WHEEL, self.AFTERTOUCH)
            )
        elif phase == "control_value":
            kind, allowed = (HelmTokenKind.CONTROL_VALUE, tuple(range(128)))
        elif phase == "control_time":
            kind, allowed = (HelmTokenKind.CONTROL_TIME, tuple(range(128)))
        elif phase == "parameter":
            parameter = self.parameters_in_order[state.parameter_position]
            kind = HelmTokenKind.PARAMETER
            field_id = parameter.index
            allowed = tuple(range(self.target_spec(kind, field_id).cardinality))
        elif phase == "route_source":
            kind = HelmTokenKind.ROUTE_SOURCE_OR_EOS
            if state.route_count >= self.MAX_ROUTES:
                allowed = (self.ROUTE_EOS,)
            else:
                destination_count = len(NON_DELAY_MODULATION_DESTINATIONS)
                sources = tuple(
                    (
                        source_id
                        for source_id in range(len(MODULATION_SOURCES))
                        if source_id * destination_count + destination_count - 1
                        > state.previous_route_pair_id
                    )
                )
                allowed = sources + (self.ROUTE_EOS,)
        elif phase == "route_destination":
            kind = HelmTokenKind.ROUTE_DESTINATION
            destination_count = len(NON_DELAY_MODULATION_DESTINATIONS)
            allowed = tuple(
                (
                    destination_id
                    for destination_id in range(destination_count)
                    if state.pending_route_source * destination_count + destination_id
                    > state.previous_route_pair_id
                )
            )
        elif phase == "route_amount":
            kind, allowed = (HelmTokenKind.ROUTE_AMOUNT, tuple(range(128)))
        else:
            raise RuntimeError(f"Unknown Helm generation phase: {phase}")
        return HelmNextTokenSpec(
            kind=kind, field_id=field_id, allowed_value_ids=allowed
        )

    def advance_generation(self, state: HelmGenerationState, value_id: int) -> None:
        spec = self.next_token_spec(state)
        if spec is None:
            raise ValueError("Cannot append a token after Helm ROUTE_EOS")
        value_id = int(value_id)
        if value_id not in spec.allowed_value_ids:
            raise ValueError(
                f"Illegal {spec.kind.name} value {value_id} in phase {state.phase}; allowed={spec.allowed_value_ids}"
            )
        phase = state.phase
        if phase == "bos":
            state.phase = "note_marker"
        elif phase == "note_marker":
            state.phase = (
                "control_marker" if value_id == self.NOTE_END else "note_pitch"
            )
        elif phase == "note_pitch":
            state.phase = "note_velocity"
        elif phase == "note_velocity":
            state.phase = "note_onset"
        elif phase == "note_onset":
            state.phase = "note_duration"
        elif phase == "note_duration":
            state.note_count += 1
            state.phase = "note_marker"
        elif phase == "control_marker":
            state.phase = (
                "parameter" if value_id == self.CONTROL_END else "control_value"
            )
        elif phase == "control_value":
            state.phase = "control_time"
        elif phase == "control_time":
            state.control_count += 1
            state.phase = "control_marker"
        elif phase == "parameter":
            state.parameter_position += 1
            state.phase = (
                "route_source"
                if state.parameter_position == len(self.parameters_in_order)
                else "parameter"
            )
        elif phase == "route_source":
            if value_id == self.ROUTE_EOS:
                state.phase = "complete"
            else:
                state.pending_route_source = value_id
                state.phase = "route_destination"
        elif phase == "route_destination":
            state.previous_route_pair_id = (
                state.pending_route_source * len(NON_DELAY_MODULATION_DESTINATIONS)
                + value_id
            )
            state.phase = "route_amount"
        elif phase == "route_amount":
            state.route_count += 1
            state.pending_route_source = -1
            state.phase = "route_source"

    def decode(self, sequence: HelmTokenSequence) -> DecodedHelmSequence:
        cursor = 0

        def consume(
            expected_kind: HelmTokenKind, expected_field: int = self.NO_FIELD
        ) -> int:
            nonlocal cursor
            if cursor >= len(sequence):
                raise ValueError(f"Sequence ended while expecting {expected_kind.name}")
            try:
                actual_kind = HelmTokenKind(sequence.kind_ids[cursor])
            except ValueError as exc:
                raise ValueError(
                    f"Unknown token kind id at position {cursor}: {sequence.kind_ids[cursor]}"
                ) from exc
            field_id = int(sequence.field_ids[cursor])
            value = int(sequence.value_ids[cursor])
            if actual_kind != expected_kind:
                raise ValueError(
                    f"Expected {expected_kind.name} at position {cursor}, got {actual_kind.name}"
                )
            if field_id != expected_field:
                raise ValueError(
                    f"Expected field_id={expected_field} at position {cursor}, got {field_id}"
                )
            cardinality = self.target_spec(actual_kind, field_id).cardinality
            if not 0 <= value < cardinality:
                raise ValueError(
                    f"{actual_kind.name} value at position {cursor} must be in [0,{cardinality - 1}], got {value}"
                )
            cursor += 1
            return value

        if consume(HelmTokenKind.BOS) != 0:
            raise ValueError("BOS token value must be zero")
        parameters = self.space.canonicalize_delay_off(self.space.defaults)

        def consume_parameters() -> None:
            for parameter in self.parameters_in_order:
                class_id = consume(HelmTokenKind.PARAMETER, parameter.index)
                parameters[parameter.index] = (
                    self._dequantize_unit(class_id)
                    if parameter.kind == "continuous"
                    else self.space.class_value(parameter.name, class_id)
                )

        notes: list[MidiNote] = []
        while True:
            marker = consume(HelmTokenKind.NOTE_OR_END)
            if marker == self.NOTE_END:
                break
            if marker != self.NOTE:
                raise ValueError(f"Invalid MIDI note marker: {marker}")
            pitch = consume(HelmTokenKind.MIDI_PITCH)
            velocity = consume(HelmTokenKind.MIDI_VELOCITY)
            onset = self._dequantize_time(consume(HelmTokenKind.MIDI_ONSET))
            duration = self._dequantize_time(consume(HelmTokenKind.MIDI_DURATION))
            notes.append(MidiNote(pitch, velocity, onset, duration))
        if not notes:
            raise ValueError("Decoded Helm sequence contains no MIDI notes")
        controls: list[MidiControl] = [MidiControl("pitch_bend", 8192, 0.0)]
        while True:
            marker = consume(HelmTokenKind.CONTROL_OR_END)
            if marker == self.CONTROL_END:
                break
            kind = self._CONTROL_ID_TO_KIND.get(marker)
            if kind is None:
                raise ValueError(f"Invalid MIDI control marker: {marker}")
            value = consume(HelmTokenKind.CONTROL_VALUE)
            time = self._dequantize_time(consume(HelmTokenKind.CONTROL_TIME))
            controls.append(MidiControl(kind, value, time))
        consume_parameters()
        routes: list[HelmRoute] = []
        previous_pair_id = -1
        while True:
            source_id = consume(HelmTokenKind.ROUTE_SOURCE_OR_EOS)
            if source_id == self.ROUTE_EOS:
                break
            destination_id = consume(HelmTokenKind.ROUTE_DESTINATION)
            amount_id = consume(HelmTokenKind.ROUTE_AMOUNT)
            pair_id = (
                source_id * len(NON_DELAY_MODULATION_DESTINATIONS) + destination_id
            )
            if pair_id <= previous_pair_id:
                raise ValueError(
                    "Helm routes must be unique and strictly ordered by source-destination pair"
                )
            previous_pair_id = pair_id
            routes.append(
                HelmRoute(
                    MODULATION_SOURCES[source_id],
                    NON_DELAY_MODULATION_DESTINATIONS[destination_id],
                    self._dequantize_amount(amount_id),
                )
            )
        if cursor != len(sequence):
            raise ValueError(
                f"Sequence contains {len(sequence) - cursor} trailing token(s) after ROUTE_EOS"
            )
        performance = MidiPerformance(
            notes=tuple(notes),
            controls=tuple(controls),
            kind="single_note" if len(notes) == 1 else "chord",
            chord_name="none" if len(notes) == 1 else "decoded",
        )
        return DecodedHelmSequence(parameters, tuple(routes), performance)

    def _quantize_time(self, value: float, label: str) -> int:
        value = float(value)
        if (
            not np.isfinite(value)
            or value < -1e-06
            or value > self.target_duration_seconds + 1e-06
        ):
            raise ValueError(
                f"{label} must be inside [0,{self.target_duration_seconds}], got {value}"
            )
        return self._quantize_unit(
            float(np.clip(value, 0.0, self.target_duration_seconds))
            / self.target_duration_seconds,
            label,
        )

    def _dequantize_time(self, class_id: int) -> float:
        return self._dequantize_unit(class_id) * self.target_duration_seconds

    def _quantize_amount(self, amount: float) -> int:
        amount = float(amount)
        if not np.isfinite(amount) or amount < -1.0 or amount > 1.0:
            raise ValueError(
                f"Route amount must be finite and inside [-1,1], got {amount}"
            )
        return self._quantize_unit((amount + 1.0) * 0.5, "route amount")

    def _dequantize_amount(self, class_id: int) -> float:
        return self._dequantize_unit(class_id) * 2.0 - 1.0

    def _quantize_unit(self, value: float, label: str) -> int:
        value = float(value)
        if not np.isfinite(value) or value < -1e-06 or value > 1.0 + 1e-06:
            raise ValueError(
                f"{label} must be finite and normalized to [0,1], got {value}"
            )
        return int(np.clip(round(float(np.clip(value, 0.0, 1.0)) * 127.0), 0, 127))

    @staticmethod
    def _dequantize_unit(class_id: int) -> float:
        class_id = int(class_id)
        if not 0 <= class_id < 128:
            raise ValueError(f"128-bin class id must be in [0,127], got {class_id}")
        return class_id / 127.0

    @staticmethod
    def _midi_pitch(value: int) -> int:
        value = int(value)
        if not 0 <= value <= 127:
            raise ValueError(f"MIDI pitch must be in [0,127], got {value}")
        return value

    @staticmethod
    def _midi_velocity(value: int) -> int:
        value = int(value)
        if not 1 <= value <= 127:
            raise ValueError(f"MIDI velocity must be in [1,127], got {value}")
        return value

    @staticmethod
    def _midi_control_value(value: int) -> int:
        value = int(value)
        if not 0 <= value <= 127:
            raise ValueError(f"MIDI control value must be in [0,127], got {value}")
        return value

    def _validate_parameter_order(
        self, parameter_order: Sequence[str]
    ) -> tuple[str, ...]:
        order = tuple((str(name).strip() for name in parameter_order))
        duplicates = sorted({name for name in order if order.count(name) > 1})
        expected = set(self.predicted_parameter_names)
        actual = set(order)
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        if (
            len(order) != len(self.predicted_parameter_names)
            or duplicates
            or missing
            or unknown
        ):
            details = [
                f"expected {len(self.predicted_parameter_names)} names, got {len(order)}"
            ]
            if duplicates:
                details.append(f"duplicates={duplicates}")
            if missing:
                details.append(f"missing={missing}")
            if unknown:
                details.append(f"unknown={unknown}")
            raise ValueError("Invalid Helm parameter order: " + "; ".join(details))
        return order

    @staticmethod
    def _coerce_kind(kind: HelmTokenKind | int) -> HelmTokenKind:
        try:
            return HelmTokenKind(int(kind))
        except ValueError as exc:
            raise ValueError(f"Unknown Helm token kind: {kind}") from exc
