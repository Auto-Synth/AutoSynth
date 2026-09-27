from __future__ import annotations
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
import numpy as np
from .helm_parameter_spec import HelmParameterSpace
from .helm_state import HelmRoute


@dataclass(frozen=True)
class MidiNote:
    pitch: int
    velocity: int
    onset: float
    duration: float


@dataclass(frozen=True)
class MidiControl:
    kind: str
    value: int
    time: float


@dataclass(frozen=True)
class MidiPerformance:
    notes: tuple[MidiNote, ...]
    controls: tuple[MidiControl, ...]
    kind: str
    chord_name: str


_CATEGORY_CHORD_PROBABILITY = {
    "arp": 0.75,
    "keys": 0.6,
    "pad": 0.55,
    "chip": 0.2,
    "lead": 0.1,
    "bass": 0.08,
    "harsh": 0.08,
    "sfx": 0.03,
    "percussion": 0.02,
    "init": 0.2,
}
_CATEGORY_NOTE_RANGES = {
    "bass": (28, 55),
    "percussion": (36, 72),
    "sfx": (30, 88),
    "pad": (40, 79),
    "keys": (40, 84),
    "lead": (45, 88),
    "arp": (42, 84),
    "chip": (40, 88),
    "harsh": (32, 84),
    "init": (40, 79),
}
_CHORD_HINTS = ("chord", "pad", "poly", "organ", "keys", "piano", "string", "choir")
_MONO_HINTS = ("bass", "lead", "mono", "kick", "snare", "hat", "drum", "fx", "sfx")


class HelmMidiSampler:
    def __init__(
        self,
        parameter_space: HelmParameterSpace,
        *,
        note_min: int = 24,
        note_max: int = 96,
        velocity_min: int = 1,
        velocity_max: int = 127,
        duration_min: float = 0.2,
        duration_max: float = 3.0,
        controller_excitation_probability: float = 0.15,
    ) -> None:
        self.space = parameter_space
        self.note_min = int(note_min)
        self.note_max = int(note_max)
        self.velocity_min = int(velocity_min)
        self.velocity_max = int(velocity_max)
        self.duration_min = float(duration_min)
        self.duration_max = float(duration_max)
        self.controller_excitation_probability = float(
            controller_excitation_probability
        )
        if not 0 <= self.note_min <= self.note_max <= 127:
            raise ValueError("MIDI note range must be inside [0,127]")
        if not 1 <= self.velocity_min <= self.velocity_max <= 127:
            raise ValueError("MIDI velocity range must be inside [1,127]")
        if not 0.0 < self.duration_min <= self.duration_max:
            raise ValueError("Invalid MIDI duration range")

    def sample(
        self,
        rng: np.random.Generator,
        *,
        category: str,
        preset_name: str,
        parameters: np.ndarray,
        routes: Sequence[HelmRoute],
    ) -> MidiPerformance:
        category_key = category.strip().lower()
        chord_probability = float(_CATEGORY_CHORD_PROBABILITY.get(category_key, 0.2))
        lowered_name = preset_name.lower()
        if any((hint in lowered_name for hint in _CHORD_HINTS)):
            chord_probability = max(chord_probability, 0.68)
        if any((hint in lowered_name for hint in _MONO_HINTS)):
            chord_probability *= 0.25
        arp_on = self._is_on(parameters, "arp_on")
        polyphony_class = self.space.class_index(
            "polyphony", parameters[self.space.name_to_index["polyphony"]]
        )
        actual_polyphony = polyphony_class + 1
        if arp_on:
            chord_probability = max(chord_probability, 0.8)
        elif actual_polyphony <= 1:
            chord_probability = 0.0
        is_chord = bool(rng.random() < chord_probability)
        duration = self._sample_duration(rng, category_key, is_chord)
        base_velocity = self._sample_velocity(rng, category_key)
        onset = 0.01
        if is_chord:
            intervals, chord_name = self._sample_chord(rng)
            low, high = self._note_range(category_key)
            root = int(rng.integers(low, high + 1))
            pitches = self._fit_chord(root, intervals)
            notes = tuple(
                (
                    MidiNote(
                        pitch=int(pitch),
                        velocity=int(
                            np.clip(
                                base_velocity + rng.integers(-7, 8),
                                self.velocity_min,
                                self.velocity_max,
                            )
                        ),
                        onset=float(onset + rng.uniform(0.0, 0.018)),
                        duration=float(duration * rng.uniform(0.96, 1.04)),
                    )
                    for pitch in pitches
                )
            )
            kind = "chord"
        else:
            low, high = self._note_range(category_key)
            pitch = int(
                np.clip(
                    round(rng.normal((low + high) / 2.0, (high - low) / 5.0)), low, high
                )
            )
            notes = (MidiNote(pitch, base_velocity, onset, duration),)
            kind = "single_note"
            chord_name = "none"
        controls = self._sample_controls(
            rng, routes, max((n.onset + n.duration for n in notes))
        )
        return MidiPerformance(
            notes=notes, controls=controls, kind=kind, chord_name=chord_name
        )

    def _note_range(self, category: str) -> tuple[int, int]:
        low, high = _CATEGORY_NOTE_RANGES.get(category, (36, 84))
        return (max(low, self.note_min), min(high, self.note_max))

    def _sample_duration(
        self, rng: np.random.Generator, category: str, is_chord: bool
    ) -> float:
        if category in {"percussion", "arp", "chip"}:
            center = 0.65
        elif category in {"pad", "keys"} or is_chord:
            center = 1.9
        elif category == "bass":
            center = 1.0
        else:
            center = 1.25
        value = float(rng.lognormal(mean=np.log(center), sigma=0.38))
        return float(np.clip(value, self.duration_min, self.duration_max))

    def _sample_velocity(self, rng: np.random.Generator, category: str) -> int:
        center = 105 if category in {"percussion", "harsh"} else 88
        return int(
            np.clip(round(rng.normal(center, 17)), self.velocity_min, self.velocity_max)
        )

    @staticmethod
    def _sample_chord(rng: np.random.Generator) -> tuple[tuple[int, ...], str]:
        family = str(
            rng.choice(
                ("triad", "seventh", "added", "unusual"), p=(0.68, 0.22, 0.07, 0.03)
            )
        )
        choices: Mapping[str, tuple[tuple[str, tuple[int, ...], float], ...]] = {
            "triad": (
                ("major", (0, 4, 7), 0.42),
                ("minor", (0, 3, 7), 0.38),
                ("sus2", (0, 2, 7), 0.09),
                ("sus4", (0, 5, 7), 0.08),
                ("diminished", (0, 3, 6), 0.03),
            ),
            "seventh": (
                ("dominant7", (0, 4, 7, 10), 0.36),
                ("minor7", (0, 3, 7, 10), 0.32),
                ("major7", (0, 4, 7, 11), 0.24),
                ("half_dim7", (0, 3, 6, 10), 0.08),
            ),
            "added": (
                ("add9", (0, 4, 7, 14), 0.45),
                ("minor_add9", (0, 3, 7, 14), 0.3),
                ("sixth", (0, 4, 7, 9), 0.25),
            ),
            "unusual": (
                ("augmented", (0, 4, 8), 0.4),
                ("quartal", (0, 5, 10), 0.35),
                ("cluster", (0, 2, 4), 0.25),
            ),
        }
        entries = choices[family]
        probabilities = np.asarray([entry[2] for entry in entries], dtype=np.float64)
        selected = entries[
            int(rng.choice(len(entries), p=probabilities / probabilities.sum()))
        ]
        name, intervals = (selected[0], list(selected[1]))
        inversion = int(
            rng.choice(len(intervals), p=_inversion_probabilities(len(intervals)))
        )
        for i in range(inversion):
            intervals[i] += 12
        intervals.sort()
        return (tuple(intervals), f"{name}_inv{inversion}")

    def _fit_chord(self, root: int, intervals: Sequence[int]) -> tuple[int, ...]:
        pitches = [root + int(interval) for interval in intervals]
        while max(pitches) > self.note_max:
            pitches = [pitch - 12 for pitch in pitches]
        while min(pitches) < self.note_min:
            pitches = [pitch + 12 for pitch in pitches]
        return tuple(
            (int(np.clip(pitch, self.note_min, self.note_max)) for pitch in pitches)
        )

    def _sample_controls(
        self, rng: np.random.Generator, routes: Sequence[HelmRoute], end_time: float
    ) -> tuple[MidiControl, ...]:
        sources = {route.source for route in routes}
        controls: list[MidiControl] = [MidiControl("pitch_bend", 8192, 0.0)]
        for source, kind in (("mod_wheel", "mod_wheel"), ("aftertouch", "aftertouch")):
            if (
                source in sources
                and rng.random() < self.controller_excitation_probability
            ):
                value = int(rng.integers(18, 109))
                controls.append(MidiControl(kind, value, 0.0))
                controls.append(MidiControl(kind, 0, float(end_time + 0.01)))
        return tuple(controls)

    def _is_on(self, parameters: np.ndarray, name: str) -> bool:
        return bool(float(parameters[self.space.name_to_index[name]]) >= 0.5)


def _inversion_probabilities(chord_size: int) -> np.ndarray:
    if chord_size == 3:
        return np.asarray((0.62, 0.24, 0.14), dtype=np.float64)
    if chord_size == 4:
        return np.asarray((0.6, 0.2, 0.13, 0.07), dtype=np.float64)
    probabilities = np.geomspace(1.0, 0.2, chord_size)
    return probabilities / probabilities.sum()


def write_standard_midi(
    performance: MidiPerformance, path: str | Path, *, ppq: int = 480
) -> None:
    ticks_per_second = float(ppq) * 2.0
    events: list[tuple[int, int, bytes]] = []
    for control in performance.controls:
        tick = max(0, int(round(control.time * ticks_per_second)))
        if control.kind == "mod_wheel":
            message = bytes((176, 1, int(np.clip(control.value, 0, 127))))
        elif control.kind == "aftertouch":
            message = bytes((208, int(np.clip(control.value, 0, 127))))
        elif control.kind == "pitch_bend":
            value = int(np.clip(control.value, 0, 16383))
            message = bytes((224, value & 127, value >> 7 & 127))
        else:
            raise ValueError(f"Unsupported MIDI control kind: {control.kind}")
        events.append((tick, 0, message))
    for note in performance.notes:
        pitch = int(np.clip(note.pitch, 0, 127))
        velocity = int(np.clip(note.velocity, 1, 127))
        start_tick = max(0, int(round(note.onset * ticks_per_second)))
        end_tick = max(
            start_tick + 1, int(round((note.onset + note.duration) * ticks_per_second))
        )
        events.append((start_tick, 2, bytes((144, pitch, velocity))))
        events.append((end_tick, 1, bytes((128, pitch, 0))))
    events.sort(key=lambda item: (item[0], item[1]))
    track = bytearray(b"\x00\xffQ\x03\x07\xa1 ")
    previous_tick = 0
    for tick, _priority, message in events:
        track.extend(_variable_length_quantity(tick - previous_tick))
        track.extend(message)
        previous_tick = tick
    track.extend(b"\x00\xff/\x00")
    header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, int(ppq))
    payload = header + b"MTrk" + struct.pack(">I", len(track)) + bytes(track)
    Path(path).write_bytes(payload)


def _variable_length_quantity(value: int) -> bytes:
    value = max(0, int(value))
    buffer = value & 127
    output = bytearray()
    while value >> 7:
        value >>= 7
        buffer <<= 8
        buffer |= value & 127 | 128
    while True:
        output.append(buffer & 255)
        if buffer & 128:
            buffer >>= 8
        else:
            break
    return bytes(output)
