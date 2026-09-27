from __future__ import annotations
import bisect
import json
from pathlib import Path
from typing import Any
import h5py
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from autosynth.synth.helm_midi import MidiControl, MidiNote, MidiPerformance
from autosynth.synth.helm_parameter_spec import HelmParameterSpace
from autosynth.synth.helm_state import HelmRoute, ROUTE_AMOUNT_ENCODING
from autosynth.tokens import (
    HelmAutoregressiveTokenizer,
    HelmTokenKind,
    load_helm_parameter_order,
)
from autosynth.paths import resolve_project_path

_CONTROL_KIND_BY_ID = {1: "mod_wheel", 2: "aftertouch", 3: "pitch_bend"}


class HelmH5Dataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        token_order_path: str | Path,
        read_audio: bool = False,
    ) -> None:
        super().__init__()
        self.dataset_root = resolve_project_path(dataset_root)
        self.split = str(split)
        self.read_audio = bool(read_audio)
        self._files: dict[int, h5py.File] = {}
        metadata_path = self.dataset_root / "dataset_metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Helm dataset metadata not found: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("route_amount_encoding") != ROUTE_AMOUNT_ENCODING:
            raise ValueError("Incompatible route encoding; regenerate the Helm dataset")
        if metadata.get("split_rng_version") != 1:
            raise ValueError("Incompatible split random streams; regenerate the Helm dataset")
        if str(metadata.get("synth", "")).lower() != "helm":
            raise ValueError(f"Expected Helm dataset metadata at {metadata_path}")
        schema_path = metadata.get("parameter_schema")
        if not isinstance(schema_path, str) or not schema_path:
            raise ValueError(
                "Helm dataset metadata must define string `parameter_schema`."
            )
        duration = metadata.get("target_duration")
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            raise ValueError(
                "Helm dataset metadata must define numeric `target_duration`."
            )
        self.sample_rate = int(metadata["sample_rate"])
        self.target_duration = float(duration)
        self.parameter_space = HelmParameterSpace(resolve_project_path(schema_path))
        self.tokenizer = HelmAutoregressiveTokenizer(
            self.parameter_space,
            parameter_order=load_helm_parameter_order(token_order_path),
            target_duration_seconds=self.target_duration,
        )
        self.shard_paths = sorted(
            (self.dataset_root / "shards").glob(f"{self.split}-*-of-*.h5")
        )
        if not self.shard_paths:
            raise FileNotFoundError(
                f"No Helm shards found for split '{self.split}' under {self.dataset_root / 'shards'}"
            )
        self._shard_lengths: list[int] = []
        self._cumulative_lengths: list[int] = []
        total = 0
        for path in self.shard_paths:
            length = self._validate_shard(path)
            self._shard_lengths.append(length)
            total += length
            self._cumulative_lengths.append(total)
        self._length = total

    def _validate_shard(self, path: Path) -> int:
        required = {
            "parameters",
            "mel_spec",
            "route_pair_ids",
            "route_amounts",
            "route_offsets",
            "midi_note_events",
            "midi_note_offsets",
            "midi_control_events",
            "midi_control_offsets",
            "performance_type",
            "chord_name",
        }
        if self.read_audio:
            required.add("audio")
        with h5py.File(path, "r") as h5:
            if h5.attrs.get("route_amount_encoding") != ROUTE_AMOUNT_ENCODING:
                raise ValueError(f"Incompatible route encoding in shard {path}; regenerate it")
            if h5.attrs.get("split_rng_version") != 1:
                raise ValueError(f"Incompatible split random streams in shard {path}; regenerate it")
            missing = sorted(required.difference(h5.keys()))
            if missing:
                raise KeyError(f"Helm shard {path} is missing datasets: {missing}")
            if str(h5.attrs.get("split", "")) != self.split:
                raise ValueError(
                    f"Helm shard {path} does not belong to split '{self.split}'"
                )
            sample_count = int(h5["parameters"].shape[0])
            if h5["parameters"].shape != (sample_count, 139):
                raise ValueError(
                    f"Expected Helm parameters shape ({sample_count},139), got {h5['parameters'].shape} in {path}"
                )
            if h5["mel_spec"].shape[:2] != (sample_count, 2):
                raise ValueError(f"Expected stereo Helm mel features in {path}")
            for offsets_name in (
                "route_offsets",
                "midi_note_offsets",
                "midi_control_offsets",
            ):
                if h5[offsets_name].shape != (sample_count + 1,):
                    raise ValueError(
                        f"Expected {offsets_name} shape ({sample_count + 1},), got {h5[offsets_name].shape} in {path}"
                    )
            return sample_count

    def __len__(self) -> int:
        return self._length

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += self._length
        if not 0 <= index < self._length:
            raise IndexError(index)
        shard_index = bisect.bisect_right(self._cumulative_lengths, index)
        shard_start = (
            0 if shard_index == 0 else self._cumulative_lengths[shard_index - 1]
        )
        return (shard_index, index - shard_start)

    def _file(self, shard_index: int) -> h5py.File:
        h5 = self._files.get(shard_index)
        if h5 is None:
            h5 = h5py.File(self.shard_paths[shard_index], "r")
            self._files[shard_index] = h5
        return h5

    @staticmethod
    def _slice_packed(
        h5: h5py.File, values_name: str, offsets_name: str, index: int
    ) -> np.ndarray:
        start = int(h5[offsets_name][index])
        end = int(h5[offsets_name][index + 1])
        return np.asarray(h5[values_name][start:end])

    @staticmethod
    def _text(value: Any) -> str:
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        shard_index, local_index = self._locate(index)
        h5 = self._file(shard_index)
        parameters = np.asarray(h5["parameters"][local_index], dtype=np.float32)
        pair_ids = self._slice_packed(
            h5, "route_pair_ids", "route_offsets", local_index
        )
        amounts = self._slice_packed(h5, "route_amounts", "route_offsets", local_index)
        routes = tuple(
            (
                HelmRoute(*self.parameter_space.route_pair(int(pair_id)), float(amount))
                for pair_id, amount in zip(pair_ids, amounts, strict=True)
            )
        )
        note_rows = self._slice_packed(
            h5, "midi_note_events", "midi_note_offsets", local_index
        )
        notes = tuple(
            (
                MidiNote(int(row[0]), int(row[1]), float(row[2]), float(row[3]))
                for row in note_rows
            )
        )
        control_rows = self._slice_packed(
            h5, "midi_control_events", "midi_control_offsets", local_index
        )
        controls = tuple(
            (
                MidiControl(
                    _CONTROL_KIND_BY_ID[int(row[0])], int(row[1]), float(row[2])
                )
                for row in control_rows
            )
        )
        performance = MidiPerformance(
            notes=notes,
            controls=controls,
            kind=self._text(h5["performance_type"][local_index]),
            chord_name=self._text(h5["chord_name"][local_index]),
        )
        sequence = self.tokenizer.encode(parameters, routes, performance)
        kind_ids, value_ids, field_ids = sequence.as_arrays()
        sample = {
            "full_parameters_backend": torch.from_numpy(parameters),
            "token_kind_ids": torch.from_numpy(kind_ids.astype(np.int64)),
            "token_value_ids": torch.from_numpy(value_ids.astype(np.int64)),
            "token_field_ids": torch.from_numpy(field_ids.astype(np.int64)),
        }
        sample["mel_spec"] = torch.from_numpy(
            np.asarray(h5["mel_spec"][local_index], dtype=np.float32)
        )
        if self.read_audio:
            sample["audio"] = torch.from_numpy(
                np.asarray(h5["audio"][local_index], dtype=np.float32)
            )
        return sample

    def close(self) -> None:
        for h5 in self._files.values():
            h5.close()
        self._files.clear()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_files"] = {}
        return state

    def __del__(self) -> None:
        self.close()


def helm_collate_batch(
    samples: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    if not samples:
        raise ValueError("Cannot collate an empty Helm batch")
    lengths = torch.tensor([sample["token_value_ids"].numel() for sample in samples])
    kind_ids = pad_sequence(
        [sample["token_kind_ids"] for sample in samples],
        batch_first=True,
        padding_value=int(HelmTokenKind.BOS),
    )
    value_ids = pad_sequence(
        [sample["token_value_ids"] for sample in samples],
        batch_first=True,
        padding_value=0,
    )
    field_ids = pad_sequence(
        [sample["token_field_ids"] for sample in samples],
        batch_first=True,
        padding_value=HelmAutoregressiveTokenizer.NO_FIELD,
    )
    token_mask = torch.arange(value_ids.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
    loss_mask = token_mask & kind_ids.ne(int(HelmTokenKind.BOS))
    batch = {
        "full_parameters_backend": torch.stack(
            [sample["full_parameters_backend"] for sample in samples]
        ),
        "token_kind_ids": kind_ids,
        "token_value_ids": value_ids,
        "token_field_ids": field_ids,
        "token_mask": token_mask,
        "loss_mask": loss_mask,
        "token_lengths": lengths,
    }
    for key in ("mel_spec", "audio"):
        if key in samples[0]:
            batch[key] = torch.stack([sample[key] for sample in samples])
    return batch
