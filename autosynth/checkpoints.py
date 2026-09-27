from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Callable, Mapping


class MetricCheckpointStore:
    def __init__(self, root: Path, *, modes: Mapping[str, str], top_k: int = 3):
        self.root = Path(root)
        self.modes = dict(modes)
        self.top_k = int(top_k)
        if (
            self.top_k < 1
            or not self.modes
            or any((mode not in {"min", "max"} for mode in self.modes.values()))
        ):
            raise ValueError(
                "Checkpoint rankings need positive top_k and min/max modes"
            )
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "checkpoint_rankings.json"

    @staticmethod
    def filename(step: int) -> str:
        return f"checkpoint_step_{step:08d}.pt"

    @staticmethod
    def _steps(index: dict) -> set[int]:
        steps = {row["step"] for rows in index["rankings"].values() for row in rows}
        if index["last_step"] is not None:
            steps.add(index["last_step"])
        return steps

    def save(
        self,
        *,
        step: int,
        metrics: Mapping[str, float],
        save_checkpoint: Callable[[Path], None],
        save_last: bool,
    ) -> None:
        if step < 0:
            raise ValueError("Checkpoint step must be nonnegative")
        old = (
            json.loads(self.index_path.read_text())
            if self.index_path.exists()
            else {
                "format_version": 1,
                "top_k": self.top_k,
                "modes": self.modes,
                "rankings": {name: [] for name in self.modes},
                "last_step": None,
            }
        )
        if (
            old["format_version"] != 1
            or old["modes"] != self.modes
            or old["top_k"] != self.top_k
        ):
            raise ValueError(
                "Checkpoint ranking definition changed; use a new experiment directory"
            )
        rankings = {}
        for name, mode in self.modes.items():
            rows = list(old["rankings"][name])
            score = float(metrics.get(name, float("nan")))
            if math.isfinite(score):
                rows = [row for row in rows if row["step"] != step]
                rows.append({"step": step, "score": score})
            rows.sort(
                key=lambda row: (
                    row["score"] if mode == "min" else -row["score"],
                    row["step"],
                )
            )
            rankings[name] = rows[: self.top_k]
        new = {
            **old,
            "rankings": rankings,
            "last_step": step if save_last else old["last_step"],
        }
        retained = self._steps(new)
        path = self.root / self.filename(step)
        if step in retained and (not path.exists()):
            save_checkpoint(path)
        if save_last:
            temporary_link = self.root / "checkpoint_last.pt.tmp"
            temporary_link.unlink(missing_ok=True)
            temporary_link.symlink_to(path.name)
            temporary_link.replace(self.root / "checkpoint_last.pt")
        temporary_index = self.index_path.with_suffix(".json.tmp")
        temporary_index.write_text(json.dumps(new, indent=2, allow_nan=False) + "\n")
        temporary_index.replace(self.index_path)
        for dropped in self._steps(old) - retained:
            (self.root / self.filename(dropped)).unlink(missing_ok=True)
