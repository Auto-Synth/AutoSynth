from __future__ import annotations
import atexit
import concurrent.futures as cf
import logging
from typing import Any, Sequence
import numpy as np
from .synth.helm_subprocess_renderer import SubprocessHelmRenderer


class SubprocessHelmRendererPool:
    def __init__(self, *, num_workers: int = 1, **renderer_kwargs: Any) -> None:
        self._workers = [
            SubprocessHelmRenderer(**renderer_kwargs)
            for _ in range(max(1, int(num_workers)))
        ]
        atexit.register(self.close)

    def render_single(self, *, decoded: Any, duration_seconds: float) -> np.ndarray:
        return np.asarray(
            self._workers[0].render(
                base_state_payload=None,
                parameters=decoded.parameters,
                routes=decoded.routes,
                performance=decoded.performance,
                duration_seconds=float(duration_seconds),
            ),
            dtype=np.float32,
        )

    def render_jobs_resilient(
        self, jobs: Sequence[dict[str, Any]]
    ) -> list[np.ndarray | None]:
        if not jobs:
            return []
        partitions: list[list[tuple[int, dict[str, Any]]]] = [[] for _ in self._workers]
        for index, job in enumerate(jobs):
            partitions[index % len(self._workers)].append((index, job))
        results: list[np.ndarray | None] = [None] * len(jobs)

        def _run_partition(
            worker: SubprocessHelmRenderer, items: Sequence[tuple[int, dict[str, Any]]]
        ) -> list[tuple[int, np.ndarray | None]]:
            rendered: list[tuple[int, np.ndarray | None]] = []
            for index, job in items:
                decoded = job["decoded"]
                try:
                    audio = worker.render(
                        base_state_payload=None,
                        parameters=decoded.parameters,
                        routes=decoded.routes,
                        performance=decoded.performance,
                        duration_seconds=float(job["duration_seconds"]),
                    )
                    rendered.append((index, np.asarray(audio, dtype=np.float32)))
                except Exception as exc:
                    logging.getLogger(__name__).warning(
                        "Helm render job %d failed: %s", index, exc
                    )
                    rendered.append((index, None))
            return rendered

        if len(self._workers) == 1:
            partitions_out = [_run_partition(self._workers[0], partitions[0])]
        else:
            partitions_out = []
            with cf.ThreadPoolExecutor(max_workers=len(self._workers)) as executor:
                futures = [
                    executor.submit(_run_partition, worker, items)
                    for worker, items in zip(self._workers, partitions, strict=True)
                    if items
                ]
                for future in cf.as_completed(futures):
                    partitions_out.append(future.result())
        for partition in partitions_out:
            for index, audio in partition:
                results[index] = audio
        return results

    def close(self) -> None:
        for worker in self._workers:
            try:
                worker.close()
            except Exception:
                pass
