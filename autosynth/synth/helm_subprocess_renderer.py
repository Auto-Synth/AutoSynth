from __future__ import annotations
import multiprocessing as mp
import os
import sys
from multiprocessing.connection import Connection
from typing import Any, Sequence
import numpy as np
from .helm_midi import MidiPerformance
from .helm_renderer import HelmDawRenderer
from .helm_state import HelmRoute


class HelmNativeRenderError(RuntimeError):
    pass


def _suppress_process_output() -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    descriptor = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(descriptor, sys.stdout.fileno())
        os.dup2(descriptor, sys.stderr.fileno())
    finally:
        if descriptor > sys.stderr.fileno():
            os.close(descriptor)


def _worker_main(connection: Connection, renderer_kwargs: dict[str, Any]) -> None:
    renderer_kwargs = dict(renderer_kwargs)
    private_xvfb = bool(renderer_kwargs.pop("_private_xvfb", True))
    fresh_plugin_per_render = bool(
        renderer_kwargs.pop("_fresh_plugin_per_render", True)
    )
    suppress_native_output = bool(renderer_kwargs.pop("_suppress_native_output", False))
    if suppress_native_output:
        _suppress_process_output()
    if private_xvfb:
        os.environ.pop("DISPLAY", None)
    renderer = HelmDawRenderer(**renderer_kwargs)
    has_rendered = False
    try:
        while True:
            try:
                message = connection.recv()
            except EOFError:
                break
            command = message[0]
            if command == "close":
                break
            if command != "render":
                connection.send(("error", f"Unknown Helm worker command: {command}"))
                continue
            _, payload, parameters, routes, performance, duration = message
            if fresh_plugin_per_render and has_rendered:
                renderer.recreate_plugin_instance()
            try:
                audio = renderer.render(
                    base_state_payload=payload,
                    parameters=np.asarray(parameters, dtype=np.float32),
                    routes=routes,
                    performance=performance,
                    duration_seconds=float(duration),
                )
                connection.send(("ok", np.asarray(audio, dtype=np.float32)))
                has_rendered = True
            except Exception as exc:
                connection.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        renderer.close()
        connection.close()


class SubprocessHelmRenderer:
    def __init__(
        self,
        *,
        timeout_seconds: float = 120.0,
        max_renders_per_worker: int = 64,
        private_xvfb: bool = True,
        fresh_plugin_per_render: bool = True,
        suppress_native_output: bool = False,
        **renderer_kwargs: Any,
    ) -> None:
        self.renderer_kwargs = dict(renderer_kwargs)
        self.renderer_kwargs["_private_xvfb"] = bool(private_xvfb)
        self.renderer_kwargs["_fresh_plugin_per_render"] = bool(fresh_plugin_per_render)
        self.renderer_kwargs["_suppress_native_output"] = bool(suppress_native_output)
        self.timeout_seconds = float(timeout_seconds)
        self.max_renders_per_worker = int(max_renders_per_worker)
        self._context = mp.get_context("spawn")
        self._process: mp.Process | None = None
        self._connection: Connection | None = None
        self._render_count = 0

    def render(
        self,
        *,
        base_state_payload: dict | None,
        parameters: np.ndarray,
        routes: Sequence[HelmRoute],
        performance: MidiPerformance,
        duration_seconds: float,
    ) -> np.ndarray:
        self._ensure_worker()
        assert self._connection is not None
        try:
            self._connection.send(
                (
                    "render",
                    base_state_payload,
                    np.asarray(parameters, dtype=np.float32),
                    tuple(routes),
                    performance,
                    float(duration_seconds),
                )
            )
            if not self._connection.poll(self.timeout_seconds):
                raise TimeoutError(f"Helm render exceeded {self.timeout_seconds:.1f}s")
            status, payload = self._connection.recv()
        except (
            BrokenPipeError,
            ConnectionResetError,
            EOFError,
            OSError,
            TimeoutError,
        ) as exc:
            exit_code = self._process.exitcode if self._process is not None else None
            self._stop_worker(force=True)
            raise HelmNativeRenderError(
                f"Helm worker crashed/timed out (exit_code={exit_code})"
            ) from exc
        if status != "ok":
            self._stop_worker(force=True)
            raise HelmNativeRenderError(str(payload))
        self._render_count += 1
        audio = np.asarray(payload, dtype=np.float32)
        if self._render_count >= self.max_renders_per_worker:
            self._stop_worker(force=False)
        return audio

    def _ensure_worker(self) -> None:
        if (
            self._process is not None
            and self._process.is_alive()
            and (self._connection is not None)
        ):
            return
        self._stop_worker(force=True)
        parent, child = self._context.Pipe(duplex=True)
        process = self._context.Process(
            target=_worker_main, args=(child, self.renderer_kwargs), daemon=True
        )
        process.start()
        child.close()
        self._connection = parent
        self._process = process
        self._render_count = 0

    def _stop_worker(self, *, force: bool) -> None:
        connection, process = (self._connection, self._process)
        self._connection = None
        self._process = None
        self._render_count = 0
        if connection is not None:
            if not force and process is not None and process.is_alive():
                try:
                    connection.send(("close",))
                except (BrokenPipeError, OSError):
                    pass
            connection.close()
        if process is not None:
            process.join(timeout=3.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=2.0)

    def close(self) -> None:
        self._stop_worker(force=False)

    def __enter__(self) -> "SubprocessHelmRenderer":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
