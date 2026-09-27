from __future__ import annotations
from pathlib import Path
from typing import List, Mapping, Sequence
import numpy as np
import torch
from scipy.spatial.distance import cdist

_MSS_MEL_PARAMS = ((10, 5, 32), (25, 10, 64), (100, 50, 128))


def trim_leading_silence_and_left_align(
    audio: torch.Tensor | np.ndarray,
    *,
    sample_rate: int,
    threshold_dbfs: float = -60.0,
    frame_seconds: float = 0.05,
    hop_seconds: float = 0.01,
) -> torch.Tensor:
    tensor = (
        audio
        if torch.is_tensor(audio)
        else torch.from_numpy(np.asarray(audio, dtype=np.float32))
    )
    tensor = tensor.detach().cpu().float()
    squeeze = False
    has_channel = False
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
        squeeze = True
    elif tensor.ndim == 3 and tensor.shape[1] == 1:
        tensor = tensor[:, 0, :]
        has_channel = True
    elif tensor.ndim != 2:
        raise ValueError(
            f"trim_leading_silence_and_left_align expects shape (T,), (B,T), or (B,1,T), got {tuple(tensor.shape)}"
        )
    batch, total_samples = tensor.shape
    if total_samples == 0:
        return audio if torch.is_tensor(audio) else tensor
    frame_length = max(int(round(frame_seconds * float(sample_rate))), 1)
    hop_length = max(int(round(hop_seconds * float(sample_rate))), 1)
    if total_samples < frame_length:
        pad = frame_length - total_samples
        work = torch.nn.functional.pad(tensor, (0, pad))
    else:
        work = tensor
    frames = work.unfold(dimension=-1, size=frame_length, step=hop_length)
    rms = torch.sqrt(torch.clamp(frames.pow(2).mean(dim=-1), min=1e-12))
    rms_db = 20.0 * torch.log10(rms)
    active = rms_db > float(threshold_dbfs)
    shifted = torch.zeros_like(tensor)
    for idx in range(batch):
        active_idx = torch.nonzero(active[idx], as_tuple=False)
        if active_idx.numel() == 0:
            shifted[idx] = tensor[idx]
            continue
        start = int(active_idx[0, 0].item()) * hop_length
        if start <= 0:
            shifted[idx] = tensor[idx]
            continue
        if start >= total_samples:
            continue
        shifted[idx, : total_samples - start] = tensor[idx, start:]
    if has_channel:
        shifted = shifted.unsqueeze(1)
    if squeeze:
        shifted = shifted[0]
    return shifted


class BasicPitchEmbeddingDistance:
    HEAD_OUTPUTS = {
        "note": "StatefulPartitionedCall:1",
        "onset": "StatefulPartitionedCall:2",
        "contour": "StatefulPartitionedCall:0",
    }

    def __init__(
        self,
        *,
        note_weight: float = 1 / 3,
        onset_weight: float = 1 / 3,
        contour_weight: float = 1 / 3,
        inference_batch_size: int = 16,
        device: torch.device | str = "cpu",
    ) -> None:
        try:
            import onnx
            import onnxruntime as ort
            from basic_pitch import FilenameSuffix, build_icassp_2022_model_path
            from basic_pitch.constants import (
                AUDIO_N_SAMPLES,
                AUDIO_SAMPLE_RATE,
                FFT_HOP,
            )
            from basic_pitch.inference import unwrap_output, window_audio_file
        except Exception as exc:
            raise ImportError(
                "Basic Pitch embedding reward requires basic-pitch, onnx and onnxruntime."
            ) from exc
        weights = np.array(
            [note_weight, onset_weight, contour_weight], dtype=np.float64
        )
        if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError(
                "Basic Pitch head weights must be finite, non-negative and sum to > 0"
            )
        if int(inference_batch_size) < 1:
            raise ValueError("inference_batch_size must be positive")
        self.head_weights = dict(
            zip(self.HEAD_OUTPUTS, weights / weights.sum(), strict=True)
        )
        self.inference_batch_size = int(inference_batch_size)
        self.sample_rate = int(AUDIO_SAMPLE_RATE)
        self.audio_n_samples = int(AUDIO_N_SAMPLES)
        self.fft_hop = int(FFT_HOP)
        self.overlapping_frames = 30
        graph = onnx.shape_inference.infer_shapes(
            onnx.load(build_icassp_2022_model_path(FilenameSuffix.onnx))
        )
        producers = {
            output: node for node in graph.graph.node for output in node.output
        }
        value_info = {value.name: value for value in graph.graph.value_info}
        self.hidden_names = []
        for output in self.HEAD_OUTPUTS.values():
            reshape = producers[output]
            sigmoid = producers[reshape.input[0]]
            conv = producers[sigmoid.input[0]]
            if (reshape.op_type, sigmoid.op_type, conv.op_type) != (
                "Reshape",
                "Sigmoid",
                "Conv",
            ):
                raise ValueError(f"Unexpected Basic Pitch prediction head: {output}")
            self.hidden_names.append(conv.input[0])
        del graph.graph.output[:]
        graph.graph.output.extend((value_info[name] for name in self.hidden_names))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        device = torch.device(device)
        if device.type == "cuda":
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError(
                    "Basic Pitch CUDA inference requires onnxruntime-gpu"
                )
            providers = [
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": device.index or 0,
                        "use_tf32": 0,
                        "cudnn_conv_algo_search": "HEURISTIC",
                    },
                )
            ]
        else:
            providers = ["CPUExecutionProvider"]
        self.model = ort.InferenceSession(
            graph.SerializeToString(), options, providers=providers
        )
        if (
            device.type == "cuda"
            and self.model.get_providers()[0] != "CUDAExecutionProvider"
        ):
            raise RuntimeError("Basic Pitch CUDA provider failed to initialize")
        self.input_name = self.model.get_inputs()[0].name
        self._unwrap_output = unwrap_output
        self._window_audio_file = window_audio_file

    @staticmethod
    def _mono(audio: np.ndarray | torch.Tensor) -> np.ndarray:
        array = (
            audio.detach().cpu().numpy()
            if torch.is_tensor(audio)
            else np.asarray(audio)
        )
        array = np.asarray(array, dtype=np.float32)
        if array.ndim == 1:
            return array
        if array.ndim == 2:
            channel_axis = 0 if array.shape[0] <= 8 else 1
            return array.mean(axis=channel_axis, dtype=np.float32)
        return array.reshape(-1).astype(np.float32, copy=False)

    def _prepare_waveform(
        self, audio: np.ndarray | torch.Tensor, *, sample_rate: int
    ) -> np.ndarray:
        import librosa

        waveform = self._mono(audio)
        waveform = np.asarray(
            trim_leading_silence_and_left_align(waveform, sample_rate=int(sample_rate)),
            dtype=np.float32,
        )
        if int(sample_rate) != self.sample_rate:
            waveform = librosa.resample(
                waveform, orig_sr=int(sample_rate), target_sr=self.sample_rate
            ).astype(np.float32, copy=False)
        return np.clip(waveform, -1.0, 1.0)

    def extract_batch(
        self, audio: Sequence[np.ndarray | torch.Tensor], *, sample_rate: int
    ) -> list[dict[str, np.ndarray]]:
        if not audio:
            return []
        overlap_samples = self.overlapping_frames * self.fft_hop
        hop_size = self.audio_n_samples - overlap_samples
        all_windows: list[np.ndarray] = []
        bounds: list[tuple[int, int, int]] = []
        for item in audio:
            waveform = self._prepare_waveform(item, sample_rate=sample_rate)
            original_length = int(waveform.shape[0])
            padded = np.concatenate(
                [np.zeros(overlap_samples // 2, dtype=np.float32), waveform]
            )
            start = len(all_windows)
            all_windows.extend(
                (window for window, _ in self._window_audio_file(padded, hop_size))
            )
            bounds.append((start, len(all_windows), original_length))
        predicted_batches: dict[str, list[np.ndarray]] = {
            name: [] for name in self.HEAD_OUTPUTS
        }
        for start in range(0, len(all_windows), self.inference_batch_size):
            window_batch = np.stack(
                all_windows[start : start + self.inference_batch_size]
            ).astype(np.float32, copy=False)
            hidden = self.model.run(self.hidden_names, {self.input_name: window_batch})
            for name, values in zip(predicted_batches, hidden, strict=True):
                values = values.transpose(0, 2, 3, 1)
                predicted_batches[name].append(
                    values.reshape(values.shape[0], values.shape[1], -1)
                )
        predicted_all = {
            name: np.concatenate(parts, axis=0)
            for name, parts in predicted_batches.items()
        }
        features: list[dict[str, np.ndarray]] = []
        for start, end, original_length in bounds:
            features.append(
                {
                    name: np.asarray(
                        self._unwrap_output(
                            values[start:end], original_length, self.overlapping_frames
                        ),
                        dtype=np.float32,
                    )
                    for name, values in predicted_all.items()
                }
            )
        return features

    def extract(
        self, audio: np.ndarray | torch.Tensor, *, sample_rate: int
    ) -> dict[str, np.ndarray]:
        return self.extract_batch([audio], sample_rate=sample_rate)[0]

    @staticmethod
    def _cosine_distance(pred: np.ndarray, target: np.ndarray) -> float:
        frames = max(int(pred.shape[0]), int(target.shape[0]))
        if frames == 0:
            return 0.0
        pred = np.pad(pred, ((0, frames - len(pred)), (0, 0)))
        target = np.pad(target, ((0, frames - len(target)), (0, 0)))
        pred_norm = np.linalg.norm(pred, axis=1)
        target_norm = np.linalg.norm(target, axis=1)
        denominator = pred_norm * target_norm
        similarity = np.divide(
            np.einsum("tf,tf->t", pred, target),
            denominator,
            out=np.zeros(frames, dtype=np.float32),
            where=denominator > 1e-12,
        )
        similarity[(pred_norm <= 1e-06) & (target_norm <= 1e-06)] = 1.0
        return float(np.mean(1.0 - np.clip(similarity, -1.0, 1.0)))

    def distance(
        self, pred: Mapping[str, np.ndarray], target: Mapping[str, np.ndarray]
    ) -> float:
        return float(
            sum(
                (
                    weight * self._cosine_distance(pred[name], target[name])
                    for name, weight in self.head_weights.items()
                )
            )
        )

    def __call__(
        self,
        pred: np.ndarray | torch.Tensor,
        target: np.ndarray | torch.Tensor,
        *,
        sample_rate: int,
    ) -> float:
        return self.distance(
            self.extract(pred, sample_rate=sample_rate),
            self.extract(target, sample_rate=sample_rate),
        )


class AudioBoxAestheticsScorer:
    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        batch_size: int = 4,
        axes: Sequence[str] = ("CE", "CU"),
        device: torch.device | str = "cpu",
    ) -> None:
        try:
            from audiobox_aesthetics.infer import AesPredictor, make_inference_batch
        except Exception as exc:
            raise ImportError(
                "AudioBox rewards require `audiobox_aesthetics==0.0.4`."
            ) from exc
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"AudioBox checkpoint not found: {path}")
        if int(batch_size) < 1:
            raise ValueError("AudioBox batch_size must be positive")
        self.predictor = AesPredictor(checkpoint_pth=str(path))
        self.predictor.device = torch.device(device)
        self.predictor.model.to(self.predictor.device).float().eval().requires_grad_(
            False
        )
        self.batch_size = int(batch_size)
        self.axes = tuple(axes)
        if (
            not self.axes
            or len(set(self.axes)) != len(self.axes)
            or (not set(self.axes) <= {"CE", "CU", "PQ"})
        ):
            raise ValueError("AudioBox axes must be distinct CE, CU or PQ names")
        self._window_audio = make_inference_batch

    @torch.inference_mode()
    def score_batch(
        self, audio: Sequence[np.ndarray | torch.Tensor], *, sample_rate: int
    ) -> list[dict[str, float]]:
        records = []
        for waveform in audio:
            mono = BasicPitchEmbeddingDistance._mono(waveform)
            records.append(
                {
                    "path": torch.from_numpy(mono.copy()).unsqueeze(0),
                    "sample_rate": int(sample_rate),
                }
            )
        scores: list[dict[str, float]] = []
        for start in range(0, len(records), self.batch_size):
            batch = records[start : start + self.batch_size]
            wavs, masks, weights, bids = self._window_audio(
                self.predictor.audio_resample_mono(batch),
                sample_rate=self.predictor.sample_rate,
            )
            device = self.predictor.device
            output = self.predictor.model(
                {
                    "wav": torch.stack(wavs).to(device),
                    "mask": torch.stack(masks).to(device),
                }
            )
            membership = torch.arange(len(batch), device=device)[
                :, None
            ] == torch.tensor(bids, device=device)
            window_weights = membership * torch.tensor(weights, device=device)
            averages = []
            for name in self.axes:
                prediction = (
                    self.predictor.target_transform[name]
                    .inverse(output[name])
                    .reshape(-1)
                )
                averages.append(
                    (window_weights * prediction).sum(dim=1) / window_weights.sum(dim=1)
                )
            for values in torch.stack(averages, dim=1).cpu().tolist():
                scores.append(
                    {
                        name.lower(): float(value)
                        for name, value in zip(self.axes, values)
                    }
                )
        if len(scores) != len(records):
            raise RuntimeError(
                f"AudioBox returned {len(scores)} scores for {len(records)} clips"
            )
        return scores


def _to_mono_1d(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
            return audio.mean(axis=0)
        return audio.mean(axis=1)
    return audio.reshape(-1).astype(np.float32, copy=False)


def _compute_mel_specs(y: np.ndarray, sample_rate: float) -> List[np.ndarray]:
    import librosa

    y_mono = _to_mono_1d(y)
    mel_specs: List[np.ndarray] = []
    for window_ms, hop_ms, n_mels in _MSS_MEL_PARAMS:
        win_length = int(window_ms * sample_rate / 1000.0)
        hop_length = int(hop_ms * sample_rate / 1000.0)
        spec = librosa.feature.melspectrogram(
            y=y_mono,
            sr=int(sample_rate),
            n_mels=int(n_mels),
            n_fft=int(win_length),
            hop_length=int(hop_length),
            window="hann",
        )
        mel_specs.append(librosa.power_to_db(spec, ref=np.max))
    return mel_specs


def compute_mss(
    target: np.ndarray, pred: np.ndarray, sample_rate: float = 44100.0
) -> float:
    target_specs = _compute_mel_specs(target, sample_rate)
    pred_specs = _compute_mel_specs(pred, sample_rate)
    dist = 0.0
    for target_spec, pred_spec in zip(target_specs, pred_specs):
        dist += float(np.mean(np.abs(target_spec - pred_spec)))
    return float(dist / max(len(target_specs), 1))


def _get_stft_mag(y: np.ndarray, sample_rate: float) -> np.ndarray:
    import librosa

    y_mono = _to_mono_1d(y)
    win_length = int(0.05 * sample_rate)
    hop_length = int(0.02 * sample_rate)
    stft = librosa.stft(
        y_mono,
        n_fft=win_length,
        hop_length=hop_length,
        win_length=win_length,
        window="hann",
    ).T
    return np.abs(stft)


def _batched_wasserstein_distance_np(
    hist1: np.ndarray, hist2: np.ndarray
) -> np.ndarray:
    bin_width = 1.0 / float(hist1.shape[-1])
    cdf1 = np.cumsum(hist1, axis=-1)
    cdf2 = np.cumsum(hist2, axis=-1)
    return np.sum(np.abs(cdf1 - cdf2), axis=-1) * bin_width


def compute_sot(
    target: np.ndarray, pred: np.ndarray, sample_rate: float = 44100.0
) -> float:
    target_stft = _get_stft_mag(target, sample_rate)
    pred_stft = _get_stft_mag(pred, sample_rate)
    target_stft = target_stft / np.clip(
        target_stft.sum(axis=-1, keepdims=True), 1e-06, None
    )
    pred_stft = pred_stft / np.clip(pred_stft.sum(axis=-1, keepdims=True), 1e-06, None)
    dists = _batched_wasserstein_distance_np(target_stft, pred_stft)
    return float(dists.mean())


def compute_rms_env_cosine(
    target: np.ndarray, pred: np.ndarray, sample_rate: float = 44100.0
) -> float:
    import librosa

    target_mono = _to_mono_1d(target)
    pred_mono = _to_mono_1d(pred)
    win_length = int(0.05 * sample_rate)
    hop_length = int(0.025 * sample_rate)
    target_rms = librosa.feature.rms(
        y=target_mono, frame_length=win_length, hop_length=hop_length
    )
    pred_rms = librosa.feature.rms(
        y=pred_mono, frame_length=win_length, hop_length=hop_length
    )
    target_norm = float(np.linalg.norm(target_rms, ord=2, axis=-1).reshape(-1)[0])
    pred_norm = float(np.linalg.norm(pred_rms, ord=2, axis=-1).reshape(-1)[0])
    denom = target_norm * pred_norm
    if not np.isfinite(denom) or denom <= 0.0:
        denom = 1e-12
    cosine_sim = float(np.dot(target_rms[0], pred_rms[0]) / denom)
    return float(cosine_sim)


class WmfccMetric:
    def __init__(
        self,
        sample_rate: int = 44100,
        n_mfcc_wmfcc: int = 20,
        n_mels_wmfcc: int = 128,
        win_sec: float = 0.05,
        hop_sec: float = 0.01,
    ) -> None:
        import librosa

        self.librosa = librosa
        self.sample_rate = int(sample_rate)
        self.n_mfcc_wmfcc = int(n_mfcc_wmfcc)
        self.n_mels_wmfcc = int(n_mels_wmfcc)
        self.win_length = int(win_sec * sample_rate)
        self.hop_length = int(hop_sec * sample_rate)

    def _mfcc_wmfcc(self, waveform: np.ndarray) -> np.ndarray:
        return self.librosa.feature.mfcc(
            y=waveform,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc_wmfcc,
            n_fft=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels_wmfcc,
        )

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        from dtw import dtw

        scores = []
        for pre, tgt in zip(pred.detach().cpu().numpy(), target.detach().cpu().numpy()):
            x, y = (self._mfcc_wmfcc(tgt), self._mfcc_wmfcc(pre))
            x, y = (x.reshape(-1, x.shape[-1]).T, y.reshape(-1, y.shape[-1]).T)
            costs = cdist(x, y, metric="cityblock") / x.shape[1]
            scores.append(
                dtw(
                    costs, step_pattern="symmetric2", distance_only=True
                ).normalizedDistance
            )
        return torch.tensor(float(np.mean(scores)), dtype=torch.float32)
