from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping
import librosa
import numpy as np
import soundfile as sf
import torch


def _load_stereo_audio(
    path: Path, *, sample_rate: int, target_samples: int
) -> tuple[np.ndarray, np.ndarray]:
    audio, source_rate = sf.read(path, dtype="float32", always_2d=True)
    stereo = np.asarray(audio.T, dtype=np.float32)
    if stereo.shape[0] == 1:
        stereo = np.repeat(stereo, 2, axis=0)
    elif stereo.shape[0] != 2:
        raise ValueError(
            f"Expected mono or stereo audio at {path}, got {stereo.shape[0]} channels"
        )
    if int(source_rate) != int(sample_rate):
        stereo = librosa.resample(
            stereo, orig_sr=int(source_rate), target_sr=int(sample_rate), axis=-1
        ).astype(np.float32, copy=False)
    if stereo.shape[-1] < target_samples:
        stereo = np.pad(stereo, ((0, 0), (0, target_samples - stereo.shape[-1])))
    else:
        stereo = stereo[:, :target_samples]
    mono = stereo.mean(axis=0, dtype=np.float32)
    return (stereo.astype(np.float32, copy=False), mono.astype(np.float32, copy=False))


def _mel_from_stereo(
    stereo: np.ndarray, *, sample_rate: int, metadata: Mapping[str, Any]
) -> torch.Tensor:
    if bool(metadata.get("use_saved_mean_and_variance", False)):
        raise ValueError(
            "Helm GRPO currently expects the unnormalized stage-1 mel representation"
        )
    mel_cfg = metadata["mel"]
    n_fft = int(float(mel_cfg["window_seconds"]) * sample_rate)
    hop_length = int(sample_rate / float(mel_cfg["frames_per_second"]))
    power = np.stack(
        [
            librosa.feature.melspectrogram(
                y=channel,
                sr=sample_rate,
                n_mels=int(mel_cfg["n_mels"]),
                n_fft=n_fft,
                hop_length=hop_length,
                window=str(mel_cfg["window"]),
            )
            for channel in stereo
        ],
        axis=0,
    )
    mel = librosa.power_to_db(power, ref=np.max).astype(np.float32, copy=False)
    target_frames = int(metadata["mel_frames"])
    if mel.shape[-1] < target_frames:
        mel = np.pad(mel, ((0, 0), (0, 0), (0, target_frames - mel.shape[-1])))
    else:
        mel = mel[..., :target_frames]
    return torch.from_numpy(mel)
