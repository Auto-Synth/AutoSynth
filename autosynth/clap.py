import numpy as np
import torch
import torch.nn.functional as F

from .paths import resolve_project_path


class ClapMetric:
    def __init__(self, checkpoint, device, batch_size=8):
        from msclap import CLAP

        self.device = torch.device(device)
        self.batch_size = batch_size
        self.wrapper = CLAP(
            model_fp=str(resolve_project_path(checkpoint)),
            version="2023",
            use_cuda=False,
        )
        self.model = (
            self.wrapper.clap.to(self.device).float().eval().requires_grad_(False)
        )
        self.sample_rate = int(self.wrapper.args.sampling_rate)

    @torch.inference_mode()
    def audio(self, waveforms, sample_rate=44100):
        import librosa

        embeddings = []
        size = 7 * self.sample_rate
        for start in range(0, len(waveforms), self.batch_size):
            items = waveforms[start : start + self.batch_size]
            windows = np.zeros((len(items), size), dtype=np.float32)
            for i, item in enumerate(items):
                item = np.asarray(item, dtype=np.float32)
                if item.ndim != 1:
                    raise ValueError("CLAP input must be mono")
                if sample_rate != self.sample_rate:
                    item = librosa.resample(
                        item, orig_sr=sample_rate, target_sr=self.sample_rate
                    )
                item = np.clip(item[:size], -1, 1)
                windows[i, : len(item)] = item
            with torch.autocast(device_type=self.device.type, enabled=False):
                value = self.model.audio_encoder(
                    torch.from_numpy(windows).to(self.device)
                )[0]
            embeddings.append(value.float().cpu())
        return torch.cat(embeddings)

    @torch.inference_mode()
    def text(self, prompts):
        features = []
        for start in range(0, len(prompts), self.batch_size):
            texts = prompts[start : start + self.batch_size]
            if any(not text.strip() for text in texts):
                raise ValueError("Empty text prompt")
            tokens = self.wrapper.tokenizer(
                [text + " <|endoftext|>" for text in texts],
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            if (
                tokens["input_ids"].shape[1]
                > self.model.caption_encoder.base.config.max_position_embeddings
            ):
                raise ValueError("Text exceeds MS-CLAP context length")
            tokens = {
                k: v.to(self.device)
                for k, v in tokens.items()
                if k in self.wrapper.token_keys
            }
            with torch.autocast(device_type=self.device.type, enabled=False):
                features.append(self.model.caption_encoder(tokens).float().cpu())
        return torch.cat(features)

    @staticmethod
    def distance(a, b):
        return 1 - F.cosine_similarity(a, b, dim=-1)
