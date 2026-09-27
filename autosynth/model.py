from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .positions import PositionalEncoding1D, PositionEmbeddingSine
from .layers import Transformer
from .cnn import CNNBackbone
from .tokens import HelmAutoregressiveTokenizer, HelmLossKind, HelmTokenKind
from .smoothing import GaussianKernelConv


class HelmAutoregressiveTransformer(nn.Module):
    MAX_CARDINALITY = HelmAutoregressiveTokenizer.CONTINUOUS_CARDINALITY
    FIELD_EMBEDDING_COUNT = 140

    def __init__(
        self,
        *,
        tokenizer: HelmAutoregressiveTokenizer,
        input_channels: int = 2,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 12,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
        normalize_before: bool = True,
        gaussian_sigma: float = 0.02,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = tokenizer.MAX_SEQUENCE_LENGTH
        self.max_card = self.MAX_CARDINALITY
        self.backbone = CNNBackbone(in_channels=int(input_channels), d_model=d_model)
        self.enc_pos_embed = PositionEmbeddingSine(d_model // 2)
        self.query_pos_embed = PositionalEncoding1D(d_model, self.seq_len)
        self.transformer = Transformer(
            num_queries=self.seq_len,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            normalize_before=normalize_before,
        )
        self.value_embed = nn.Embedding(self.MAX_CARDINALITY, d_model)
        self.kind_embed = nn.Embedding(len(HelmTokenKind), d_model)
        self.field_embed = nn.Embedding(self.FIELD_EMBEDDING_COUNT, d_model)
        self.proj = nn.Linear(d_model, self.MAX_CARDINALITY)
        self.dropout = nn.Dropout(float(dropout))
        self.gaussian_conv = GaussianKernelConv(sigma=gaussian_sigma)

    @staticmethod
    def _field_embedding_ids(field_ids: torch.Tensor) -> torch.Tensor:
        return field_ids + 1

    def forward(
        self,
        spectrogram: torch.Tensor,
        token_values: torch.Tensor,
        token_kinds: torch.Tensor,
        token_fields: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory, src_pos = self.encode_audio(spectrogram)
        return self.decode_tokens(
            memory, src_pos, token_values, token_kinds, token_fields, token_mask
        )

    def encode_audio(
        self, spectrogram: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(spectrogram)
        pos_enc = self.enc_pos_embed(feats)
        src = feats.flatten(2).permute(2, 0, 1)
        src_pos = pos_enc.flatten(2).permute(2, 0, 1)
        return (self.transformer.encoder(src, pos=src_pos), src_pos)

    def decode_tokens(
        self,
        memory: torch.Tensor,
        src_pos: torch.Tensor,
        token_values: torch.Tensor,
        token_kinds: torch.Tensor,
        token_fields: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        if (
            not token_values.shape
            == token_kinds.shape
            == token_fields.shape
            == token_mask.shape
        ):
            raise ValueError(
                "Helm token values, kinds, fields, and mask must have equal shape"
            )
        if token_values.ndim != 2:
            raise ValueError(
                f"Expected Helm tokens with shape (B,L), got {token_values.shape}"
            )
        if token_values.shape[1] > self.seq_len:
            raise ValueError(
                f"Helm sequence length {token_values.shape[1]} exceeds maximum {self.seq_len}"
            )
        token_mask = token_mask.bool()
        batch, length = token_values.shape
        if memory.shape[1] != batch or src_pos.shape != memory.shape:
            raise ValueError(
                "Encoded Helm audio memory does not match token batch size"
            )
        bos_values = torch.zeros(
            (batch, 1), dtype=torch.long, device=token_values.device
        )
        bos_kinds = torch.full(
            (batch, 1),
            int(HelmTokenKind.BOS),
            dtype=torch.long,
            device=token_values.device,
        )
        bos_fields = torch.full(
            (batch, 1),
            HelmAutoregressiveTokenizer.NO_FIELD,
            dtype=torch.long,
            device=token_values.device,
        )
        previous_values = torch.cat((bos_values, token_values[:, :-1]), dim=1)
        previous_kinds = torch.cat((bos_kinds, token_kinds[:, :-1]), dim=1)
        previous_fields = torch.cat((bos_fields, token_fields[:, :-1]), dim=1)
        tgt = (
            self.value_embed(previous_values)
            + self.kind_embed(previous_kinds)
            + self.field_embed(self._field_embedding_ids(previous_fields))
        ).transpose(0, 1)
        causal = torch.triu(
            torch.ones((length, length), device=token_values.device, dtype=torch.bool),
            diagonal=1,
        )
        query_pos = (
            self.query_pos_embed.encoding[0, :length].unsqueeze(1).expand(-1, batch, -1)
        )
        query_pos = query_pos.to(device=tgt.device, dtype=tgt.dtype)
        query_pos = query_pos + (
            self.kind_embed(token_kinds)
            + self.field_embed(self._field_embedding_ids(token_fields))
        ).transpose(0, 1)
        decoded = self.transformer.decoder(
            tgt,
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=~token_mask,
            pos=src_pos,
            query_pos=query_pos,
        )
        if decoded.dim() == 4:
            decoded = decoded[-1]
        return self.proj(self.dropout(decoded.transpose(0, 1)).contiguous())

    def loss(
        self,
        logits,
        targets,
        token_kinds,
        token_fields,
        loss_mask,
        *,
        temperature=0.2,
        return_components=False,
    ):
        losses = []
        pairs = torch.stack((token_kinds[loss_mask], token_fields[loss_mask]), dim=1)
        for raw_kind, field in torch.unique(pairs, dim=0).tolist():
            kind = HelmTokenKind(raw_kind)
            mask = loss_mask & token_kinds.eq(raw_kind) & token_fields.eq(field)
            spec = self.tokenizer.target_spec(kind, field)
            target = F.one_hot(targets[mask], num_classes=spec.cardinality).float()
            if spec.loss_kind == HelmLossKind.GAUSSIAN:
                target = self.gaussian_conv(target)
            logp = F.log_softmax(
                logits[mask][:, : spec.cardinality] / temperature, dim=-1
            )
            losses.append((kind, -(target * logp).sum(-1).mean()))
        if not losses:
            raise ValueError("SFT batch has no valid target fields")
        total = torch.stack([value for _, value in losses]).mean()
        if not return_components:
            return total
        midi = {
            HelmTokenKind.NOTE_OR_END,
            HelmTokenKind.MIDI_PITCH,
            HelmTokenKind.MIDI_VELOCITY,
            HelmTokenKind.MIDI_ONSET,
            HelmTokenKind.MIDI_DURATION,
            HelmTokenKind.CONTROL_OR_END,
            HelmTokenKind.CONTROL_VALUE,
            HelmTokenKind.CONTROL_TIME,
        }
        param_loss = torch.stack(
            [value for kind, value in losses if kind not in midi]
        ).mean()
        midi_loss = torch.stack(
            [value for kind, value in losses if kind in midi]
        ).mean()
        return total, param_loss, midi_loss
