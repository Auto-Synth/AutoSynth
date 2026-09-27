from __future__ import annotations
from collections.abc import Sequence
import torch
import torch.nn.functional as F
from .tokens import HelmAutoregressiveTokenizer, HelmTokenSequence


def group_advantages(rewards):
    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError("Expected at least two candidates per target")
    return (
        (rewards - rewards.mean(dim=1, keepdim=True))
        / (rewards.std(dim=1, keepdim=True, unbiased=False) + 1e-6)
    ).detach()


def grpo_loss(log_probs, decision_mask, advantages):
    if (
        log_probs.shape != decision_mask.shape
        or advantages.numel() != log_probs.shape[0]
    ):
        raise ValueError("Policy scores and group advantages do not align")
    sequence_log_probs = (log_probs.float() * decision_mask).sum(dim=1)
    return -(advantages.detach().reshape(-1) * sequence_log_probs).mean()


def _validate_temperature(temperature: float) -> float:
    value = float(temperature)
    if not value > 0.0:
        raise ValueError(f"sampling temperature must be positive, got {temperature}")
    return value


def epsilon_greedy_probs(
    logits: torch.Tensor, *, epsilon: float, top_k: int, min_prob: float
) -> torch.Tensor:
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("sampling.epsilon must be in [0, 1]")
    if top_k < 0 or not 0.0 <= min_prob <= 1.0:
        raise ValueError("sampling.top_k must be >= 0 and min_prob in [0, 1]")
    probs_full = F.softmax(logits.float(), dim=-1)
    greedy = torch.argmax(logits, dim=-1, keepdim=True)
    greedy_probs = torch.zeros_like(probs_full).scatter(-1, greedy, 1.0)
    if epsilon == 0.0:
        return probs_full * 0.0 + greedy_probs
    if 0 < top_k < logits.shape[-1]:
        probabilities, indices = torch.topk(probs_full, k=top_k, dim=-1)
    else:
        probabilities = probs_full
        indices = torch.arange(logits.shape[-1], device=logits.device).expand_as(logits)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    keep = probabilities >= min_prob
    retained = probabilities * keep
    mass = retained.sum(dim=-1, keepdim=True)
    normalized = retained / mass.clamp_min(torch.finfo(retained.dtype).tiny)
    filtered = torch.zeros_like(probs_full).scatter(-1, indices, normalized)
    return torch.where(
        mass > 0, epsilon * filtered + (1.0 - epsilon) * greedy_probs, greedy_probs
    )


@torch.no_grad()
def sample_sequences(
    network: torch.nn.Module,
    tokenizer: HelmAutoregressiveTokenizer,
    spectrogram: torch.Tensor,
    *,
    temperature: float,
    epsilon: float,
    top_k: int,
    min_prob: float,
    generator: torch.Generator,
    greedy: bool = False,
) -> list[HelmTokenSequence]:
    temperature = _validate_temperature(temperature)
    batch_size = int(spectrogram.shape[0])
    states = [tokenizer.initial_generation_state() for _ in range(batch_size)]
    kind_ids: list[list[int]] = [[] for _ in range(batch_size)]
    value_ids: list[list[int]] = [[] for _ in range(batch_size)]
    field_ids: list[list[int]] = [[] for _ in range(batch_size)]
    memory, source_positions = network.encode_audio(spectrogram)
    shape = (batch_size, tokenizer.MAX_SEQUENCE_LENGTH)
    values_buffer = torch.zeros(shape, dtype=torch.long, device=spectrogram.device)
    kinds_buffer = torch.zeros_like(values_buffer)
    fields_buffer = torch.zeros_like(values_buffer)
    previous_active = list(range(batch_size))
    active_ids = torch.arange(batch_size, device=spectrogram.device)
    active_memory, active_positions = (memory, source_positions)
    for column in range(tokenizer.MAX_SEQUENCE_LENGTH):
        active = [
            index
            for index, state in enumerate(states)
            if tokenizer.next_token_spec(state) is not None
        ]
        if not active:
            break
        next_specs = [tokenizer.next_token_spec(states[index]) for index in active]
        for index, next_spec in zip(active, next_specs, strict=True):
            assert next_spec is not None
            kind_ids[index].append(int(next_spec.kind))
            value_ids[index].append(0)
            field_ids[index].append(int(next_spec.field_id))
        if active != previous_active:
            active_ids = torch.tensor(active, device=spectrogram.device)
            active_memory = memory[:, active_ids]
            active_positions = source_positions[:, active_ids]
            previous_active = active
        metadata = torch.tensor(
            [(int(spec.kind), int(spec.field_id)) for spec in next_specs],
            dtype=torch.long,
            device=spectrogram.device,
        )
        kinds_buffer[active_ids, column] = metadata[:, 0]
        fields_buffer[active_ids, column] = metadata[:, 1]
        if len(active) == batch_size:
            values = values_buffer[:, : column + 1]
            kinds = kinds_buffer[:, : column + 1]
            fields = fields_buffer[:, : column + 1]
        else:
            values = values_buffer[active_ids, : column + 1]
            kinds = kinds_buffer[active_ids, : column + 1]
            fields = fields_buffer[active_ids, : column + 1]
        token_mask = torch.ones_like(values, dtype=torch.bool)
        logits = network.decode_tokens(
            active_memory, active_positions, values, kinds, fields, token_mask
        )
        legal_mask = torch.zeros((len(active), logits.shape[-1]), dtype=torch.bool)
        for row, next_spec in enumerate(next_specs):
            assert next_spec is not None
            legal_mask[row, list(next_spec.allowed_value_ids)] = True
        allowed_logits = (logits[:, -1].float() / temperature).masked_fill(
            ~legal_mask.to(spectrogram.device), float("-inf")
        )
        if greedy or epsilon == 0.0:
            selected_values = allowed_logits.argmax(dim=-1)
        else:
            probabilities = epsilon_greedy_probs(
                allowed_logits, epsilon=epsilon, top_k=top_k, min_prob=min_prob
            )
            selected_values = torch.multinomial(
                probabilities, 1, generator=generator
            ).squeeze(-1)
        values_buffer[active_ids, column] = selected_values
        for index, selected in zip(active, selected_values.cpu().tolist(), strict=True):
            value_ids[index][-1] = selected
            tokenizer.advance_generation(states[index], selected)
    if any((tokenizer.next_token_spec(state) is not None for state in states)):
        raise RuntimeError(
            f"Helm generation exceeded {tokenizer.MAX_SEQUENCE_LENGTH} tokens"
        )
    return [
        HelmTokenSequence(tuple(kinds), tuple(values), tuple(fields))
        for kinds, values, fields in zip(kind_ids, value_ids, field_ids, strict=True)
    ]


def pad_sequences(
    sequences: Sequence[HelmTokenSequence], *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not sequences:
        raise ValueError("Cannot pad an empty Helm sequence batch")
    max_length = max((len(sequence) for sequence in sequences))
    batch_size = len(sequences)
    values = torch.zeros((batch_size, max_length), dtype=torch.long, device=device)
    kinds = torch.zeros((batch_size, max_length), dtype=torch.long, device=device)
    fields = torch.full(
        (batch_size, max_length),
        int(HelmAutoregressiveTokenizer.NO_FIELD),
        dtype=torch.long,
        device=device,
    )
    mask = torch.zeros((batch_size, max_length), dtype=torch.bool, device=device)
    for row, sequence in enumerate(sequences):
        length = len(sequence)
        values[row, :length] = torch.tensor(
            sequence.value_ids, dtype=torch.long, device=device
        )
        kinds[row, :length] = torch.tensor(
            sequence.kind_ids, dtype=torch.long, device=device
        )
        fields[row, :length] = torch.tensor(
            sequence.field_ids, dtype=torch.long, device=device
        )
        mask[row, :length] = True
    return (values, kinds, fields, mask)


def token_log_probs(
    network: torch.nn.Module,
    tokenizer: HelmAutoregressiveTokenizer,
    spectrogram: torch.Tensor,
    sequences: Sequence[HelmTokenSequence],
    *,
    temperature: float,
    epsilon: float,
    top_k: int,
    min_prob: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    temperature = _validate_temperature(temperature)
    if int(spectrogram.shape[0]) != len(sequences):
        raise ValueError(
            f"Spectrogram and Helm sequence batch sizes differ: {spectrogram.shape[0]} vs {len(sequences)}"
        )
    values, kinds, fields, token_mask = pad_sequences(
        sequences, device=spectrogram.device
    )
    logits = network(spectrogram, values, kinds, fields, token_mask)
    legal_mask = torch.zeros((*values.shape, logits.shape[-1]), dtype=torch.bool)
    decision_mask = torch.zeros(values.shape, dtype=torch.bool)
    for row, sequence in enumerate(sequences):
        state = tokenizer.initial_generation_state()
        for column, selected in enumerate(sequence.value_ids):
            spec = tokenizer.next_token_spec(state)
            if spec is None:
                raise ValueError("Helm sequence contains tokens after ROUTE_EOS")
            if int(sequence.kind_ids[column]) != int(spec.kind):
                raise ValueError(f"Illegal token kind at Helm position {column}")
            if int(sequence.field_ids[column]) != int(spec.field_id):
                raise ValueError(f"Illegal field id at Helm position {column}")
            if int(selected) not in spec.allowed_value_ids:
                raise ValueError(f"Illegal token value at Helm position {column}")
            legal_mask[row, column, list(spec.allowed_value_ids)] = True
            decision_mask[row, column] = len(spec.allowed_value_ids) > 1
            tokenizer.advance_generation(state, int(selected))
        if tokenizer.next_token_spec(state) is not None:
            raise ValueError("Incomplete Helm sequence without ROUTE_EOS")
        legal_mask[row, len(sequence) :, 0] = True
    legal_logits = (logits.float() / temperature).masked_fill(
        ~legal_mask.to(logits.device), float("-inf")
    )
    probabilities = epsilon_greedy_probs(
        legal_logits, epsilon=epsilon, top_k=top_k, min_prob=min_prob
    )
    log_probs = (
        probabilities.gather(-1, values.unsqueeze(-1))
        .squeeze(-1)
        .clamp_min(1e-12)
        .log()
    )
    return (log_probs, decision_mask.to(logits.device))
