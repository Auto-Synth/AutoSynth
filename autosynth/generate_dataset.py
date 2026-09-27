from __future__ import annotations
import argparse
import hashlib
import json
import logging
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import h5py
import librosa
import numpy as np
import yaml
from autosynth.synth.helm_augmentation import AugmentedPatch, HelmAugmenter
from autosynth.synth.helm_midi import HelmMidiSampler, MidiPerformance
from autosynth.synth.helm_parameter_spec import HelmParameterSpace
from autosynth.synth.helm_renderer import HelmDawRenderer
from autosynth.synth.helm_subprocess_renderer import (
    HelmNativeRenderError,
    SubprocessHelmRenderer,
)
from autosynth.synth.helm_state import HelmPresetSeed, ROUTE_AMOUNT_ENCODING
from autosynth.paths import (
    project_relative_string,
    require_project_relative_path,
    resolve_project_path,
)

logger = logging.getLogger(__name__)
AUDIO_CHANNELS = 2
AUDIO_LAYOUT = "channels_first"
SPLIT_IDS = {"train": 0, "val": 1, "test": 2}


@dataclass(frozen=True)
class Recipe:
    source: Path
    output_root: Path
    plugin_path: Path
    preset_root: Path
    parameter_schema_path: Path
    catalog_cache_path: Path
    num_samples: int
    sample_rate: int
    block_size: int
    target_duration: float
    min_rms: float
    max_abs_peak: float
    max_tries: int
    seed: int
    train_ratio: float
    val_ratio: float
    n_mels: int
    window_seconds: float
    frames_per_second: float
    window: str
    state_load_flush_seconds: float
    tail_flush_seconds: float
    fadeout_seconds: float
    auto_start_xvfb: bool
    xvfb_path: str
    render_timeout_seconds: float
    max_renders_per_worker: int
    fresh_plugin_per_render: bool
    note_min: int
    note_max: int
    velocity_min: int
    velocity_max: int
    duration_min: float
    duration_max: float
    controller_excitation_probability: float
    preset_probability: float
    category_uniform_probability: float
    edit_geometric_p: float
    max_parameter_edits: int
    route_edit_probability: float
    coverage_probability: float
    route_zero_threshold: float
    wild_parameter_strength_multiplier: float
    log_interval: int
    shard_counts: Mapping[str, int]


@dataclass(frozen=True)
class GeneratedSample:
    sample_id: str
    audio: np.ndarray
    mel: np.ndarray
    patch: AugmentedPatch
    performance: MidiPerformance


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Recipe section `{key}` must be a mapping")
    return value


def load_recipe(path: str | Path) -> Recipe:
    source = resolve_project_path(path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"Helm recipe must be a mapping: {source}")
    paths = _mapping(raw, "paths")
    dataset = _mapping(raw, "dataset")
    mel = _mapping(raw, "mel")
    render = _mapping(raw, "render")
    midi = _mapping(raw, "midi")
    augmentation = _mapping(raw, "augmentation")
    runtime = _mapping(raw, "runtime")
    sharding = _mapping(raw, "sharding")
    split = _mapping(dataset, "split")
    if bool(midi.get("short_melodies", False)):
        raise ValueError(
            "Helm short melodies are intentionally disabled in this generator"
        )
    recipe = Recipe(
        source=source,
        output_root=require_project_relative_path(
            str(paths["output_root"]), label="paths.output_root", source=source
        ),
        plugin_path=require_project_relative_path(
            str(paths["plugin_path"]), label="paths.plugin_path", source=source
        ),
        preset_root=require_project_relative_path(
            str(paths["preset_root"]), label="paths.preset_root", source=source
        ),
        parameter_schema_path=require_project_relative_path(
            str(paths["parameter_schema_path"]),
            label="paths.parameter_schema_path",
            source=source,
        ),
        catalog_cache_path=require_project_relative_path(
            str(paths["catalog_cache_path"]),
            label="paths.catalog_cache_path",
            source=source,
        ),
        num_samples=int(dataset["num_samples"]),
        sample_rate=int(dataset["sample_rate"]),
        block_size=int(dataset["block_size"]),
        target_duration=float(dataset["target_duration"]),
        min_rms=float(dataset["min_rms"]),
        max_abs_peak=float(dataset["max_abs_peak"]),
        max_tries=int(dataset["max_tries"]),
        seed=int(dataset["seed"]),
        train_ratio=float(split["train"]),
        val_ratio=float(split["val"]),
        n_mels=int(mel["n_mels"]),
        window_seconds=float(mel["window_seconds"]),
        frames_per_second=float(mel["frames_per_second"]),
        window=str(mel["window"]),
        state_load_flush_seconds=float(render["state_load_flush_seconds"]),
        tail_flush_seconds=float(render["tail_flush_seconds"]),
        fadeout_seconds=float(render["fadeout_seconds"]),
        auto_start_xvfb=bool(render["auto_start_xvfb"]),
        xvfb_path=str(render["xvfb_path"]),
        render_timeout_seconds=float(render["render_timeout_seconds"]),
        max_renders_per_worker=int(render["max_renders_per_worker"]),
        fresh_plugin_per_render=bool(render["fresh_plugin_per_render"]),
        note_min=int(midi["note_min"]),
        note_max=int(midi["note_max"]),
        velocity_min=int(midi["velocity_min"]),
        velocity_max=int(midi["velocity_max"]),
        duration_min=float(midi["duration_min"]),
        duration_max=float(midi["duration_max"]),
        controller_excitation_probability=float(
            midi["controller_excitation_probability"]
        ),
        preset_probability=float(augmentation["preset_probability"]),
        category_uniform_probability=float(
            augmentation["category_uniform_probability"]
        ),
        edit_geometric_p=float(augmentation["edit_geometric_p"]),
        max_parameter_edits=int(augmentation["max_parameter_edits"]),
        route_edit_probability=float(augmentation["route_edit_probability"]),
        coverage_probability=float(augmentation["coverage_probability"]),
        route_zero_threshold=float(augmentation["route_zero_threshold"]),
        wild_parameter_strength_multiplier=float(
            augmentation["wild_parameter_strength_multiplier"]
        ),
        log_interval=int(runtime["log_interval"]),
        shard_counts={name: int(sharding[name]) for name in ("train", "val", "test")},
    )
    if recipe.num_samples < 3 or recipe.sample_rate <= 0 or recipe.block_size <= 0:
        raise ValueError("Invalid dataset size, sample rate, or block size")
    if (
        recipe.train_ratio <= 0
        or recipe.val_ratio <= 0
        or recipe.train_ratio + recipe.val_ratio >= 1
    ):
        raise ValueError("Split ratios must be positive and sum to less than one")
    if abs(recipe.preset_probability - 0.9) > 1e-09:
        raise ValueError("The approved Helm recipe requires preset_probability: 0.90")
    if any((count <= 0 for count in recipe.shard_counts.values())):
        raise ValueError("Every split must have at least one shard")
    return recipe


def split_factory_presets(
    factory: Sequence[HelmPresetSeed], init: HelmPresetSeed, recipe: Recipe
) -> dict[str, list[HelmPresetSeed]]:
    result: dict[str, list[HelmPresetSeed]] = {"train": [init], "val": [], "test": []}
    train_threshold = recipe.train_ratio
    val_threshold = recipe.train_ratio + recipe.val_ratio
    for seed in factory:
        raw = hashlib.sha1(f"{recipe.seed}:{seed.preset_id}".encode("utf-8")).digest()[
            :8
        ]
        unit = int.from_bytes(raw, "big") / float(2**64)
        split = (
            "train"
            if unit < train_threshold
            else "val"
            if unit < val_threshold
            else "test"
        )
        result[split].append(seed)
    for split, seeds in result.items():
        if not seeds:
            raise RuntimeError(f"Hash split produced no preset seeds for {split}")
    return result


def split_counts(total: int, train_ratio: float, val_ratio: float) -> dict[str, int]:
    weights = np.asarray(
        (train_ratio, val_ratio, 1.0 - train_ratio - val_ratio), dtype=np.float64
    )
    raw = weights * total
    counts = np.floor(raw).astype(np.int64)
    for index in np.argsort(-(raw - counts))[: total - int(counts.sum())]:
        counts[int(index)] += 1
    return {
        name: int(counts[index]) for index, name in enumerate(("train", "val", "test"))
    }


def partition(total: int, parts: int, index: int) -> tuple[int, int]:
    if not 0 <= index < parts:
        raise ValueError(f"shard-index must be in [0,{parts - 1}], got {index}")
    base, remainder = divmod(total, parts)
    start = index * base + min(index, remainder)
    size = base + int(index < remainder)
    return (start, size)


def sample_seed(
    rng: np.random.Generator,
    seeds: Sequence[HelmPresetSeed],
    category_uniform_probability: float,
) -> HelmPresetSeed:
    if rng.random() >= category_uniform_probability:
        return seeds[int(rng.integers(0, len(seeds)))]
    by_category: dict[str, list[HelmPresetSeed]] = defaultdict(list)
    for seed in seeds:
        by_category[seed.category].append(seed)
    category = str(rng.choice(sorted(by_category)))
    choices = by_category[category]
    return choices[int(rng.integers(0, len(choices)))]


def make_mel(audio: np.ndarray, recipe: Recipe) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 2 or audio.shape[0] != AUDIO_CHANNELS:
        raise ValueError(
            f"Expected stereo audio shaped ({AUDIO_CHANNELS}, samples), got {audio.shape}"
        )
    n_fft = int(recipe.window_seconds * recipe.sample_rate)
    hop_length = int(recipe.sample_rate / recipe.frames_per_second)
    power = np.stack(
        [
            librosa.feature.melspectrogram(
                y=channel,
                sr=recipe.sample_rate,
                n_mels=recipe.n_mels,
                n_fft=n_fft,
                hop_length=hop_length,
                window=recipe.window,
            )
            for channel in audio
        ],
        axis=0,
    )
    return librosa.power_to_db(power, ref=np.max).astype(np.float32)


def fit_audio(audio: np.ndarray, length: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 2 or audio.shape[0] != AUDIO_CHANNELS:
        raise ValueError(
            f"Expected stereo audio shaped ({AUDIO_CHANNELS}, samples), got {audio.shape}"
        )
    if audio.shape[1] >= length:
        return audio[:, :length]
    return np.pad(audio, ((0, 0), (0, length - audio.shape[1])))


def generate_one(
    *,
    split: str,
    global_index: int,
    use_preset: bool,
    seeds: Sequence[HelmPresetSeed],
    init: HelmPresetSeed,
    augmenter: HelmAugmenter,
    midi_sampler: HelmMidiSampler,
    renderer: HelmDawRenderer,
    recipe: Recipe,
    seen_fingerprints: set[str],
) -> GeneratedSample:
    audio_length = int(round(recipe.target_duration * recipe.sample_rate))
    for attempt in range(recipe.max_tries):
        rng = np.random.default_rng(
            np.random.SeedSequence(
                (recipe.seed, SPLIT_IDS[split], global_index, attempt, 731)
            )
        )
        if use_preset:
            seed = sample_seed(rng, seeds, recipe.category_uniform_probability)
            patch = augmenter.augment_preset(rng, seed)
        else:
            patch = augmenter.random_patch(rng, init)
        performance = midi_sampler.sample(
            rng,
            category=patch.category,
            preset_name=patch.source_name,
            parameters=patch.parameters,
            routes=patch.routes,
        )
        fingerprint = sample_fingerprint(patch, performance)
        if fingerprint in seen_fingerprints:
            continue
        try:
            audio = renderer.render(
                base_state_payload=patch.base_state_payload,
                parameters=patch.parameters,
                routes=patch.routes,
                performance=performance,
                duration_seconds=recipe.target_duration,
            )
        except HelmNativeRenderError as exc:
            logger.warning(
                "Retrying after native Helm render failure at index %d attempt %d: %s",
                global_index,
                attempt,
                exc,
            )
            continue
        audio = fit_audio(audio, audio_length)
        if not np.isfinite(audio).all():
            continue
        rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
        peak = float(np.max(np.abs(audio)))
        if rms < recipe.min_rms or peak > recipe.max_abs_peak:
            continue
        mel = make_mel(audio, recipe)
        seen_fingerprints.add(fingerprint)
        sample_id = hashlib.sha1(
            f"{split}:{recipe.seed}:{global_index}:{attempt}:{fingerprint}".encode()
        ).hexdigest()[:20]
        return GeneratedSample(sample_id, audio, mel, patch, performance)
    raise RuntimeError(
        f"Could not generate valid Helm sample after {recipe.max_tries} tries: split={split}, index={global_index}, source={('preset' if use_preset else 'random')}"
    )


def sample_fingerprint(patch: AugmentedPatch, performance: MidiPerformance) -> str:
    digest = hashlib.sha1(np.asarray(patch.parameters, dtype="<f4").tobytes())
    for route in patch.routes:
        digest.update(
            f"{route.source}>{route.destination}:{route.amount:.7f};".encode()
        )
    for note in performance.notes:
        digest.update(
            f"n:{note.pitch}:{note.velocity}:{note.onset:.5f}:{note.duration:.5f};".encode()
        )
    for control in performance.controls:
        digest.update(f"c:{control.kind}:{control.value}:{control.time:.5f};".encode())
    return digest.hexdigest()


def create_h5(
    path: Path, count: int, audio_length: int, mel_shape: tuple[int, int, int]
) -> h5py.File:
    h5 = h5py.File(path, "w")
    h5.attrs["route_amount_encoding"] = ROUTE_AMOUNT_ENCODING
    h5.attrs["split_rng_version"] = 1
    h5.create_dataset("audio", (count, AUDIO_CHANNELS, audio_length), dtype=np.float16)
    h5.create_dataset("mel_spec", (count, *mel_shape), dtype=np.float32)
    h5.create_dataset("parameters", (count, 139), dtype=np.float32)
    h5.create_dataset("changed_parameter_mask", (count, 139), dtype=np.uint8)
    string_type = h5py.string_dtype(encoding="utf-8")
    for name in (
        "sample_id",
        "source_type",
        "source_preset_id",
        "source_name",
        "category",
        "role",
        "performance_type",
        "chord_name",
        "operations",
    ):
        h5.create_dataset(name, (count,), dtype=string_type)
    h5.create_dataset(
        "route_pair_ids", (0,), maxshape=(None,), dtype=np.int16, chunks=True
    )
    h5.create_dataset(
        "route_amounts", (0,), maxshape=(None,), dtype=np.float32, chunks=True
    )
    h5.create_dataset("route_offsets", (count + 1,), dtype=np.int64)
    h5.create_dataset(
        "midi_note_events", (0, 4), maxshape=(None, 4), dtype=np.float32, chunks=True
    )
    h5.create_dataset("midi_note_offsets", (count + 1,), dtype=np.int64)
    h5.create_dataset(
        "midi_control_events", (0, 3), maxshape=(None, 3), dtype=np.float32, chunks=True
    )
    h5.create_dataset("midi_control_offsets", (count + 1,), dtype=np.int64)
    return h5


def append_sample(
    h5: h5py.File, index: int, sample: GeneratedSample, space: HelmParameterSpace
) -> None:
    patch, performance = (sample.patch, sample.performance)
    h5["audio"][index] = sample.audio.astype(np.float16)
    h5["mel_spec"][index] = sample.mel
    h5["parameters"][index] = patch.parameters
    h5["changed_parameter_mask"][index] = patch.changed_parameter_mask
    text_values = {
        "sample_id": sample.sample_id,
        "source_type": patch.source_type,
        "source_preset_id": patch.source_preset_id,
        "source_name": patch.source_name,
        "category": patch.category,
        "role": patch.role,
        "performance_type": performance.kind,
        "chord_name": performance.chord_name,
        "operations": json.dumps(patch.operations, ensure_ascii=False),
    }
    for name, value in text_values.items():
        h5[name][index] = value
    route_rows = [
        (space.route_pair_id(route.source, route.destination), route.amount)
        for route in patch.routes
    ]
    _append_vector(h5["route_pair_ids"], [row[0] for row in route_rows])
    _append_vector(h5["route_amounts"], [row[1] for row in route_rows])
    h5["route_offsets"][index + 1] = h5["route_pair_ids"].shape[0]
    note_rows = [
        [note.pitch, note.velocity, note.onset, note.duration]
        for note in performance.notes
    ]
    _append_matrix(h5["midi_note_events"], note_rows, 4)
    h5["midi_note_offsets"][index + 1] = h5["midi_note_events"].shape[0]
    control_kind = {"mod_wheel": 1, "aftertouch": 2, "pitch_bend": 3}
    control_rows = [
        [control_kind[control.kind], control.value, control.time]
        for control in performance.controls
    ]
    _append_matrix(h5["midi_control_events"], control_rows, 3)
    h5["midi_control_offsets"][index + 1] = h5["midi_control_events"].shape[0]


def _append_vector(dataset: h5py.Dataset, values: Sequence[float | int]) -> None:
    old_size = int(dataset.shape[0])
    dataset.resize((old_size + len(values),))
    if values:
        dataset[old_size:] = values


def _append_matrix(
    dataset: h5py.Dataset, rows: Sequence[Sequence[float]], width: int
) -> None:
    old_size = int(dataset.shape[0])
    dataset.resize((old_size + len(rows), width))
    if rows:
        dataset[old_size:] = np.asarray(rows, dtype=np.float32)


def generation_schedule(
    count: int, preset_probability: float, rng: np.random.Generator
) -> np.ndarray:
    preset_count = int(round(count * preset_probability))
    schedule = np.zeros(count, dtype=np.uint8)
    schedule[:preset_count] = 1
    rng.shuffle(schedule)
    return schedule


def output_path_for(recipe: Recipe, split: str, shard_index: int) -> Path:
    shard_count = recipe.shard_counts[split]
    return (
        recipe.output_root
        / "shards"
        / f"{split}-{shard_index:05d}-of-{shard_count:05d}.h5"
    )


def write_metadata(
    recipe: Recipe, split_seeds: Mapping[str, Sequence[HelmPresetSeed]]
) -> None:
    path = recipe.output_root / "dataset_metadata.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("route_amount_encoding") != ROUTE_AMOUNT_ENCODING:
            raise ValueError("Use a new dataset directory for destination-normalized routes")
        if existing.get("split_rng_version") != 1:
            raise ValueError("Use a new dataset directory for split-specific random streams")
    elif any((recipe.output_root / "shards").glob("*.h5")):
        raise ValueError("Dataset shards without metadata cannot be reused; use a new directory")
    audio_length = int(round(recipe.target_duration * recipe.sample_rate))
    mel_frames = int(
        make_mel(
            np.zeros((AUDIO_CHANNELS, audio_length), dtype=np.float32), recipe
        ).shape[-1]
    )
    payload = {
        "format_version": 4,
        "route_amount_encoding": ROUTE_AMOUNT_ENCODING,
        "split_rng_version": 1,
        "synth": "Helm",
        "audio_channels": AUDIO_CHANNELS,
        "audio_layout": AUDIO_LAYOUT,
        "mel_channels": AUDIO_CHANNELS,
        "mel_layout": "channels_mels_frames",
        "mel_frames": mel_frames,
        "mel": {
            "n_mels": recipe.n_mels,
            "window_seconds": recipe.window_seconds,
            "frames_per_second": recipe.frames_per_second,
            "window": recipe.window,
        },
        "use_saved_mean_and_variance": False,
        "sample_rate": recipe.sample_rate,
        "target_duration": recipe.target_duration,
        "fixed_parameter_count": 139,
        "modulation_source_count": 13,
        "modulation_destination_count": 81,
        "route_pair_count": 1053,
        "factory_preset_count": 274,
        "seed_count_including_init": 275,
        "preset_to_random_ratio": "9:1",
        "delay_policy": "canonicalized_off",
        "wild_parameter_strength_multiplier": recipe.wild_parameter_strength_multiplier,
        "performance_types": ["single_note", "chord"],
        "short_melodies_enabled": False,
        "split_seed_counts": {name: len(seeds) for name, seeds in split_seeds.items()},
        "parameter_schema": project_relative_string(recipe.parameter_schema_path),
        "recipe": project_relative_string(recipe.source),
    }
    recipe.output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=recipe.output_root, suffix=".tmp", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> Path | None:
    recipe = load_recipe(args.config)
    renderer = HelmDawRenderer(
        recipe.plugin_path,
        parameter_schema_path=recipe.parameter_schema_path,
        sample_rate=recipe.sample_rate,
        block_size=recipe.block_size,
        state_load_flush_seconds=recipe.state_load_flush_seconds,
        tail_flush_seconds=recipe.tail_flush_seconds,
        fadeout_seconds=recipe.fadeout_seconds,
        auto_start_xvfb=recipe.auto_start_xvfb,
        xvfb_path=recipe.xvfb_path,
    )
    safe_renderer = None
    try:
        init, factory = renderer.build_or_load_catalog(
            recipe.preset_root, recipe.catalog_cache_path, rebuild=args.rebuild_catalog
        )
        split_seeds = split_factory_presets(factory, init, recipe)
        write_metadata(recipe, split_seeds)
        logger.info(
            "Helm catalog ready: %d factory + Init; split seeds train=%d val=%d test=%d",
            len(factory),
            *(len(split_seeds[name]) for name in ("train", "val", "test")),
        )
        if args.catalog_only:
            return None
        split = args.split
        split_total = split_counts(
            recipe.num_samples, recipe.train_ratio, recipe.val_ratio
        )[split]
        shard_count = recipe.shard_counts[split]
        shard_start, configured_count = partition(
            split_total, shard_count, args.shard_index
        )
        count = (
            int(args.num_samples) if args.num_samples is not None else configured_count
        )
        if count <= 0 or int(args.start_offset) < 0:
            raise ValueError(
                "num-samples must be positive and start-offset must be non-negative"
            )
        global_start = (
            shard_start
            if args.num_samples is None
            else shard_start + int(args.start_offset)
        )
        out_path = (
            resolve_project_path(args.output)
            if args.output
            else output_path_for(recipe, split, args.shard_index)
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        space = renderer.space
        renderer.close()
        augmenter = HelmAugmenter(
            space,
            preset_probability=recipe.preset_probability,
            edit_geometric_p=recipe.edit_geometric_p,
            max_parameter_edits=recipe.max_parameter_edits,
            route_edit_probability=recipe.route_edit_probability,
            coverage_probability=recipe.coverage_probability,
            route_zero_threshold=recipe.route_zero_threshold,
            wild_parameter_strength_multiplier=recipe.wild_parameter_strength_multiplier,
        )
        safe_renderer = SubprocessHelmRenderer(
            plugin_path=recipe.plugin_path,
            parameter_schema_path=recipe.parameter_schema_path,
            sample_rate=recipe.sample_rate,
            block_size=recipe.block_size,
            state_load_flush_seconds=recipe.state_load_flush_seconds,
            tail_flush_seconds=recipe.tail_flush_seconds,
            fadeout_seconds=recipe.fadeout_seconds,
            auto_start_xvfb=recipe.auto_start_xvfb,
            xvfb_path=recipe.xvfb_path,
            timeout_seconds=recipe.render_timeout_seconds,
            max_renders_per_worker=recipe.max_renders_per_worker,
            fresh_plugin_per_render=recipe.fresh_plugin_per_render,
        )
        midi_sampler = HelmMidiSampler(
            space,
            note_min=recipe.note_min,
            note_max=recipe.note_max,
            velocity_min=recipe.velocity_min,
            velocity_max=recipe.velocity_max,
            duration_min=recipe.duration_min,
            duration_max=recipe.duration_max,
            controller_excitation_probability=recipe.controller_excitation_probability,
        )
        schedule_rng = np.random.default_rng(
            np.random.SeedSequence(
                (recipe.seed, SPLIT_IDS[split], global_start, count, 991)
            )
        )
        schedule = generation_schedule(count, recipe.preset_probability, schedule_rng)
        audio_length = int(round(recipe.target_duration * recipe.sample_rate))
        dummy_mel = make_mel(
            np.zeros((AUDIO_CHANNELS, audio_length), dtype=np.float32), recipe
        )
        temp_path = out_path.with_name(f".{out_path.name}.{os.getpid()}.tmp")
        seen: set[str] = set()
        mel_sum = np.zeros(dummy_mel.shape, dtype=np.float64)
        mel_sumsq = np.zeros(dummy_mel.shape, dtype=np.float64)
        with create_h5(temp_path, count, audio_length, dummy_mel.shape) as h5:
            h5.attrs.update(
                {
                    "synth": "Helm",
                    "split": split,
                    "shard_index": args.shard_index,
                    "shard_count": shard_count,
                    "sample_rate": recipe.sample_rate,
                    "audio_channels": AUDIO_CHANNELS,
                    "audio_layout": AUDIO_LAYOUT,
                    "audio_length": audio_length,
                    "mel_channels": AUDIO_CHANNELS,
                    "n_mels": recipe.n_mels,
                    "mel_frames": dummy_mel.shape[-1],
                    "parameter_count": 139,
                    "route_pair_count": 1053,
                    "preset_probability": recipe.preset_probability,
                }
            )
            for local_index in range(count):
                global_index = global_start + local_index
                sample = generate_one(
                    split=split,
                    global_index=global_index,
                    use_preset=bool(schedule[local_index]),
                    seeds=split_seeds[split],
                    init=init,
                    augmenter=augmenter,
                    midi_sampler=midi_sampler,
                    renderer=safe_renderer,
                    recipe=recipe,
                    seen_fingerprints=seen,
                )
                append_sample(h5, local_index, sample, space)
                mel64 = sample.mel.astype(np.float64)
                mel_sum += mel64
                mel_sumsq += mel64**2
                if (
                    local_index + 1
                ) % recipe.log_interval == 0 or local_index + 1 == count:
                    logger.info(
                        "%s shard %d: %d/%d",
                        split,
                        args.shard_index,
                        local_index + 1,
                        count,
                    )
        os.replace(temp_path, out_path)
        mean = mel_sum / max(count, 1)
        std = np.sqrt(np.maximum(mel_sumsq / max(count, 1) - mean**2, 0.0))
        np.savez(
            out_path.with_suffix(".stats.npz"),
            mel_mean=mean.astype(np.float32),
            mel_std=std.astype(np.float32),
            count=count,
        )
        return out_path
    finally:
        if safe_renderer is not None:
            safe_renderer.close()
        renderer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Override this shard's sample count",
    )
    parser.add_argument(
        "--start-offset", type=int, default=0, help="Used only with --num-samples"
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--catalog-only", action="store_true")
    parser.add_argument("--rebuild-catalog", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    output = run(args)
    if output is not None:
        print(output)


if __name__ == "__main__":
    main()
