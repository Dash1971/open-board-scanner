"""Reading the crop shards written by train.make_crops."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def shard_starts(folder: str | Path) -> list[str]:
    return sorted(path.stem.split("-")[1] for path in Path(folder).glob("labels-*.npz"))


def load_shard(folder: str | Path, start: str) -> dict[str, np.ndarray]:
    labels = np.load(Path(folder) / f"labels-{start}.npz")
    shard = {name: labels[name] for name in labels.files}
    shard["images"] = np.load(Path(folder) / f"images-{start}.npy")
    return shard


def load_all(folder: str | Path) -> dict[str, np.ndarray]:
    shards = [load_shard(folder, start) for start in shard_starts(folder)]
    return {name: np.concatenate([shard[name] for shard in shards]) for name in shards[0]}


def batches(folder: str | Path, batch_size: int, rng: np.random.Generator, group: int = 4):
    """Yield shuffled (images, squares, white_at_bottom) batches, mixing `group` shards at a time."""
    starts = shard_starts(folder)
    order = rng.permutation(len(starts))
    for position in range(0, len(order), group):
        loaded = [load_shard(folder, starts[i]) for i in order[position : position + group]]
        images = np.concatenate([shard["images"] for shard in loaded])
        squares = np.concatenate([shard["squares"] for shard in loaded])
        white = np.concatenate([shard["white_at_bottom"] for shard in loaded])
        shuffle = rng.permutation(len(images))
        for offset in range(0, len(shuffle) - batch_size + 1, batch_size):
            chosen = np.sort(shuffle[offset : offset + batch_size])
            yield images[chosen], squares[chosen], white[chosen]
