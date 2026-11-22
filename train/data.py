"""Reading the crop shards written by train.make_crops."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Shards written before piece-set names were stored used this fixed list.
LEGACY_SET_NAMES = (
    "cburnett", "celtic", "chessnut", "fantasy", "firi", "merida", "mpchess",
    "papercut", "pirouetti", "pixel", "rhosgfx", "spatial", "totoy",
)  # fmt: skip


def shard_starts(folder: str | Path) -> list[str]:
    return sorted(path.stem.split("-")[1] for path in Path(folder).glob("labels-*.npz"))


def load_shard(folder: str | Path, start: str) -> dict[str, np.ndarray]:
    labels = np.load(Path(folder) / f"labels-{start}.npz")
    shard = {name: labels[name] for name in labels.files}
    shard["images"] = np.load(Path(folder) / f"images-{start}.npy")
    shard.setdefault("quarter_turns", np.zeros(len(shard["images"]), np.uint8))
    shard.setdefault("set_names", np.array(LEGACY_SET_NAMES))
    return shard


def load_all(folder: str | Path) -> dict[str, np.ndarray]:
    shards = [load_shard(folder, start) for start in shard_starts(folder)]
    data = {name: np.concatenate([shard[name] for shard in shards]) for name in shards[0] if name != "set_names"}
    data["set_names"] = shards[0]["set_names"]
    return data


def batches(folder: str | Path, batch_size: int, rng: np.random.Generator, group: int = 2):
    """Yield shuffled (images, squares, white_at_bottom, quarter_turns) batches, mixing `group` shards at a time."""
    starts = shard_starts(folder)
    order = rng.permutation(len(starts))
    for position in range(0, len(order), group):
        loaded = [load_shard(folder, starts[i]) for i in order[position : position + group]]
        columns = [np.concatenate([shard[name] for shard in loaded]) for name in ("images", "squares", "white_at_bottom", "quarter_turns")]
        shuffle = rng.permutation(len(columns[0]))
        for offset in range(0, len(shuffle) - batch_size + 1, batch_size):
            chosen = np.sort(shuffle[offset : offset + batch_size])
            yield tuple(column[chosen] for column in columns)
