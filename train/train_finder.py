"""Train the board finder on whole scenes, synthetic and real.

    python -m train.train_finder --scenes data/scenes/train --test data/scenes/test \\
        --labels eval/private/labels/my-photos.json --photos eval/private/my-photos --out runs/finder-v1

Real photos are optional. With --test-source (repeatable) the named sources
are kept out of training and the finder is scored on them.

A corner error is the distance between a found corner and the labelled one,
as a share of the board's side; a board's error is that of its worst corner.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from boardscan.pipeline import find_board

from .evaluate import pick_device
from .finder import GRID, BoardFinder, parameter_count
from .finder_data import Scene, batch_of, load_scenes
from .train import augment

# The classifier is trained on crops whose corners are off by up to this share of the board's side.
TOLERANCE = 0.025


def torch_finder(model: BoardFinder, device: torch.device):
    @torch.no_grad()
    def finder(view: np.ndarray):
        corners, present, _ = model(torch.from_numpy(view).to(device).permute(2, 0, 1)[None].float())
        return corners[0].cpu().numpy(), float(present[0])

    return finder


def score_finder(finder, scenes: list[Scene], looks: int = 2) -> dict:
    """How closely `finder` locates the board in each whole scene."""
    if not scenes:
        return {"scenes": 0}
    errors, seen = [], []
    for scene in scenes:
        corners, in_view = find_board(finder, scene.pixels(), looks)
        side = np.mean([np.linalg.norm(scene.corners[i] - scene.corners[(i + 1) % 4]) for i in range(4)])
        errors.append(float(np.linalg.norm(corners - scene.corners, axis=1).max() / side))
        seen.append(in_view)
    errors = np.array(errors)
    return {
        "scenes": len(scenes),
        "median_error": round(float(np.median(errors)), 4),
        "error_95th_percentile": round(float(np.percentile(errors, 95)), 4),
        "worst_error": round(float(errors.max()), 4),
        "within_tolerance": int((errors <= TOLERANCE).sum()),
        "board_reported_in_view": int((np.array(seen) >= 0.5).sum()),
    }


def heatmap_targets(corners: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """A small blob at each corner, N x 4 x GRID x GRID, each summing to one."""
    cells = torch.arange(GRID, device=corners.device, dtype=torch.float32)
    across = torch.exp(-((cells - (corners[..., 0:1] * GRID - 0.5)) ** 2) / (2 * sigma**2))
    down = torch.exp(-((cells - (corners[..., 1:2] * GRID - 0.5)) ** 2) / (2 * sigma**2))
    blobs = down[..., :, None] * across[..., None, :]
    return blobs / blobs.sum(dim=(2, 3), keepdim=True).clamp_min(1e-9)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenes", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--labels", action="append", default=[], help="label file; repeat with --photos for several photo sets")
    parser.add_argument("--photos", action="append", default=[])
    parser.add_argument("--test-source", action="append", default=[])
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--real-share", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-scenes", type=int, default=300, help="synthetic test scenes to score on")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    synthetic, synthetic_test = load_scenes(args.scenes), load_scenes(args.test)[: args.test_scenes]
    real_train, real_test = [], []
    if args.labels:
        from .finetune import split
        from .real import load_labelled_sets

        train_boards, test_boards = split(load_labelled_sets(args.labels, args.photos), args.test_source, None, args.seed)
        real_train = [Scene(None, board.corners, board.image) for board in train_boards]
        real_test = [Scene(None, board.corners, board.image) for board in test_boards]
    real_count = round(args.batch * args.real_share) if real_train else 0

    model = BoardFinder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.02)
    warmup = min(300, args.steps // 10)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (step + 1) / warmup if step < warmup else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, args.steps - warmup))),
    )
    print(
        f"device {device}, {parameter_count(model):,} parameters; {len(synthetic):,} synthetic scenes, {len(real_train)} real photos to train on, "
        f"{len(real_test)} real photos held out; {real_count} real views in each batch of {args.batch}",
        flush=True,
    )

    history, began = [], time.time()
    pool = ThreadPoolExecutor(2)
    model.train()
    for step in range(1, args.steps + 1):
        columns = batch_of(synthetic, args.batch - real_count, rng, pool)
        if real_count:
            columns = [np.concatenate(pair) for pair in zip(columns, batch_of(real_train, real_count, rng, pool))]
        images = augment(torch.from_numpy(columns[0]).to(device).permute(0, 3, 1, 2).float(), generator)
        target = torch.from_numpy(columns[1]).to(device)
        present = torch.from_numpy(columns[2]).to(device)
        corners, present_logit, logits = model(images)
        shown = present.bool()
        place = (corners[shown] - target[shown]).abs().mean()
        blobs = -(heatmap_targets(target[shown]).flatten(2) * F.log_softmax(logits[shown].flatten(2), dim=2)).sum(dim=2).mean()
        loss = 20 * place + blobs + 0.5 * F.binary_cross_entropy_with_logits(present_logit, present)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        schedule.step()
        if step % 200 == 0:
            print(f"  step {step}/{args.steps}  loss {loss.item():.3f}  corner offset {place.item():.4f}  {(time.time() - began) / 60:.1f} min", flush=True)
        if step % 2000 == 0 or step == args.steps:
            model.eval()
            finder = torch_finder(model, device)
            entry = {"step": step, "minutes": round((time.time() - began) / 60, 1), "synthetic": score_finder(finder, synthetic_test), "real_held_out": score_finder(finder, real_test)}
            if step == args.steps:
                entry["synthetic_one_look"] = score_finder(finder, synthetic_test, looks=1)
                entry["real_held_out_one_look"] = score_finder(finder, real_test, looks=1)
            history.append(entry)
            print(json.dumps(entry), flush=True)
            torch.save(model.state_dict(), out / "model.pt")
            (out / "history.json").write_text(json.dumps({"test_sources": args.test_source, "real_train": len(real_train), "history": history}, indent=2) + "\n")
            model.train()


if __name__ == "__main__":
    main()
