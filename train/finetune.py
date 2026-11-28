"""Fine-tune a trained classifier on labelled real photos, mixed with synthetic crops.

    python -m train.finetune --labels eval/private/labels/my-photos.json --photos eval/private/my-photos \\
        --start runs/boardnet-v1/model.pt --synthetic data/crops/train-v1 --out runs/boardnet-v2

Boards are held out for testing in one of two ways: whole sources by name
(--test-source, repeatable), which says how the model does on a book it has
never seen, and every k-th board (--fold I K), which says how it does on
unseen pages of books it has seen. The two can be combined. With neither,
every labelled board is trained on and nothing real is left to test with.
--compare scores an earlier checkpoint on the same held-out boards.

Held-out boards are scored twice: read once as they lie, and read at all four
quarter turns with the readings averaged.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import batches, load_all
from .evaluate import pick_device, score
from .model import BoardNet
from .real import load_labelled_sets, real_batch, score_real, turn_crops
from .train import augment


FOUR_TURNS = (0, 1, 2, 3)  # reading a crop at every quarter turn and averaging costs four passes


def split(boards, test_sources: list[str], fold: tuple[int, int] | None, seed: int, fold_pool: int | None = None):
    """Divide labelled boards into (train, test).

    Boards from `test_sources` are held out, and so is every k-th board of
    `fold`. With `fold_pool`, the fold is drawn from the first that many
    boards only, so that a later run with more photos holds out exactly the
    boards an earlier run did.
    """
    unknown = set(test_sources) - {board.source for board in boards}
    if unknown:
        raise SystemExit(f"no labelled boards from {sorted(unknown)}")
    held: set[int] = {i for i, board in enumerate(boards) if board.source in test_sources}
    if fold:
        index, folds = fold
        pool = len(boards) if fold_pool is None else fold_pool
        held |= set(np.random.default_rng(seed).permutation(pool)[index::folds].tolist())
    return [b for i, b in enumerate(boards) if i not in held], [b for i, b in enumerate(boards) if i in held]


def synthetic_batches(folder: str, count: int, rng: np.random.Generator):
    while True:
        yield from batches(folder, count, rng)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", action="append", required=True, help="label file; repeat with --photos for several photo sets")
    parser.add_argument("--photos", action="append", required=True)
    parser.add_argument("--start", required=True, help="checkpoint to start from")
    parser.add_argument("--synthetic", required=True, help="folder of synthetic training crops to mix in")
    parser.add_argument("--synthetic-test", help="folder of synthetic test crops, to check nothing was forgotten")
    parser.add_argument("--out", required=True)
    parser.add_argument("--test-source", action="append", default=[])
    parser.add_argument("--fold", type=int, nargs=2, metavar=("I", "K"))
    parser.add_argument("--fold-sets", type=int, help="draw the fold from the first this-many --labels files only")
    parser.add_argument("--compare", action="append", default=[], help="another checkpoint to score on the same held-out boards (repeatable)")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--real-share", type=float, default=0.5, help="share of each batch drawn from real photos")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    fold_pool = len(load_labelled_sets(args.labels[: args.fold_sets], args.photos[: args.fold_sets], boards_only=True)) if args.fold_sets else None
    train, test = split(load_labelled_sets(args.labels, args.photos, boards_only=True), args.test_source, tuple(args.fold) if args.fold else None, args.seed, fold_pool)
    if not train:
        raise SystemExit("no labelled boards left to train on")
    model = BoardNet().to(device)
    model.load_state_dict(torch.load(args.start, map_location=device))
    synthetic_test = load_all(args.synthetic_test) if args.synthetic_test else None
    report = {
        "start": args.start, "steps": args.steps, "lr": args.lr, "real_share": args.real_share, "seed": args.seed,
        "train_boards": len(train), "train_sources": sorted({b.source for b in train}),
        "test_boards": len(test), "test_sources": sorted({b.source for b in test}),
        "before": {"real_test": score_real(model, test, device), "real_test_four_turns": score_real(model, test, device, FOUR_TURNS), "real_train": score_real(model, train, device)},
    }  # fmt: skip
    if synthetic_test:
        report["before"]["synthetic"] = score(model, synthetic_test, device)["all"]
    report["compare"] = {}
    for checkpoint in args.compare:  # an earlier model on the same held-out boards
        other = BoardNet().to(device)
        other.load_state_dict(torch.load(checkpoint, map_location=device))
        report["compare"][checkpoint] = {"real_test": score_real(other, test, device), "real_test_four_turns": score_real(other, test, device, FOUR_TURNS)}
        del other

    real_count = max(1, min(args.batch - 1, round(args.batch * args.real_share)))
    synthetic = synthetic_batches(args.synthetic, args.batch - real_count, rng)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.02)
    warmup = max(1, args.steps // 20)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (step + 1) / warmup if step < warmup else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, args.steps - warmup))),
    )
    print(f"device {device}; {len(train)} real boards to train on, {len(test)} held out; {real_count} real and {args.batch - real_count} synthetic crops per step", flush=True)
    began = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        real = real_batch(train, real_count, rng)
        images, squares, white, turns = next(synthetic)
        images, squares, turns = turn_crops(images, squares, turns, rng)
        columns = [np.concatenate([a, b]) for a, b in zip(real, (images, squares, white, turns))]
        images = augment(torch.from_numpy(columns[0]).to(device).permute(0, 3, 1, 2).float(), generator)
        logits, orientation, rotation = model(images)
        loss = (
            F.cross_entropy(logits, torch.from_numpy(columns[1]).to(device).long().view(-1, 8, 8))
            + 0.2 * F.binary_cross_entropy_with_logits(orientation, torch.from_numpy(columns[2]).to(device).float())
            + 0.2 * F.cross_entropy(rotation, torch.from_numpy(columns[3]).to(device).long())
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        schedule.step()
        if step % max(100, args.steps // 30) == 0 or step == args.steps:
            held = score_real(model, test, device)
            print(f"  step {step}/{args.steps}  loss {loss.item():.4f}  held-out exact {held.get('exact_boards', '-')}/{held['boards']}  wrong squares {held.get('wrong_squares', '-')}  {(time.time() - began) / 60:.1f} min", flush=True)
            torch.save(model.state_dict(), out / "checkpoint.pt")  # so a stopped run is not lost; pass it as --start to carry on
            model.train()

    report["minutes"] = round((time.time() - began) / 60, 1)
    report["after"] = {"real_test": score_real(model, test, device), "real_test_four_turns": score_real(model, test, device, FOUR_TURNS), "real_train": score_real(model, train, device)}
    if synthetic_test:
        report["after"]["synthetic"] = score(model, synthetic_test, device)["all"]
    torch.save(model.state_dict(), out / "model.pt")
    (out / "finetune.json").write_text(json.dumps(report, indent=2) + "\n")
    for when in ("before", "after"):
        held, voted, seen = report[when]["real_test"], report[when]["real_test_four_turns"], report[when]["real_train"]
        line = (
            f"{when:6s} held-out exact {held.get('exact_boards', '-')}/{held['boards']}, wrong squares {held.get('wrong_squares', '-')}; "
            f"read at four turns: exact {voted.get('exact_boards', '-')}/{voted['boards']}, wrong squares {voted.get('wrong_squares', '-')}; "
            f"trained-on exact {seen['exact_boards']}/{seen['boards']}"
        )
        if synthetic_test:
            line += f"; synthetic exact {report[when]['synthetic']['exact_boards']:.3f}"
        print(line)


if __name__ == "__main__":
    main()
