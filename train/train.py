"""Train the board classifier on pre-rendered crops.

    python -m train.train --train data/crops/train --test data/crops/test --out runs/boardnet-v0
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

from .data import batches, load_all, shard_starts
from .evaluate import pick_device, score
from .model import BoardNet, parameter_count


def augment(images: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Cheap photometric changes on the GPU; geometry is already varied in the crops."""
    count, device = images.shape[0], images.device

    def uniform(low: float, high: float, *shape: int) -> torch.Tensor:
        return torch.rand(count, *shape, generator=generator, device=device) * (high - low) + low

    grey = images.mean(dim=1, keepdim=True)
    images = torch.where(uniform(0, 1, 1, 1, 1) < 0.15, grey.expand_as(images), images)
    mean = images.mean(dim=(1, 2, 3), keepdim=True)
    images = (images - mean) * uniform(0.7, 1.4, 1, 1, 1) + mean * uniform(0.75, 1.25, 1, 1, 1)
    images = images * uniform(0.92, 1.08, 3, 1, 1)
    noise = torch.randn(images.shape, generator=generator, device=device) * uniform(0, 8, 1, 1, 1)
    soft = F.interpolate(F.avg_pool2d(images, 2), scale_factor=2, mode="bilinear", align_corners=False)
    images = torch.where(uniform(0, 1, 1, 1, 1) < 0.2, soft, images)
    return (images + noise).clamp(0, 255)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    model = BoardNet().to(device)
    test = load_all(args.test)
    shards = len(shard_starts(args.train))
    crops = sum(len(np.load(Path(args.train) / f"labels-{s}.npz")["kind"]) for s in shard_starts(args.train))
    steps_per_epoch = crops // args.batch
    total = steps_per_epoch * args.epochs
    warmup = min(300, total // 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.02)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (step + 1) / warmup if step < warmup else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup))),
    )
    print(f"device {device}, {parameter_count(model):,} parameters, {crops:,} training crops in {shards} shards, {len(test['images']):,} test crops", flush=True)

    history, step, began = [], 0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for images, squares, white in batches(args.train, args.batch, rng, group=2):
            batch = augment(torch.from_numpy(images).to(device).permute(0, 3, 1, 2).float(), generator)
            target = torch.from_numpy(squares).to(device).long().view(-1, 8, 8)
            logits, orientation = model(batch)
            loss = F.cross_entropy(logits, target) + 0.2 * F.binary_cross_entropy_with_logits(
                orientation, torch.from_numpy(white).to(device).float()
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            schedule.step()
            step += 1
            if step % 20 == 0:
                running += loss.item()
                seen += 1
            if step % 100 == 0:
                gpu = torch.mps.driver_allocated_memory() / 1e9 if device.type == "mps" else 0.0
                print(f"  step {step}/{total}  loss {running / max(1, seen):.4f}  gpu {gpu:.1f} GB  {(time.time() - began) / 60:.1f} min", flush=True)
        report = score(model, test, device)
        entry = {"epoch": epoch, "step": step, "minutes": round((time.time() - began) / 60, 1), "train_loss": round(running / max(1, seen), 4), **report}
        history.append(entry)
        print(
            f"epoch {epoch:2d}  loss {entry['train_loss']:.4f}  exact {report['all']['exact_boards']:.3f}  "
            f"squares {report['all']['square_accuracy']:.4f}  seen-sets exact {report['seen_piece_sets']['exact_boards']:.3f}  "
            f"held-out exact {report['held_out_piece_sets']['exact_boards']:.3f}  {entry['minutes']} min",
            flush=True,
        )
        torch.save(model.state_dict(), out / "model.pt")
        (out / "history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
