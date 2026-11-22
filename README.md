# Open Board Scanner

A small, freely licensed model that turns a screenshot or a photo of a printed
chess diagram into an editable position, entirely offline.

**Status: a 2 MB classifier trained on synthetic data only. It is not yet a
usable scanner.** It reads about 86% of synthetic test boards exactly and
turns sideways photos upright reliably, but in hand checks on real photos it
read 14 of 24 boards exactly, with nearly all errors on one book typeface the
generator cannot imitate. No weights are released yet. Work is tracked in
[Issues](../../issues).

## Why

[Mobile Maia](https://github.com/Dash1971/maia-chess-android) wants a
scan-to-Board-Editor feature that works offline and fits free-software
distribution, including F-Droid. Our
[November 2026 study](https://github.com/Dash1971/mobile-maia-research/blob/main/studies/offline-board-recognition-2026-11-13/REPORT.md)
tested the available models on twelve photos and found no candidate that is
small, accurate and unrestricted at once:

| Model | File size | Exact boards / 12 | Licence |
| --- | ---: | ---: | --- |
| CrispChess | 0.84 MB | 1 (common crops) | MIT |
| Fenshot | 1.29 MB | 5 (common crops) | MIT |
| Fenify | 127.15 MB | 6 (common crops) | MIT project; weight terms unconfirmed |
| ChessQ Lite V4 INT8 | 41.37 MB | 10 (full photos) | PolyForm Noncommercial 1.0.0 |

The accurate model is large and noncommercial. The small models are free but
unreliable on printed diagrams. Classical contour detection, used to find the
board in a full photo, scored 1 to 2 of 12 in every route tested.

## Targets

| | Target |
| --- | --- |
| Total model size | 10 MB or less as INT8 ONNX; 1 MB is a stretch goal for screenshots |
| Inputs | Direct screenshots; phone photos of book diagrams; photos of a board on a monitor |
| Output | Piece placement for all 64 squares, plus which side is at the bottom |
| Runtime | ONNX Runtime on an Android CPU, no network |
| Out of scope | Photographs of physical boards and pieces |

"Accurate" will be defined against a held-out set of real photos that was not
used during development. The twelve study photos are too few, and have already
been examined, so they serve only as a first comparison with the study's table.

## Approach

1. **Synthetic training data.** Render random positions in many openly licensed
   chess fonts and piece sets, including the hatched dark squares common in
   books. Degrade the renders with perspective, page curl, uneven lighting,
   blur, halftone, paper texture, JPEG artefacts and screen moiré. Labels are
   exact and unlimited.
2. **Board localizer.** A small network that predicts the four board corners
   from a downscaled photo, replacing contour detection.
3. **Whole-board classifier.** A small network that reads the rectified board
   in one pass and outputs an 8×8 grid of 13 classes, so each square sees its
   neighbours and a slightly misaligned crop is tolerated.
4. **Orientation.** An extra output for which side is at the bottom.
5. **Real photos.** A held-out evaluation set, and fine-tuning data if the
   synthetic-only model falls short on books.

ChessQ and other restricted models are used for comparison only. Their outputs
are not used as training labels.

## Synthetic data generator

`boardgen` produces labelled images of three kinds: phone photos of printed
diagrams, direct screenshots, and phone photos of a monitor. Every sample is a
pure function of `(seed, index)`, so a dataset never has to be stored or shared.

![Twelve generated samples. Left of each pair: the image with the labelled board outline. Right: the board as the classifier will see it, cropped with deliberate error.](docs/samples/sheet-seed1.jpg)

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python tools/fetch_assets.py
.venv/bin/python -m boardgen sheet --seed 1 --count 12 --out docs/samples/sheet-seed1.jpg
.venv/bin/python -m boardgen dump --seed 1 --count 1000 --out data/gen/seed1
.venv/bin/python -m pytest tests
```

From Python, `boardgen.generate(seed, index)` returns the image and its labels,
and `boardgen.rectify(image, corners)` cuts out the board.

What varies:

- **Positions**: random legal play, sparse endgames, pieces scattered without
  regard to legality, the starting position and the empty board.
- **Piece artwork**: 18 sets, seven of which look like book typefaces. Printed
  pieces are also stretched and their strokes thickened or thinned, because
  typefaces differ in proportion and weight far more than the available
  artwork does.
- **Printed diagrams**: hatched, cross-hatched, dotted, grey and halftone dark
  squares; frames; coordinates; white halos around pieces; a page of Latin or
  Cyrillic text; a second diagram nearby; print showing through the paper.
- **Screens**: board colours, last-move and check highlights,
  legal-move dots, arrows, circles, coordinates, and a generic app layout with
  player bars, clocks, an evaluation bar and a move list.
- **Camera**: perspective, tilt, page curl, uneven light, shadows, glare,
  screen moiré, blur, sensor noise, low resolution and JPEG compression.
- **Sideways photos**: about a third of book photos are turned by a quarter
  turn, as happens when the phone is held sideways.

Each sample is labelled with:

| Field | Meaning |
| --- | --- |
| `corners` | The four outer corners of the 8×8 playing area in the image, clockwise from the top-left as displayed, in pixel-centre coordinates. Frames and coordinate margins are outside. |
| `squares` | 64 piece symbols as seen in the image, row by row from the top-left; `""` is an empty square. |
| `placement` | The FEN piece-placement field of the real position. |
| `quarter_turns` | How many quarter turns counter-clockwise bring the image upright. |
| `white_at_bottom` | `false` when the upright board is shown from Black's side. |
| `kind`, `meta` | `book`, `screenshot` or `monitor`, and the style choices made for the sample. |

Known gaps: the typeface behind most real-photo errors so far is commercial
and has no open look-alike here, and the degradations have not been tuned
against real photos. One process generates about two full samples a second on
an Apple M4.

Piece sets and fonts are downloaded on demand and listed with their licences
in [THIRD_PARTY.md](THIRD_PARTY.md).

## Classifier: BoardNet

A narrow ResNet (1.95 million parameters) takes a 256×256 crop of the board,
in whatever orientation the photo has, and outputs:

- 13 classes for each of the 64 squares, as they appear in the image;
- how many quarter turns bring the board upright;
- which side is at the bottom of the upright board.

```bash
.venv/bin/pip install -r requirements-train.txt
.venv/bin/python -m train.make_crops --seed 1000 --count 3000 --crops 1 --out data/crops/test-v1
.venv/bin/python -m train.make_crops --seed 2 --count 24000 --holdout --out data/crops/train-v1
.venv/bin/python -m train.train --train data/crops/train-v1 --test data/crops/test-v1 --out runs/boardnet-v1
.venv/bin/python -m train.export_onnx --checkpoint runs/boardnet-v1/model.pt --data data/crops/test-v1 --out runs/boardnet-v1
```

Two versions have been trained, each for 10 epochs on an Apple M4:

| | v0 | v1 |
| --- | --- | --- |
| Generator | 13 piece sets, upright only | 18 piece sets, shape variation, sideways photos |
| Training crops | 32,000 | 48,000 |
| Outputs | squares, viewing side | squares, viewing side, quarter turns |
| Training time | 30 minutes | 41 minutes |

Two piece sets, mpchess and fantasy, are kept out of training in both
versions to measure unseen artwork.

### Synthetic test sets

Each test set is 3,000 crops from a seed never used in training. The v1 test
set is harder than the v0 one: it includes the new piece sets, shape
variation and sideways crops.

| Exact boards | v0 on v0 test | v1 on v0 test | v1 on v1 test |
| --- | ---: | ---: | ---: |
| All | 88.9% | 88.1% | 85.5% |
| Piece sets seen in training | 93.7% | 92.6% | 88.3% |
| Piece sets held out | 73.5% | 73.7% | 65.8% |
| Printed diagrams | 88.1% | 87.0% | 82.7% |
| Screenshots | 91.5% | 91.9% | 93.3% |
| Photos of a monitor | 87.5% | 86.2% | 83.8% |

On the v1 test set, v1 gets 99.54% of squares right, picks the correct
quarter turn for 99.97% of non-empty boards, and names the viewing side
correctly for 88%. Putting all three together, 76.1% of crops give the exact
position. The viewing side is the weak part: a diagram without coordinates
often does not show which way it faces.

The broader generator did not raise accuracy on the old test set, and the gap
between seen and held-out piece sets is about as wide as before.

### ONNX files (v1)

| File | Size | Exact boards (2,600 test crops) |
| --- | ---: | ---: |
| FP32 | 7.79 MB | 85.4% |
| INT8 | 2.03 MB | 85.2% |

INT8 ranges were calibrated on 400 test crops that were then left out of the
scoring. On a Mac CPU with two threads, the v0 files of the same design took
5.1 ms (FP32) and 2.1 ms (INT8) per board. Android has not been measured.

### Real photos

Twenty-four phone photos were checked by hand, in two sets of twelve, each
with six printed book pages and six photos of an e-book on a monitor.

| | Exact boards | Wrong squares | Turned upright correctly |
| --- | ---: | ---: | ---: |
| First twelve, v0 (sideways crops turned by hand) | 6 of 12 | 19 of 768 | not applicable |
| First twelve, v1 | 7 of 12 | 25 of 768 | 12 of 12 |
| Second twelve, v1 | 7 of 12 | 30 of 768 | 12 of 12 |

What this shows:

- **Rotation works.** v1 turned all 24 crops the right way, including the ten
  sideways ones, with no help.
- **Piece reading on real photos did not improve.** v1 fixed some boards and
  made others worse.
- **One typeface causes nearly all the errors.** It appears in one of the
  printed books and in the e-books, and its pawns, bishops and kings are
  confused with each other and sometimes given the wrong colour. Boards in
  the other printed typeface, and sparse positions in any typeface, were read
  without error. The troublesome typeface appears to be a commercial one, so
  it cannot be added to the generator.

These are hand checks, not measurements:

- Photos were sampled by position in the folder. The board outline came from
  plain contour detection, which worked on all 24.
- The correct positions were read by eye from the crops and have not been
  independently verified.
- v1's generator changes were chosen after seeing the errors on the first
  twelve, so only the second twelve are a fair test of v1. v0 was not run on
  the second twelve.
- The photos are private and are not published.

The likely way forward is to label real photos and fine-tune on them, keeping
whole books out of training to measure the result honestly.

Recorded metrics are in [`results/`](results).

## Related work

**MoveAPiece.** [MoveAPiece](https://github.com/egdels/moveapiece) (GPL-3.0) is
also training its own freely licensed model, for the same licensing reason. Its
author [describes](https://github.com/egdels/moveapiece/issues/2) a deliberately
narrow problem: photographs of a DGT Pegasus board with its original pieces,
where the board already reports which squares are occupied, so the model only
classifies the pieces on those squares. No code or weights for it were public
when checked on 22 November 2026. That is a different problem from ours:
physical pieces with occupancy known, against flat diagrams with the board
still to be found.

**DocQuadNet-256.** The same author's document scanner,
[MakeACopy](https://github.com/egdels/makeacopy), ships a self-trained
corner-detection model under Apache-2.0, with the
[training pipeline documented](https://github.com/egdels/makeacopy/blob/main/training/README.md).
It is the closest precedent for our board localizer:

- MobileNetV3 backbone with a light feature pyramid, 256×256 input.
- Two outputs at 64×64: four corner heatmaps and a segmentation mask.
  Confidence combines the heatmap peak with corner-to-mask agreement.
- The predicted quad is snapped to image gradients at full resolution.
- A few fixed test-time rotations when the first pass is not confident.
- The shipped model file is 13.4 MB.
- On-device, ONNX Runtime's CPU provider with two threads was the fastest
  option they measured.

Their data notes also apply here: keep near-duplicate frames out of separate
splits, and replace backgrounds to stop the model learning the table.

## Licensing

Copyright (c) 2026 Dash. Code, documentation and released model weights in
this repository are licensed under the
[GNU Affero General Public License v3.0 only](LICENSE) (`AGPL-3.0-only`), the
same licence as Mobile Maia. Contributions are accepted under the same licence.

Every font and piece set used to generate training data must have a licence
that allows this. They are listed with their sources and terms in
[THIRD_PARTY.md](THIRD_PARTY.md), which also records the sets that were left
out because their terms are noncommercial or unclear. Private evaluation
photos of copyrighted books are kept in `eval/private/` and are not committed.
