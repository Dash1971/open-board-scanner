# Open Board Scanner

A small, freely licensed model that turns a screenshot or a photo of a printed
chess diagram into an editable position, entirely offline.

**Status: a first 2 MB classifier has been trained on synthetic data only. It
is a baseline, not a usable scanner:** it reads 89% of synthetic test boards
exactly, and six of twelve real photos in a first hand check. No weights are
released yet. Work is tracked in [Issues](../../issues).

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
- **Printed diagrams**: hatched, cross-hatched, dotted, grey and halftone dark
  squares; frames; coordinates; white halos around pieces; a page of Latin or
  Cyrillic text; a second diagram nearby; print showing through the paper.
- **Screens**: board colours, 13 piece sets, last-move and check highlights,
  legal-move dots, arrows, circles, coordinates, and a generic app layout with
  player bars, clocks, an evaluation bar and a move list.
- **Camera**: perspective, rotation, page curl, uneven light, shadows, glare,
  screen moiré, blur, sensor noise, low resolution and JPEG compression.

Each sample is labelled with:

| Field | Meaning |
| --- | --- |
| `corners` | The four outer corners of the 8×8 playing area in the image, clockwise from the top-left as displayed, in pixel-centre coordinates. Frames and coordinate margins are outside. |
| `squares` | 64 piece symbols as seen in the image, row by row from the top-left; `""` is an empty square. |
| `placement` | The FEN piece-placement field of the real position. |
| `white_at_bottom` | `false` when the board is shown from Black's side, so `squares` is the position turned by 180°. |
| `kind`, `meta` | `book`, `screenshot` or `monitor`, and the style choices made for the sample. |

Known gaps: only three of the piece sets resemble book typefaces, no sample
is rotated by a quarter turn, and the degradations have not been tuned
against real photos. One process generates about two full samples a second on
an Apple M4.

Piece sets and fonts are downloaded on demand and listed with their licences
in [THIRD_PARTY.md](THIRD_PARTY.md).

## First classifier: BoardNet v0

A narrow ResNet (1.95 million parameters) takes a 256×256 crop of the board
and outputs 13 classes for each of the 64 squares, plus which side is at the
bottom. It was trained for 10 epochs on 32,000 synthetic crops (16,000
generated samples, two crops each), in 30 minutes on an Apple M4.

```bash
.venv/bin/pip install -r requirements-train.txt
.venv/bin/python -m train.make_crops --seed 1000 --count 3000 --crops 1 --out data/crops/test
.venv/bin/python -m train.make_crops --seed 1 --count 16000 --holdout --out data/crops/train
.venv/bin/python -m train.train --train data/crops/train --test data/crops/test --out runs/boardnet-v0
.venv/bin/python -m train.export_onnx --checkpoint runs/boardnet-v0/model.pt --data data/crops/test --out runs/boardnet-v0
```

### Synthetic test set

3,000 crops from a seed never used in training. Two piece sets, mpchess and
fantasy, were kept out of training entirely to measure unseen artwork.

| Test crops | Boards | Exact boards | At most one wrong square | Squares correct |
| --- | ---: | ---: | ---: | ---: |
| All | 3,000 | 88.9% | 96.0% | 99.69% |
| Piece sets seen in training | 2,286 | 93.7% | 97.6% | 99.79% |
| Piece sets held out | 714 | 73.5% | 91.2% | 99.38% |
| Printed diagrams | 1,773 | 88.1% | 94.9% | 99.64% |
| Screenshots | 757 | 91.6% | 98.4% | 99.83% |
| Photos of a monitor | 470 | 87.5% | 96.4% | 99.67% |

The gap between seen and held-out piece sets is the main result: accuracy
depends on having seen the artwork, so the generator needs many more piece
styles. The commonest mistakes were a black rook read as a black pawn, a white
pawn read as empty, and a white queen read as a white rook. Which side is at
the bottom was right for 87% of non-empty boards.

### ONNX files

| File | Size | Exact boards (2,600 test crops) | One board, Mac CPU, 2 threads |
| --- | ---: | ---: | ---: |
| FP32 | 7.78 MB | 88.9% | 5.1 ms |
| INT8 | 2.03 MB | 88.8% | 2.1 ms |

INT8 ranges were calibrated on 400 test crops that were then left out of the
scoring. Android speed and memory have not been measured.

### First look at real photos

Twelve phone photos were checked by hand: six of printed book pages and six of
an e-book on a monitor. The model read six of the twelve boards exactly and
got 19 squares wrong across the other six. The errors were mostly one book
typeface's pawns read as bishops, and kings confused with queens.

This is a first look, not a measurement:

- The twelve were sampled by position in the folder, the board outline came
  from plain contour detection, which happened to work on all twelve, and five
  book photos taken sideways were turned upright by hand.
- The correct positions were read by eye from the crops and have not been
  independently verified.
- The photos are private and are not published.

Two things this showed: photos taken with the phone turned sideways are
common, so rotated boards must be generated and recognised; and typefaces the
generator lacks are where the errors are.

Recorded metrics are in [`results/boardnet-v0`](results/boardnet-v0).

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
