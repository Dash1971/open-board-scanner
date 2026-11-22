# Open Board Scanner

A small, freely licensed model that turns a screenshot or a photo of a printed
chess diagram into an editable position, entirely offline.

**Status: planning. Nothing has been trained or measured yet.** This repository
holds the plan, the code and, later, the evidence. Work is tracked in
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

Every font, piece set and texture used to generate training data must have a
licence that allows this, and will be listed with its source and terms before
any weights are released. Private evaluation photos of copyrighted books are
not committed.
