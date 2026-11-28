# Reproducing the results

Every command and seed used for the models in the [report](REPORT.md).

## What can be reproduced

| | From this repository alone |
| --- | --- |
| Synthetic training and test data | Yes, identical files. Checked by regenerating the first shard of each set. |
| BoardNet v1 (synthetic only) | Yes, to within GPU randomness. |
| The board finder, trained on synthetic scenes only | Yes, to within GPU randomness. |
| The 1,650-board test | Yes. The boards are published as a [release](https://github.com/Dash1971/open-board-scanner/releases/tag/test-boards-v1). |
| BoardNet v2 and v4, and the shipped finder | No. They also train on 438 labelled photos of copyrighted pages, which are private. The commands are below for use with your own photos. |
| Results on real photos and screenshots | No. Those images are private. |
| The trained models | Published as a [release](https://github.com/Dash1971/open-board-scanner/releases/tag/models-v1), so the tests can be run without training. |

Training on a GPU does not give the same weights twice, so expect numbers
close to ours, not equal.

## Setup

Python 3.12. Check out the `models-v1` tag first: the generator has gained new
kinds of sample since, so the current code draws different images from the
same seeds.

```sh
git checkout models-v1
```

```sh
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt -r requirements-train.txt
python tools/fetch_assets.py
```

On an Apple GPU, set these before training:

```sh
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5
```

The first models used the eighteen piece sets the project started with:

```sh
FIRST=cburnett,celtic,chessnut,dejavu,fantasy,firi,gnome-fancy,gnome-simple,merida,mpchess,noto-symbols2,papercut,pirouetti,pixel,rhosgfx,skaknew,spatial,totoy
```

`--holdout` keeps fantasy and mpchess out of training data.

## Synthetic data

```sh
# Reader: 48,000 training crops and 3,000 test crops
python -m train.make_crops --seed 2 --count 24000 --holdout --sets $FIRST --out data/crops/train-v1
python -m train.make_crops --seed 1000 --count 3000 --crops 1 --sets $FIRST --out data/crops/test-v1

# Reader: 20,000 more crops in the seventeen sets added for v4
python -m train.make_crops --seed 11 --start 1000000 --count 10000 --crops 2 \
    --sets anarcandy,caliente,california,cardinal,chessboart,dubrovny,fresca,horsey,icpieces,kiwen-suwi,letter,maestro,minimal-warmth,shapes,staunty,tatiana,xkcd \
    --kinds screenshot,screenshot,monitor --out data/crops/train-newsets
mkdir -p data/crops/train-v2
ln -s "$PWD"/data/crops/train-v1/*.np* "$PWD"/data/crops/train-newsets/*.np* data/crops/train-v2/

# Finder: 16,000 training scenes and 800 test scenes
python -m train.make_scenes --seed 5 --count 16000 --holdout --sets $FIRST --out data/scenes/train
python -m train.make_scenes --seed 5000 --count 800 --sets $FIRST --out data/scenes/test
```

## Models

`A` and `B` stand for the two labelled photo sets: a label file made with
the [labelling tools](README.md#labelling-real-photos) and its photo folder.

```sh
# BoardNet v1: synthetic only
python -m train.train --train data/crops/train-v1 --test data/crops/test-v1 --out runs/boardnet-v1 --epochs 10 --batch 32

# BoardNet v2: v1 plus the first photo set
python -m train.finetune --labels A.json --photos A --start runs/boardnet-v1/model.pt \
    --synthetic data/crops/train-v1 --synthetic-test data/crops/test-v1 --steps 1000 --out runs/boardnet-v2
# its held-out tests: one run per source, 800 steps each
python -m train.finetune --labels A.json --photos A --start runs/boardnet-v1/model.pt \
    --synthetic data/crops/train-v1 --steps 800 --test-source "printed book 1" --out runs/held-out-book-1

# BoardNet v4: two rounds, with one fifth of all labelled boards held out
python -m train.finetune --labels A.json --photos A --labels B.json --photos B \
    --start runs/boardnet-v1/model.pt --synthetic data/crops/train-v2 --synthetic-test data/crops/test-v1 \
    --steps 6000 --real-share 0.3 --fold 0 5 --out runs/boardnet-v4-round1
python -m train.finetune --labels A.json --photos A --labels B.json --photos B \
    --start runs/boardnet-v4-round1/model.pt --synthetic data/crops/train-v2 --synthetic-test data/crops/test-v1 \
    --steps 24000 --real-share 0.3 --fold 0 5 --out runs/boardnet-v4
python -m train.export_onnx --checkpoint runs/boardnet-v4/model.pt --data data/crops/test-v1 --out runs/boardnet-v4

# Board finder v1: synthetic scenes plus the first photo set
python -m train.train_finder --scenes data/scenes/train --test data/scenes/test \
    --labels A.json --photos A --steps 6000 --out runs/finder-v1
python -m train.export_finder --checkpoint runs/finder-v1/model.pt --scenes data/scenes/test --out runs/finder-v1
```

Leave out `--labels` and `--photos` to train the finder on synthetic scenes
only. The finder's held-out test in the README used
`--test-source "printed book 3" --test-source "e-book 1" --steps 8000`.

All seeds are the default, 0.

BoardNet v0 came from an earlier generator. To rebuild it, check out commit
`275f871` and follow the commands in that version's README.

## Tests

```sh
# Photo in, position out, on labelled photos held out of training
python -m train.evaluate_scan --finder runs/finder-v1/boardfinder-int8.onnx --reader runs/boardnet-v4/boardnet-int8.onnx \
    --labels A.json --photos A --labels B.json --photos B --fold 0 5

# The 1,650 rendered boards: unzip the release file into data/external/ first
python -m train.evaluate_renders --data data/external/chess-piece-sets-33x50 \
    --finder runs/finder-v1/boardfinder-int8.onnx --reader runs/boardnet-v4/boardnet-int8.onnx

# The comparison with ChessQ, Fenshot and Fenify, then the report's charts
python tools/compare_models.py --others data/external/others --renders data/external/chess-piece-sets-33x50 \
    --labels A.json --photos A --labels B.json --photos B --fold 0 5 --out results/comparison
python tools/make_report_figures.py
```

The other projects' model files go in `data/external/others/`. Their sources
and checksums are in the
[earlier study](https://github.com/Dash1971/mobile-maia-research/blob/main/studies/offline-board-recognition-2026-11-13/data/source-pins.json)
and its `data/chessq/model-pins.json`; the file names expected are listed at
the top of `tools/compare_models.py`.
