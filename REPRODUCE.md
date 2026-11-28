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
| BoardNet v5 | No. It also trains on 548 private photos and 28 private screenshots. Its other 1,156 real boards (Wikisource, Internet Archive, Lichess) and all its synthetic data can be fetched or regenerated. |
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

### BoardNet v5

Not yet in a release. The synthetic crops come from the current generator
with every set in the manifests (`python tools/fetch_assets.py` first):

```sh
python -m train.make_crops --seed 21 --count 60000 --crops 2 --holdout --out data/crops/train-v3
python -m train.make_crops --seed 2100 --count 3000 --crops 1 --out data/crops/test-v3
python -m train.make_crops --seed 31 --start 2000000 --count 3000 --crops 2 \
    --sets fantasy,mpchess,cooke,gioco,monarchy --out data/crops/train-heldsets
python -m train.make_crops --seed 41 --start 3000000 --count 4000 --crops 2 \
    --kinds screenshot,screenshot,monitor --out data/crops/train-screens
python -m train.make_crops --seed 42 --start 4000000 --count 12000 --crops 2 \
    --kinds screenshot,screenshot,monitor --out data/crops/train-screens

# The public Chess Positions boards: 79,939 to train on, 20,061 kept as a test
python tools/fetch_chess_positions.py
python -m train.import_chess_positions --source data/external/chess-positions-full --out data/crops/kaggle
```

`data/crops/train-v5` is a folder of links to every shard of `train-v3`,
`train-newsets`, `train-heldsets`, `train-screens` and
`kaggle/chess-positions-train`. Shard numbers must not clash, so the Chess
Positions shards are linked under numbers from 9000000.

Real boards. Four private photo sets and the private chess.com screenshots
cannot be fetched; the others can:

```sh
python tools/fetch_wikisource.py                                          # Wikisource pages
python tools/wikisource_manifest.py read && python tools/wikisource_manifest.py build   # diagrams matched to positions; doubtful ones are settled by eye in between
python tools/external_labels.py                                           # label files for training
python tools/fetch_ia_diagrams.py fetch                                   # Internet Archive pages
python tools/ia_labels.py restore                                         # their labels, from assets/archive_diagrams.json
```

```sh
# Check run: whole sources held out, and the fifth of the first two photo sets that v4 held out.
# Its synthetic folder was train-v3 plus the first 9,000 Chess Positions crops.
python -m train.finetune --labels A.json --photos A ... --labels data/external/lichess-shots/labels.json --photos data/external/lichess-shots/images \
    --fold 0 5 --fold-sets 2 \
    --test-source "app screenshot 1" --test-source "e-book 9" ... --test-source "e-book 15" \
    --test-source "Staunton, The Chess-Player's Text Book (1849)" --test-source "Capablanca, My Chess Career (1920)" \
    --test-source "chess.com app" --test-source "lichess site" \
    --start runs/boardnet-v4/model.pt --synthetic data/crops/train-v3c --synthetic-test data/crops/test-v3 \
    --steps 30000 --real-share 0.3 --out runs/boardnet-v5-check

# Final run: every real board, the Internet Archive ones included, nothing held out
python -m train.finetune --labels A.json --photos A ... --labels data/external/ia/labels.json --photos data/external/ia \
    --start runs/boardnet-v5-check/model.pt --synthetic data/crops/train-v5 \
    --steps 30000 --real-share 0.3 --out runs/boardnet-v5
python -m train.export_onnx --checkpoint runs/boardnet-v5/model.pt --data data/crops/test-v3 --out runs/boardnet-v5

# Every test in one file
python -m train.scoreboard --name boardnet-v5 --finder runs/finder-v1/boardfinder-int8.onnx --reader runs/boardnet-v5/boardnet-int8.onnx
```

The final run needs about 2.5 GB of memory and 85 minutes on a MacBook Air.
On a 16 GB machine set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.3` and
`PYTORCH_MPS_LOW_WATERMARK_RATIO=0.2` first. A checkpoint is saved at every
evaluation.

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
