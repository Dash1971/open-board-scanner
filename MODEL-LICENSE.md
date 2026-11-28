# Model licence and provenance

**Trained models are licensed under [AGPL-3.0-only](LICENSE), the same as the
code.**

The model files are published as a [release](https://github.com/Dash1971/open-board-scanner/releases/tag/models-v2)
(the earlier files are in [models-v1](https://github.com/Dash1971/open-board-scanner/releases/tag/models-v1)). This page states the
terms they carry and, separately, what they were trained on.

## What the models were trained on

- **Rendered boards in openly licensed piece sets and fonts.**
- **Rendered boards in sixteen Lichess piece sets licensed CC BY-NC-SA**, for
  noncommercial use. From BoardNet v4 on. They are listed in
  [THIRD_PARTY.md](THIRD_PARTY.md).
- **Photographs of diagrams in copyrighted books and e-books.** From BoardNet
  v2 on. The photographs are not distributed.
- **From BoardNet v5 on** (the models-v2 release):
  - scanned pages of old books from Wikisource and the Internet Archive;
  - screenshots of the chess.com phone app and of Lichess;
  - the public "Chess Positions" image set, whose boards are drawn with
    artwork of unstated terms;
  - rendered boards in more piece sets and board textures, some with no
    stated terms.

  Each is listed with its source in [THIRD_PARTY.md](THIRD_PARTY.md).

The models contain none of this artwork or these photographs. They output
piece positions only.

## What we do and do not claim

- **We do not claim the training data restricts how the models may be used.**
  Whether a trained model is a derivative of its training images is not
  settled in law, and we treat ours as not being one.
- **We do not claim it is settled the other way either.** Anyone planning a
  commercial use should take their own advice.
- **We say plainly what went in**, so that nobody has to guess.

## Why the noncommercial sets are there

Without them the reader got 388 of 850 screen boards drawn in those sets
right. With them it gets 800. They include some of the most used pieces on
Lichess.

A model trained without them can be made from this repository:
`python -m train.make_crops --sets open` generates training data from the
openly licensed sets only.

## Status

Inclusion in Mobile Maia is pending a review of the licensing implications.
