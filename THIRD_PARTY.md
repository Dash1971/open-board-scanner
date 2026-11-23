# Third-party material

The generator renders training images from the artwork and fonts below. None
of these files are stored in this repository. `tools/fetch_assets.py` downloads
them from pinned commits and checks each file against the SHA-256 hash recorded
in [`assets/piece_sets.json`](assets/piece_sets.json),
[`assets/glyph_fonts.json`](assets/glyph_fonts.json) and
[`assets/fonts.json`](assets/fonts.json).

The sample sheet in `docs/samples/` contains renders of these piece sets and
fonts and is distributed under their respective licences.

## Piece sets

Source: [lichess-org/lila](https://github.com/lichess-org/lila) at commit
`1f25f6ae91088627103f1e3b6da76fa78e78ff8d`, directory `public/piece`. Authors
and licences are as stated in that commit's
[COPYING.md](https://github.com/lichess-org/lila/blob/1f25f6ae91088627103f1e3b6da76fa78e78ff8d/COPYING.md).

| Set | Author | Licence |
| --- | --- | --- |
| cburnett | Colin M.L. Burnett | GPL-2.0-or-later |
| merida | Armando Hernandez Marroquin | GPL-2.0-or-later |
| mpchess | Maxime Chupin | GPL-3.0-or-later |
| pirouetti | pirouetti | AGPL-3.0-or-later |
| pixel | therealqtpi | AGPL-3.0-or-later |
| chessnut | Alexis Luengas | Apache-2.0 |
| fantasy | Maurizio Monge | MIT |
| spatial | Maurizio Monge | MIT |
| celtic | Maurizio Monge | MIT |
| rhosgfx | RhosGFX | CC0-1.0 |
| firi | James Faure | CC-BY-4.0 |
| totoy | Kosal Sen | CC-BY-4.0 |
| papercut | Nikolay Anzarov | CC-BY-4.0 |
| kiwen-suwi | neverRare | CC-BY-4.0 |
| shapes | flugsio | CC-BY-SA-4.0 |
| letter | usolando | AGPL-3.0-or-later |

#### Noncommercial sets

The sets below, from the same commit, are licensed for noncommercial use
only. They are marked `"noncommercial": true` in the manifest.

| Set | Author | Licence |
| --- | --- | --- |
| anarcandy | caderek | CC-BY-NC-SA-4.0 |
| caliente | avi | CC-BY-NC-SA-4.0 |
| california | Jerry S. | CC-BY-NC-SA-4.0 |
| cardinal | sadsnake1 | CC-BY-NC-SA-4.0 |
| chessboart | Michał Fudalej / ChessboArt | CC-BY-NC-SA-4.0 |
| cooke | fejfar | CC-BY-NC-SA-4.0 |
| dubrovny | sadsnake1 | CC-BY-NC-SA-4.0 |
| fresca | sadsnake1 | CC-BY-NC-SA-4.0 |
| gioco | sadsnake1 | CC-BY-NC-SA-4.0 |
| horsey | cham, michael1241 | CC-BY-NC-SA-4.0 |
| icpieces | sadsnake1 | CC-BY-NC-SA-4.0 |
| maestro | sadsnake1 | CC-BY-NC-SA-4.0 |
| minimal-warmth | blunder_reign | CC-BY-NC-SA-4.0 |
| staunty | sadsnake1 | CC-BY-NC-SA-4.0 |
| tatiana | sadsnake1 | CC-BY-NC-SA-4.0 |
| xkcd | Randall Munroe | CC-BY-NC-SA-2.5 |

What follows from using them:

- **They are many people's everyday pieces.** staunty, maestro, horsey and
  others are popular on Lichess, and a model that has never seen them reads
  their screenshots badly, so the project's owner chose to train on them.
- **BoardNet from v4 on is trained with them.** Whether trained weights are a
  derivative of the artwork is unsettled. This project does not treat them
  as one, and states the provenance in
  [MODEL-LICENSE.md](MODEL-LICENSE.md) so that others can judge.
- **A model without them can still be trained.** `python -m train.make_crops --sets
  open` draws only from sets that allow commercial use, and
  `boardgen.assets.open_piece_sets()` lists them.
- **Sample images avoid them.** The sample sheet in `docs/samples/` was
  rendered before these sets were added.

Source: [GNOME/gnome-chess](https://github.com/GNOME/gnome-chess) at commit
`de63bdec5cf9f0770890086262e6a127d5c4a619`, directory `data/pieces`.

| Set | Author | Licence |
| --- | --- | --- |
| gnome-simple | W3C SVG working group, via GNOME Chess | W3C Software Notice and License, as stated in the set's `COPYING.html` |
| gnome-fancy | GNOME Chess contributors | GPL-3.0-or-later, the repository's licence; the set has no separate notice |

### Pieces drawn from fonts

These sets are drawn from the chess characters of a font. Enclosed areas of
each glyph are filled with white, as a printed diagram shows them.

| Set | Font and source | Author | Licence |
| --- | --- | --- | --- |
| skaknew | `SkakNew-Diagram.otf` from the [skaknew](https://ctan.org/pkg/skaknew) package on CTAN | Ulrich Dirr, after Piet Tutelaers, Torben Hoffmann and Dirk Bächle | LPPL-1.2 |
| dejavu | `DejaVuSans.ttf` as shipped in [matplotlib](https://github.com/matplotlib/matplotlib) at commit `c21a3923e2fa7735af4b9544c79d5fd81dea1dab` | DejaVu fonts team | Bitstream Vera licence, with DejaVu's changes in the public domain |
| noto-symbols2 | `NotoSansSymbols2-Regular.ttf` from google/fonts at the commit listed under Fonts | Google | OFL-1.1 |

CTAN is not a git repository, so the skaknew file is fetched by URL and
relies on its recorded hash alone.

### Sets deliberately not used

The skaknew package also contains `AlphaDia`, described in its README as a
version of Eric Bentzen's Chess Alpha. Chess Alpha is free for personal
non-commercial use only, so AlphaDia is not used.


Lichess ships more piece sets than these. The rest were left out because they
forbid derivatives (shahi-ivory-brown), are free for personal use only
(alpha), are not stated clearly enough to rely on (chess7, companion, leipzig,
reillycraig, riohacha), are stored as raster images this tool does not fetch
(monarchy), or cannot be labelled: mono draws both sides alike and disguised
hides which piece is which.

## Fonts

Source: [google/fonts](https://github.com/google/fonts) at commit
`06bfe0d62a98375cb09c942fa18e68bd32cff057`. All are under the SIL Open Font
License 1.1.

| Font | Path in google/fonts |
| --- | --- |
| PT Serif Regular, Bold | `ofl/ptserif` |
| Crimson Text Regular | `ofl/crimsontext` |
| EB Garamond | `ofl/ebgaramond` |
| Libre Baskerville | `ofl/librebaskerville` |
| Arimo | `ofl/arimo` |
| Source Sans 3 | `ofl/sourcesans3` |

## Python packages

| Package | Licence |
| --- | --- |
| python-chess (`chess`) | GPL-3.0-or-later |
| NumPy | BSD-3-Clause |
| OpenCV (`opencv-python-headless`) | Apache-2.0 |
| Pillow | MIT-CMU |
| resvg-py | MIT |
| pytest | MIT |
| PyTorch (`torch`), training only | BSD-3-Clause, with bundled components under Apache-2.0, MIT and similar licences |
| ONNX, training only | Apache-2.0 |
| ONNX Runtime, training only | MIT |
