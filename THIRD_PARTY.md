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
| alpha | Eric Bentzen | "free for personal non commercial use" |

#### Freeware sets

Also from the same commit. Lichess lists these as "freeware", with no licence
text. They are marked `"freeware": true` in the manifest and are left out by
`--sets open`.

| Set | Author | Terms as listed by Lichess |
| --- | --- | --- |
| leipzig | Armando Hernandez Marroquin | "freeware" |
| companion | David L. Brown (Good Companion) | "freeware" |
| chess7 | Style-7 | "freeware" |

Good Companion is the typeface with tall, collared pawns and a boxed cross
on the king that earlier models could not read in two of the books
photographed for testing.

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

#### Sets from other chess programs

| Set | Author | Licence | Source |
| --- | --- | --- | --- |
| governor | the lila authors | AGPL-3.0-or-later | lichess-org/lila, same commit as above |
| kilfiger | James Kilfiger | Public domain, as stated in each file's metadata | [pychess/pychess](https://github.com/pychess/pychess) at `d434d37d35736601cd8c384c8947ad4e06921f8d`, `pieces/` |
| kosal | Kosal Sen | CC-BY-4.0 | PyChess, same commit |
| libra | sadsnake1 | CC-BY-NC-SA-4.0 (noncommercial) | PyChess, same commit |
| book-classic-b | Ben Citak | GPL-2.0 | [lukasmonk/lucaschessR2](https://github.com/lukasmonk/lucaschessR2) at `b3ab7678077d6433e45d85f55acac2102e13e82e`, `Resources/Pieces/` |
| book-diagram-b | Ben Citak | GPL-2.0 | Lucas Chess, same commit |
| book-strategie-b | Ben Citak | GPL-2.0 | Lucas Chess, same commit |
| kaneo | Kadagaden | CC-BY-4.0 | [luanlv/chess-pieces](https://github.com/luanlv/chess-pieces) at `4152933d2633c67195988631c0af325d4c3935bf` |
| kaneo-midnight | Kadagaden | CC-BY-4.0 | chess-pieces, same commit |

The three Book sets were drawn for Lucas Chess to look like printed
diagrams. Book Classic follows the same design as Good Companion, which
gives `--sets open` an openly licensed version of that typeface.

Sets in these programs with no stated terms, or that copy artwork already
listed here, were left out.

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
| alphadia | `AlphaDia.otf` from the skaknew package on CTAN | Eric Bentzen's Chess Alpha, adapted by Ulrich Dirr | Chess Alpha's terms: free for personal non-commercial use |

CTAN and En Passant are not git repositories, so these files are fetched by
URL and rely on their recorded hashes alone.

#### Fonts from the En Passant collection

The [En Passant chess font page](https://www.enpassant.dk/chess/fonteng.htm)
lists free chess diagram fonts with each author's terms; CTAN's
`fonts/chess/enpassant` holds Type 1 versions of some. None comes with a
licence text. Those marked "freeware" are described that way by their author
or by the page; those marked "non-commercial use" say so in their own notes.
All are left out by `--sets open`.

| Set | File and source | Author | Terms |
| --- | --- | --- | --- |
| chess-adventurer | `AVENFONT.TTF` from www.enpassant.dk/chess/downl/adventur.zip | Armando H. Marroquin | freeware |
| chess-cases | `CASEFONT.TTF` from www.enpassant.dk/chess/downl/cases.zip | Matthieu Leschemelle, redesigned by Armando H. Marroquin | freeware |
| chess-lucena | `LUCEFONT.TTF` from www.enpassant.dk/chess/downl/lucena.zip | Armando H. Marroquin | freeware |
| chess-usual | `TT_USUAL.TTF` from www.enpassant.dk/chess/downl/usual.zip | Armando H. Marroquin, after Chess Regular by Alastair Scott | freeware |
| chess-utrecht | `chess1.ttf` from www.enpassant.dk/chess/downl/chessttf.zip | Hans Bodlaender | freeware |
| smart-regular | `stchess.ttf` from www.enpassant.dk/chess/downl/smart.zip | Christoph Wirth | freeware |
| skak | `SKAK-K6.PFB` from www.enpassant.dk/chess/downl/skak.zip | Egon Madsen | non-commercial use |
| chess-berlin | `chess-berlin-board-fig-raw.pfb` from mirrors.ctan.org/fonts/chess/enpassant/chess-berlin-board-fig-raw.pfb | Eric Bentzen; Type 1 conversion by Ulrike Fischer | non-commercial use |
| chess-mark | `MARKFONT.TTF` from www.enpassant.dk/chess/downl/chesmark.zip | Armando H. Marroquin | freeware |
| chess-maya | `MAYAFONT.TTF` from www.enpassant.dk/chess/downl/chesmaya.zip | Armando H. Marroquin | freeware |
| chess-condal | `CONDFONT.TTF` from www.enpassant.dk/chess/downl/condal.zip | Armando H. Marroquin | freeware |
| chess-harlequin | `HARLFONT.TTF` from www.enpassant.dk/chess/downl/harlequi.zip | Armando H. Marroquin | freeware |
| chess-kingdom | `KINGFONT.TTF` from www.enpassant.dk/chess/downl/kingdom.zip | Armando H. Marroquin | freeware |
| chess-magnetic | `MAGNFONT.TTF` from www.enpassant.dk/chess/downl/magnetic.zip | Armando H. Marroquin | freeware |
| chess-marroquin | `MARRFONT.TTF` from www.enpassant.dk/chess/downl/marroqui.zip | Armando H. Marroquin | freeware |
| chess-mediaeval | `MVALFONT.TTF` from www.enpassant.dk/chess/downl/medie_tt.zip | Armando H. Marroquin | freeware |
| chess-motif | `MOTIFONT.TTF` from www.enpassant.dk/chess/downl/motif.zip | Armando H. Marroquin | freeware |
| chess-pirat | `PIRATDIA.TTF` from www.enpassant.dk/chess/downl/pirat.zip | Klaus Wolf | freeware |

Left out: Chess Regular, a conversion of Adobe's Cheq whose file carries
another company's copyright notice; Alfonso-X, Line and Millennia, which are
decorative and not used for diagrams; and fonts whose download no longer
works. Chess Usual is its author's redrawing of the Cheq design and is kept.

### Sets deliberately not used

Lichess ships more piece sets than these. The rest were left out because they
forbid modification (shahi-ivory-brown), have no stated terms (reillycraig,
riohacha), are stored as raster images this tool does not fetch (monarchy),
or cannot be labelled: mono draws both sides alike and disguised hides which
piece is which.

## Board textures and positions

`tools/fetch_boards.py` and `tools/fetch_positions.py` download these; the
manifests are [`assets/boards.json`](assets/boards.json) and
[`assets/positions.json`](assets/positions.json).

| Data | Source | Author | Terms |
| --- | --- | --- | --- |
| 24 board images | lichess-org/lila, `public/images/board` | the lila authors and pirouetti | AGPL-3.0-or-later |
| 13 pairs of square textures | pychess/pychess, `boards/` | not stated | none stated for the boards; the repository is GPL-3.0 |
| 15 board textures | Isarhamster/chessx, `data/themes/boards` | not stated | none stated for the boards; the repository is GPL-2.0-or-later |
| About 60,000 puzzle positions | database.lichess.org | Lichess | CC0 |

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
