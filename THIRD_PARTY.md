# Third-party material

The generator renders training images from the artwork and fonts below. None
of these files are stored in this repository. `tools/fetch_assets.py` downloads
them from pinned commits and checks each file against the SHA-256 hash recorded
in [`assets/piece_sets.json`](assets/piece_sets.json) and
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

### Sets deliberately not used

Lichess ships more piece sets than these. The rest were left out because their
terms are noncommercial (the CC BY-NC-SA sets, and alpha), forbid derivatives
(shahi-ivory-brown), or are not stated clearly enough to rely on (chess7,
companion, leipzig, reillycraig, riohacha). Four permitted sets were left out
because they do not look like chess pieces or have no separate black and white
artwork: letter, shapes, kiwen-suwi and mono.

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
