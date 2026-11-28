"""Read {{Chess diagram}} templates out of MediaWiki wikitext.

The template (Module:Chessboard on Wikisource and Wikipedia) takes either 64
positional squares or a `fen=` parameter:

    {{Chess diagram | <alignment> | <header>
    | a8 | b8 | ... | h8
    ...
    | a1 | ... | h1
    | <footer> | reverse = true }}

Positional parameters 3 to 66 are the squares, a8 to h1. A square holds a
two-letter code, piece then colour: `kd` is a black king, `rl` a white rook;
blank, spaces or underscores mean empty. As in the module, the first two
letters or digits in the parameter are the code. Besides the six pieces there
are fairy pieces (archbishop `a`, upside-down pieces `h m s f e g`, and so
on), markers (`xx ox xo oo`), arrows (`ul ua ur la ra dl da dr lr ud`),
digits (`x0` to `x9`) and text labels (`?a`). With `fen=`, digits are runs
of empty squares and upper case is White. `reverse=true` draws the board
from Black's side, h1 at the top left.

`parse` returns each diagram on the page in order, with the board as the
template draws it: 8 rows of 8, top row first, `PNBRQK` for White and
`pnbrqk` for Black, `.` for an empty square.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

PIECES = "kqrbnp"
FAIRY = "aczwthmsfegj"  # j is the documented spelling of the upside-down bishop e
MARKERS = {"xx", "ox", "xo", "oo", "ul", "ua", "ur", "la", "ra", "dl", "da", "dr", "lr", "ud"} | {f"x{digit}" for digit in range(10)}
START = re.compile(r"\{\{\s*[Cc]hess[ _]diagram\s*(?=[|}])")


@dataclass
class Diagram:
    rows: list[str]  # 8 strings of 8 characters, top row as drawn first
    source: str  # "squares" or "fen"
    reverse: bool  # drawn from Black's side
    markers: list[str] = field(default_factory=list)  # arrows, crosses, digits and labels, by square as drawn
    fairy: list[str] = field(default_factory=list)  # fairy or unknown codes, by square as drawn
    problems: list[str] = field(default_factory=list)  # reasons the template does not give a plain 8 x 8 board
    header: str = ""
    footer: str = ""
    offset: int = 0  # where the template starts in the wikitext

    def placement(self) -> str:
        """The rows in FEN style, top row as drawn first."""
        return "/".join(re.sub(r"\.+", lambda run: str(len(run.group())), row) for row in self.rows)


def _template_end(text: str, start: int) -> int:
    """Index just past the `}}` that closes the template opened at `start`."""
    depth, index = 0, start
    while index < len(text):
        pair = text[index : index + 2]
        if pair in ("{{", "[["):
            depth += 1
            index += 2
        elif pair in ("}}", "]]"):
            depth -= 1
            index += 2
            if depth == 0:
                return index
        else:
            index += 1
    raise ValueError("unclosed template")


def _split(body: str) -> list[str]:
    """Split a template body on the pipes that are not inside nested templates or links."""
    parts, depth, current, index = [], 0, "", 0
    while index < len(body):
        pair = body[index : index + 2]
        if pair in ("{{", "[["):
            depth += 1
            current += pair
            index += 2
        elif pair in ("}}", "]]"):
            depth -= 1
            current += pair
            index += 2
        elif body[index] == "|" and depth == 0:
            parts.append(current)
            current, index = "", index + 1
        else:
            current += body[index]
            index += 1
    parts.append(current)
    return parts


def _top_level_equals(part: str) -> int:
    depth = 0
    for index, char in enumerate(part):
        if part[index : index + 2] in ("{{", "[["):
            depth += 1
        elif part[index : index + 2] in ("}}", "]]"):
            depth -= 1
        elif char == "=" and depth == 0:
            return index
    return -1


def arguments(body: str) -> tuple[dict[int, str], dict[str, str]]:
    """MediaWiki's reading of template arguments: positional ones keep their spaces, named ones are trimmed."""
    positional, named, counter = {}, {}, 0
    for part in _split(body)[1:]:  # the first part is the template name
        equals = _top_level_equals(part)
        if equals >= 0:
            key = part[:equals].strip()
            if key.isdigit():
                positional[int(key)] = part[equals + 1 :].strip()
            else:
                named[key] = part[equals + 1 :].strip()
        else:
            counter += 1  # unnamed arguments are numbered among themselves
            positional[counter] = part
    return positional, named


def _square(value: str) -> tuple[str, str]:
    """('piece', 'K') or ('marker', 'xx') or ('fairy', 'Nl') or ('empty', '') for one square parameter."""
    if re.match(r"^\?.", value):
        return "marker", value.strip()
    match = re.search(r"[A-Za-z0-9]{2}", value)
    if not match:
        return "empty", ""
    code = match.group()
    piece, colour = code[0], code[1]
    if piece in PIECES and colour in "ld":
        return "piece", piece.upper() if colour == "l" else piece
    if code in MARKERS:
        return "marker", code
    return "fairy", code


def _fen_squares(fen: str) -> list[str]:
    """Square codes as Module:Chessboard's convertFenToArgs makes them."""
    squares = []
    for row in re.findall(r"/([A-Za-z0-9]+)", "/" + fen):
        for char in row:
            if char.isdigit():
                squares += [""] * int(char)
            else:
                squares.append(char.lower() + ("l" if char.isupper() else "d"))
    return squares


def _diagram(body: str, offset: int) -> Diagram:
    positional, named = arguments(body)
    reverse = named.get("reverse", "").lower() == "true"
    problems = []
    if "fen" in named:
        squares = _fen_squares(named["fen"])
        if len(squares) != 64:
            problems.append(f"fen gives {len(squares)} squares")
        source, header, footer = "fen", named.get("header", ""), named.get("footer", "")
    else:
        squares = [positional.get(number, "") for number in range(3, 67)]
        source, header, footer = "squares", positional.get(2, ""), positional.get(67, "")
        extra = [number for number in positional if number > 67 and positional[number].strip()]
        if extra:
            problems.append(f"{len(extra)} positional parameters after the footer")
        if re.search(r"[A-Za-z0-9]{2}", footer) and _square(footer)[0] == "piece" and len(footer.strip()) <= 3:
            problems.append("footer looks like a square: the squares may be shifted")
    squares = (squares + [""] * 64)[:64]
    grid, markers, fairy = [["."] * 8 for _ in range(8)], [], []
    for index, value in enumerate(squares):
        row, column = divmod(index, 8)  # row 0 is rank 8
        if reverse:
            row, column = 7 - row, 7 - column
        name = "abcdefgh"[column] + str(8 - row)  # the square's place as drawn, named as if White were at the bottom
        kind, code = _square(value)
        if kind == "piece":
            grid[row][column] = code
        elif kind == "marker":
            markers.append(f"{name}:{code}")
        elif kind == "fairy":
            fairy.append(f"{name}:{code}")
    if fairy:
        problems.append("fairy or unknown codes: " + " ".join(fairy))
    return Diagram(["".join(row) for row in grid], source, reverse, markers, fairy, problems, header.strip(), footer.strip(), offset)


def parse(wikitext: str) -> list[Diagram]:
    """Every {{Chess diagram}} in the wikitext, in the order they appear."""
    diagrams = []
    for match in START.finditer(wikitext):
        end = _template_end(wikitext, match.start())
        diagrams.append(_diagram(wikitext[match.start() + 2 : end - 2], match.start()))
    return diagrams
