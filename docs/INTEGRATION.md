# Putting the scanner in an app

Everything an app needs to turn an image into a position with the two model
files from the [models release](https://github.com/Dash1971/open-board-scanner/releases/tag/models-v1).
[`boardscan/pipeline.py`](../boardscan/pipeline.py) and
[`boardscan/geometry.py`](../boardscan/geometry.py) are the reference: about
150 lines, with no dependencies beyond NumPy, OpenCV and ONNX Runtime.

## The two models

Both take `image`: float32, shape `1 x 3 x 256 x 256`, RGB, channels first,
values **0 to 255**. Normalisation is inside the models; do not divide by 255.

| File | Outputs |
| --- | --- |
| `boardfinder-v1-int8.onnx` (1.4 MB) | `corners`: `1 x 4 x 2`, x and y of the board's top-left, top-right, bottom-right and bottom-left corners **as fractions of the 256 x 256 view**. `present`: `1`, a logit; above 0 means a whole board is in view. |
| `boardnet-v4-int8.onnx` (2.0 MB) | `squares`: `N x 13 x 8 x 8` logits, row 0 at the top of the crop. `rotation`: `N x 4` logits for how many anticlockwise quarter turns bring the crop upright. `white_at_bottom`: `N` logits; above 0 means White is at the bottom of the upright board. |

The finder takes one image at a time. The reader takes a batch.

Square classes, in order: empty, then `P N B R Q K p n b r q k`.

## The steps

1. **First look.** Take a square window centred on the image, 1.12 times its
   longer side, and resample it to 256 x 256 with black outside the image.
   Run the finder. Map the four corners back to image pixels.
2. **Second look.** Take a square window centred on those corners' bounding
   box, 1.35 times its larger side. Resample, run the finder again, map back.
   These are the final corners.
3. **Cut the board out.** Warp the four corners to a 256 x 256 crop
   (perspective transform, bilinear).
4. **Read it four ways.** Run the reader on the crop turned anticlockwise by
   0, 1, 2 and 3 quarter turns. Apply softmax to each `squares` output, turn
   each result back, and average the four.
5. **Find which way is up.** Apply softmax to each `rotation` output, shift
   the one from the crop turned `k` times by `k` places, average, and take
   the largest. Turn the averaged squares by that many anticlockwise quarter
   turns.
6. **Finish.** Each square's class is the largest value; that value is its
   confidence. Average the four `white_at_bottom` values (after a sigmoid).
   If Black is at the bottom, turn the grid half round before writing the
   position.

Pixel centres sit at whole numbers. A fraction `f` of a 256-pixel view is
pixel `f * 256 - 0.5`. `geometry.py` has the exact transforms; small
differences in resampling are tolerated, a half-pixel slip in the corner
mapping is not.

## What to show the user

- **Squares with confidence under 0.8** are worth marking for a second look.
- **`present` under 0** means no whole board was found; ask for another
  picture instead of showing a reading.
- **Which side is at the bottom is the least reliable output**, and it cannot
  be known for some positions. Give the user a flip button.
- **A page with two diagrams** gives one board. Let the user crop first.

## Checking a port

[`integration/`](integration) has three generated images and
[`expected.json`](integration/expected.json), the result for each from the
reference pipeline with the INT8 files. A port should give the same
`placement`, `quarter_turns` and `white_at_bottom`, and corners within a
pixel or two.

```sh
python -m boardscan --finder boardfinder-v1-int8.onnx --reader boardnet-v4-int8.onnx docs/integration/*.png
```

## Size and speed

- 3.5 MB for both files.
- Two finder passes and four reader passes per image.
- About 33 ms per photo on a Mac CPU with ONNX Runtime. Not measured on a
  phone.

## Licence

The models are under AGPL-3.0-only. What they were trained on is stated in
[MODEL-LICENSE.md](../MODEL-LICENSE.md).
