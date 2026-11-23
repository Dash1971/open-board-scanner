"""Export a trained board finder to ONNX, quantise it to INT8 and score both files.

    python -m train.export_finder --checkpoint runs/finder-v1/model.pt --scenes data/scenes/test --out runs/finder-v1

The ONNX files take `image` (1 x 3 x 256 x 256, 0..255) and give `corners`
(1 x 4 x 2, fractions of the view) and `present` (1, a logit). The batch size
is fixed at one: the finder looks at one view at a time, and a fixed shape
keeps the graph simple for quantising and for mobile runtimes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

from boardscan.pipeline import onnx_finder

from .finder import BoardFinder
from .finder_data import batch_of, load_scenes
from .train_finder import score_finder


class _Export(torch.nn.Module):
    """The finder without its heatmap output, which only training needs."""

    def __init__(self, model: BoardFinder) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor):
        corners, present, _ = self.model(image)
        return corners, present


class _Calibration(CalibrationDataReader):
    def __init__(self, images: np.ndarray) -> None:
        self.batches = iter({"image": image.transpose(2, 0, 1)[None].astype(np.float32)} for image in images)

    def get_next(self):
        return next(self.batches, None)


def _heatmap_convolutions(path: Path) -> list[str]:
    """Names of the convolutions that write the four corner heatmaps.

    They stay in floating point, as does everything after them. Corners are
    read as a mean position under each heatmap, and eight-bit heatmaps moved
    that mean by about 3% of the board's side.
    """
    import onnx

    graph = onnx.load(str(path)).graph
    four_wide = {item.name for item in graph.initializer if len(item.dims) == 4 and item.dims[0] == 4}
    return [node.name for node in graph.node if node.op_type == "Conv" and node.input[1] in four_wide]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scenes", required=True, help="synthetic test scenes: the first --calibration calibrate INT8, the rest are scored")
    parser.add_argument("--out", required=True)
    parser.add_argument("--calibration", type=int, default=300)
    args = parser.parse_args()
    out = Path(args.out)

    model = BoardFinder()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()
    float_path, prepared, int8_path = out / "boardfinder-fp32.onnx", out / "boardfinder-prepared.onnx", out / "boardfinder-int8.onnx"
    torch.onnx.export(
        _Export(model), torch.zeros(1, 3, 256, 256), str(float_path), input_names=["image"], output_names=["corners", "present"],
        opset_version=17, dynamo=False,
    )  # fmt: skip
    scenes = load_scenes(args.scenes)
    calibration, scored = scenes[: args.calibration], scenes[args.calibration :]
    views, _, _ = batch_of(calibration, 2 * len(calibration), np.random.default_rng(0))
    quant_pre_process(str(float_path), str(prepared))
    quantize_static(
        str(prepared), str(int8_path), _Calibration(views), quant_format=QuantFormat.QDQ, per_channel=True,
        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
        op_types_to_quantize=["Conv"], nodes_to_exclude=_heatmap_convolutions(prepared),
    )  # fmt: skip
    prepared.unlink()
    report = {name: {"file_mb": round(path.stat().st_size / 1e6, 3), **score_finder(onnx_finder(path), scored)} for name, path in (("fp32", float_path), ("int8", int8_path))}
    (out / "onnx.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
