"""Export a trained classifier to ONNX, quantise it to INT8 and score both files.

    python -m train.export_onnx --checkpoint runs/boardnet-v0/model.pt --data data/crops/test --out runs/boardnet-v0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

from .data import load_all
from .evaluate import _summary
from .model import BoardNet


class _Calibration(CalibrationDataReader):
    def __init__(self, images: np.ndarray, batch: int = 32) -> None:
        self.batches = iter(
            {"image": images[i : i + batch].transpose(0, 3, 1, 2).astype(np.float32)} for i in range(0, len(images), batch)
        )

    def get_next(self):
        return next(self.batches, None)


def run_onnx(path: Path, images: np.ndarray, batch: int = 64) -> np.ndarray:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    predicted = []
    for offset in range(0, len(images), batch):
        feed = images[offset : offset + batch].transpose(0, 3, 1, 2).astype(np.float32)
        predicted.append(session.run(["squares"], {"image": feed})[0].argmax(axis=1).reshape(len(feed), 64))
    return np.concatenate(predicted)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--calibration", type=int, default=400, help="crops used to calibrate INT8 ranges")
    args = parser.parse_args()
    out = Path(args.out)

    model = BoardNet()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()
    float_path, prepared, int8_path = out / "boardnet-fp32.onnx", out / "boardnet-prepared.onnx", out / "boardnet-int8.onnx"
    torch.onnx.export(
        model,
        torch.zeros(1, 3, 256, 256),
        str(float_path),
        input_names=["image"],
        output_names=["squares", "white_at_bottom", "rotation"],
        dynamic_axes={name: {0: "batch"} for name in ("image", "squares", "white_at_bottom", "rotation")},
        opset_version=17,
        dynamo=False,
    )

    data = load_all(args.data)
    # Calibrate on the first crops and score on the rest, so INT8 is not judged on its own calibration data.
    calibration, images, truth = data["images"][: args.calibration], data["images"][args.calibration :], data["squares"][args.calibration :]
    quant_pre_process(str(float_path), str(prepared))
    quantize_static(
        str(prepared),
        str(int8_path),
        _Calibration(calibration),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
    )
    prepared.unlink()

    report = {}
    for name, path in (("fp32", float_path), ("int8", int8_path)):
        report[name] = {"file_mb": round(path.stat().st_size / 1e6, 3), **_summary(run_onnx(path, images), truth)}
    (out / "onnx.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
