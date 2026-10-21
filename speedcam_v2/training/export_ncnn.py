"""Export a trained YOLO model to NCNN format."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--imgsz", type=int, default=320)
    args = parser.parse_args()

    weights = Path(args.weights).resolve()
    model = YOLO(str(weights))
    model.export(format="ncnn", int8=True, imgsz=args.imgsz)


if __name__ == "__main__":
    main()
