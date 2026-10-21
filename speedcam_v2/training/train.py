"""Train YOLOv8n on a custom one-class car dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


def build_data_yaml(dataset_dir: Path) -> Path:
    """Create data.yaml next to the dataset root."""
    payload = {
        "path": str(dataset_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "car"},
    }
    output = dataset_dir / "data.yaml"
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return output


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--model", default="yolov8n.pt")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset).resolve()
    data_yaml = build_data_yaml(dataset_dir)
    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        patience=20,
        project="runs/detect",
        name="speedcam_car",
        single_cls=True,
    )


if __name__ == "__main__":
    main()
