from pathlib import Path

from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = ROOT_DIR / "yolov8n.pt"
DATA_CONFIG = ROOT_DIR / "leaf-desease-detection.v1i.yolov8" / "data.yaml"
RUNS_DIR = ROOT_DIR / "runs"


def main() -> None:
    model = YOLO(str(DEFAULT_WEIGHTS))
    model.train(
        data=str(DATA_CONFIG),
        epochs=50,
        imgsz=640,
        batch=16,
        project=str(RUNS_DIR),
        name="leaf_disease_detect",
    )


if __name__ == "__main__":
    main()