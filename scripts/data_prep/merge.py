import shutil
import uuid
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT_DIR / "Tomato_Leaf_Dataset"
TARGET_DIR = ROOT_DIR / "dataset_5class"

MAPPING = {
    "Tomato_Early_blight": "Blight",
    "Tomato_leaf_late_blight": "Blight",
    "Tomato_septora_leaf_spot": "Septoria",
    "Tomato_mold_leaf": "Leaf_Mold",
    "Tomato_leaf_yellow_curl_virus": "Virus",
    "Tomato_Healthy": "Healthy",
}


def main() -> None:
    for src_class, target_class in MAPPING.items():
        src_path = SOURCE_DIR / src_class
        target_path = TARGET_DIR / target_class
        target_path.mkdir(parents=True, exist_ok=True)

        for source_file in src_path.iterdir():
            if not source_file.is_file():
                continue

            destination_file = target_path / f"{uuid.uuid4()}.jpg"
            shutil.copy(source_file, destination_file)

    print("Dataset merged successfully.")


if __name__ == "__main__":
    main()