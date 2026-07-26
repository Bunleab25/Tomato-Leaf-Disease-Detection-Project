import random
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
BLIGHT_PATH = ROOT_DIR / "dataset_5class" / "Blight"
TARGET_COUNT = 1200


def main() -> None:
    files = [path for path in BLIGHT_PATH.iterdir() if path.is_file()]
    print(f"Total files found: {len(files)}")

    if len(files) <= TARGET_COUNT:
        print(f"Skipping: Only {len(files)} images found (target {TARGET_COUNT}).")
        return

    selected = set(random.sample(files, TARGET_COUNT))
    removed_count = 0
    for file_path in files:
        if file_path not in selected:
            file_path.unlink()
            removed_count += 1

    print(f"Removed {removed_count} extra images. Blight balanced to {TARGET_COUNT} images.")


if __name__ == "__main__":
    main()