import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
from torchvision import datasets, models, transforms

IMG_SIZE = 224
ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT_DIR / "dataset_5class"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "mobilenet_model_best.pth"


def build_model(num_classes: int) -> nn.Module:
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(1280, num_classes)
    return model


def load_class_names(dataset_root: Path) -> list[str]:
    dataset = datasets.ImageFolder(str(dataset_root))
    return dataset.classes


def preprocess_image(image_path: Path) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def preprocess_pil_image(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    return transform(image.convert("RGB")).unsqueeze(0)


def load_model_and_classes(
    model_path: Path,
    dataset_root: Path,
    device: torch.device | None = None,
) -> tuple[nn.Module, list[str], torch.device]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_root}")

    class_names = load_class_names(dataset_root)
    num_classes = len(class_names)

    resolved_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes).to(resolved_device)

    state_dict = torch.load(str(model_path), map_location=resolved_device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, class_names, resolved_device


def predict_tensor(
    image_tensor: torch.Tensor,
    model: nn.Module,
    class_names: list[str],
    device: torch.device,
    top_k: int,
) -> dict[str, Any]:
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)

    num_classes = len(class_names)
    top_k = max(1, min(top_k, num_classes))
    top_probs, top_indices = torch.topk(probs, k=top_k, dim=1)

    top_predictions: list[dict[str, Any]] = []
    for prob, idx in zip(top_probs[0].cpu().tolist(), top_indices[0].cpu().tolist()):
        top_predictions.append(
            {
                "label": class_names[idx],
                "confidence": float(prob),
            }
        )

    all_probs = probs[0].cpu().tolist()
    probabilities = {
        class_name: float(probability)
        for class_name, probability in zip(class_names, all_probs)
    }

    return {
        "prediction": top_predictions[0]["label"],
        "confidence": top_predictions[0]["confidence"],
        "top_predictions": top_predictions,
        "probabilities": probabilities,
    }


def predict(
    image_path: Path,
    model_path: Path,
    dataset_root: Path,
    top_k: int,
) -> dict[str, Any]:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model, class_names, device = load_model_and_classes(model_path, dataset_root)
    image_tensor = preprocess_image(image_path)
    result = predict_tensor(image_tensor, model, class_names, device, top_k)

    result["device"] = str(device)
    result["image"] = str(image_path)
    result["model"] = str(model_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict tomato leaf class with MobileNetV2")
    parser.add_argument("--image", required=True, type=Path, help="Path to image file")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to model checkpoint (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
        help=f"Path to dataset root used for class names (default: {DATASET_ROOT})",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of top predictions to print")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = predict(args.image, args.model, args.dataset_root, args.top_k)

    print(f"Using device: {output['device']}")
    print(f"Image: {output['image']}")
    print(f"Model: {output['model']}")
    print("\nTop predictions:")

    for rank, pred in enumerate(output["top_predictions"], start=1):
        print(f"{rank}. {pred['label']} -> {pred['confidence'] * 100:.2f}%")


if __name__ == "__main__":
    main()
