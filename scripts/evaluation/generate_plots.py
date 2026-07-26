import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_ROOT = ROOT_DIR / "dataset_5class"
SEED = 42
BATCH_SIZE = 32


class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, dataset: datasets.ImageFolder, indices, transform):
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        image = self.transform(image)
        return image, label


class CNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 14 * 14, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


def build_mobilenet(num_classes: int) -> nn.Module:
    try:
        model = models.mobilenet_v2(weights=None)
    except TypeError:
        model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(1280, num_classes)
    return model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_splits(dataset_len: int, seed: int):
    train_size = int(0.7 * dataset_len)
    val_size = int(0.15 * dataset_len)
    test_size = dataset_len - train_size - val_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(range(dataset_len), [train_size, val_size, test_size], generator=generator)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate confusion matrix and training curves plots.")
    parser.add_argument("--model", choices=["cnn", "mobilenet"], default="mobilenet")
    parser.add_argument("--weights", default=str(ROOT_DIR / "mobile" / "mobilenet_model_best.pth"))
    parser.add_argument("--dataset-root", default=str(DATASET_ROOT))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--cm-output", default=str(ROOT_DIR / "reports" / "confusion_matrix_plot.png"))
    parser.add_argument("--curves-output", default=str(ROOT_DIR / "reports" / "training_curves.png"))
    parser.add_argument("--cnn-curve", default=str(ROOT_DIR / "cnn" / "cnn_loss_curve.png"))
    parser.add_argument("--mobilenet-curve", default=str(ROOT_DIR / "mobile" / "mobilenet_loss_curve.png"))
    parser.add_argument("--history", default="")
    return parser.parse_args()


def get_model_and_transform(model_name: str, num_classes: int):
    if model_name == "cnn":
        model = CNN(num_classes)
        transform = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ]
        )
    else:
        model = build_mobilenet(num_classes)
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )
    return model, transform


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    threshold = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                f"{cm[i, j]}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_training_curves(history_path: Path, output_path: str) -> None:
    if not history_path.exists():
        raise FileNotFoundError(f"Missing history file: {history_path}")

    history = json.loads(history_path.read_text(encoding="utf-8"))
    epochs = history["epochs"]
    train_acc = [v * 100.0 for v in history["train_acc"]]
    val_acc = [v * 100.0 for v in history["val_acc"]]
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].plot(epochs, train_acc, label="Train Accuracy")
    axes[0].plot(epochs, val_acc, label="Val Accuracy")
    axes[0].set_title("Accuracy Over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_loss, label="Train Loss")
    axes[1].plot(epochs, val_loss, label="Val Loss")
    axes[1].set_title("Loss Over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dataset = datasets.ImageFolder(str(args.dataset_root))
    class_names = base_dataset.classes

    _, _, test_idx = make_splits(len(base_dataset), args.seed)
    model, transform = get_model_and_transform(args.model, len(class_names))
    test_data = TransformSubset(base_dataset, test_idx.indices, transform)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, args.cm_output)
    if args.history:
        history_path = Path(args.history)
    elif args.model == "cnn":
        history_path = ROOT_DIR / "cnn" / "cnn_history.json"
    else:
        history_path = ROOT_DIR / "mobile" / "mobilenet_history.json"

    plot_training_curves(history_path, args.curves_output)

    print(f"Saved confusion matrix plot to {args.cm_output}")
    print(f"Saved training curves plot to {args.curves_output}")


if __name__ == "__main__":
    main()