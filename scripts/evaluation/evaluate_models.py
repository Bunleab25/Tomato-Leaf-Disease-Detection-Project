import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
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
    parser = argparse.ArgumentParser(description="Evaluate saved tomato models on the test split.")
    parser.add_argument("--model", choices=["cnn", "mobilenet"], required=True)
    parser.add_argument("--weights", required=True, help="Path to .pth weights file")
    parser.add_argument("--dataset-root", default=str(DATASET_ROOT))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "cnn":
        transform = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )

    base_dataset = datasets.ImageFolder(str(args.dataset_root))
    class_names = base_dataset.classes

    _, _, test_idx = make_splits(len(base_dataset), args.seed)
    test_data = TransformSubset(base_dataset, test_idx.indices, transform)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    if args.model == "cnn":
        model = CNN(num_classes=len(class_names))
    else:
        model = build_mobilenet(num_classes=len(class_names))

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

    test_acc = sum(int(t == p) for t, p in zip(y_true, y_pred)) / len(y_true) if y_true else 0.0

    print(f"Using device: {device}")
    print(f"Model type: {args.model}")
    print(f"Weights: {args.weights}")
    print(f"Test samples: {len(test_data)}")
    print(f"Test Accuracy: {test_acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()