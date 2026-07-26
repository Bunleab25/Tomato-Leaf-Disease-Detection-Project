import random
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

# CONFIG
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
SEED = 42
ROOT_DIR = Path(__file__).resolve().parents[1]
RUN_DIR = Path(__file__).resolve().parent
DATASET_ROOT = ROOT_DIR / "dataset_5class"
MODEL_PATH = RUN_DIR / "mobilenet_model.pth"
BEST_MODEL_PATH = RUN_DIR / "mobilenet_model_best.pth"
EARLY_STOPPING_PATIENCE = 3
LOSS_PLOT_PATH = RUN_DIR / "mobilenet_loss_curve.png"
HISTORY_PATH = RUN_DIR / "mobilenet_history.json"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_splits(dataset_len: int, seed: int):
    train_size = int(0.7 * dataset_len)
    val_size = int(0.15 * dataset_len)
    test_size = dataset_len - train_size - val_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(range(dataset_len), [train_size, val_size, test_size], generator=generator)


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


def build_mobilenet(num_classes: int) -> nn.Module:
    try:
        weights = models.MobileNet_V2_Weights.DEFAULT
        model = models.mobilenet_v2(weights=weights)
    except AttributeError:
        model = models.mobilenet_v2(pretrained=True)

    model.classifier[1] = nn.Linear(1280, num_classes)
    for param in model.features.parameters():
        param.requires_grad = False

    for block in model.features[-3:]:
        for param in block.parameters():
            param.requires_grad = True

    return model


def evaluate_metrics(
    model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            total_loss += loss.item() * labels.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def save_loss_plot(train_losses: list[float], val_losses: list[float], output_path: Path) -> None:
    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker="o", label="Train Loss")
    plt.plot(epochs, val_losses, marker="o", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MobileNetV2 Training and Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )

    base_dataset = datasets.ImageFolder(str(DATASET_ROOT))
    class_names = base_dataset.classes

    train_idx, val_idx, test_idx = make_splits(len(base_dataset), SEED)
    train_data = TransformSubset(base_dataset, train_idx.indices, train_transform)
    val_data = TransformSubset(base_dataset, val_idx.indices, eval_transform)
    test_data = TransformSubset(base_dataset, test_idx.indices, eval_transform)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    print(
        f"Dataset split sizes -> train: {len(train_data)}, val: {len(val_data)}, test: {len(test_data)}"
    )
    print(f"Classes ({len(class_names)}): {class_names}")

    model = build_mobilenet(len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    epochs_run = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        total_train = 0
        correct_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
            total_loss += loss.item() * labels.size(0)

        avg_train_loss = total_loss / total_train if total_train > 0 else 0.0
        train_acc = correct_train / total_train if total_train > 0 else 0.0
        val_loss, val_acc = evaluate_metrics(model, val_loader, device, criterion)
        epochs_run.append(epoch + 1)
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        print(
            f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {avg_train_loss:.4f}, "
            f"Train Acc: {train_acc:.3f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), str(BEST_MODEL_PATH))
            print(f"Saved best checkpoint to {BEST_MODEL_PATH}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(
                    f"Early stopping triggered after {epoch + 1} epochs "
                    f"(patience={EARLY_STOPPING_PATIENCE})."
                )
                break

    torch.save(model.state_dict(), str(MODEL_PATH))
    print(f"Saved model to {MODEL_PATH}")

    history = {
        "epochs": epochs_run,
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_acc": train_accuracies,
        "val_acc": val_accuracies,
    }
    HISTORY_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved history to {HISTORY_PATH}")

    save_loss_plot(train_losses, val_losses, LOSS_PLOT_PATH)
    print(f"Saved loss plot to {LOSS_PLOT_PATH}")

    best_checkpoint = Path(BEST_MODEL_PATH)
    if best_checkpoint.exists():
        model.load_state_dict(torch.load(str(BEST_MODEL_PATH), map_location=device))
        print(f"Loaded best checkpoint from {BEST_MODEL_PATH} for test evaluation")

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
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print("\nMobileNet Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()
