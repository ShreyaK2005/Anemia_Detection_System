"""
Anemia Detection from Eye Images using ResNet-18 with Transfer Learning
=======================================================================
Dataset : master_dataset_1  (train / val / test split)

Description
-----------
Binary classifier (anemia vs. normal) built on a pre-trained ResNet-18 backbone.
Key design choices:
  - Only layer3, layer4, and the classification head are fine-tuned; earlier
    layers remain frozen to preserve low-level ImageNet features.
  - Class-weighted cross-entropy loss compensates for any label imbalance.
  - StepLR scheduler gradually decays the learning rate for stable convergence.
  - Best checkpoint (by validation accuracy) is saved and re-loaded for testing.

Usage
-----
    python resnet_normalization.py

Requirements
------------
    torch torchvision scikit-learn
    Install via:  pip install torch torchvision scikit-learn
"""

import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def get_args() -> argparse.Namespace:
    """Parse command-line arguments so hyperparameters are easy to tune."""
    parser = argparse.ArgumentParser(
        description="Train ResNet-18 for anemia detection from eye images."
    )
    parser.add_argument(
        "--data_dir", type=str, default="master_dataset_1",
        help="Root dataset directory (must contain train/, val/, test/ sub-folders)."
    )
    parser.add_argument(
        "--epochs", type=int, default=25,
        help="Number of training epochs (default: 25)."
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for all data loaders (default: 16)."
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Initial learning rate for Adam (default: 1e-4)."
    )
    parser.add_argument(
        "--lr_step", type=int, default=5,
        help="StepLR epoch step size (default: 5)."
    )
    parser.add_argument(
        "--lr_gamma", type=float, default=0.5,
        help="StepLR decay factor (default: 0.5)."
    )
    parser.add_argument(
        "--checkpoint", type=str, default="best_model_resnet18.pth",
        help="Path to save the best model checkpoint (default: best_model_resnet18.pth)."
    )
    parser.add_argument(
        "--img_size", type=int, default=224,
        help="Input image size (default: 224)."
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

# ImageNet statistics used for normalizing pre-trained ResNet inputs
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_transforms(img_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    """
    Return (train_transform, eval_transform).

    Training pipeline applies random augmentations to improve generalisation:
      - Horizontal flip + small rotation add viewpoint robustness.
      - ColorJitter accounts for lighting variation across imaging devices.
    Validation / test pipeline uses only deterministic resize + normalise.
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])

    return train_transform, eval_transform


def build_loaders(
    data_dir: str,
    batch_size: int,
    img_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """
    Load ImageFolder datasets from data_dir/{train,val,test} and return
    (train_loader, val_loader, test_loader, class_names).
    """
    train_tf, eval_tf = build_transforms(img_size)

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, "val"),   transform=eval_tf)
    test_ds  = datasets.ImageFolder(os.path.join(data_dir, "test"),  transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader, train_ds.classes


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(num_classes: int = 2) -> nn.Module:
    """
    Build a fine-tuned ResNet-18 classifier.

    Strategy:
      - Load ImageNet pre-trained weights.
      - Freeze layers 1 & 2 (generic low-level feature extractors).
      - Unfreeze layers 3 & 4 (task-specific higher-level features).
      - Replace the final fully-connected head with a fresh linear layer.

    Args:
        num_classes: Number of output classes (default: 2 for anemia/normal).

    Returns:
        Configured nn.Module ready for training.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze deeper residual blocks for task-specific fine-tuning
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace classification head
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def build_class_weights(data_dir: str, class_names: list[str], device: torch.device) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from the training split.

    Weight for class i = (total samples) / (num_classes * count_i),
    which up-weights the minority class and stabilises training on
    imbalanced datasets.

    Args:
        data_dir    : Root dataset directory.
        class_names : Ordered list of class folder names (from ImageFolder).
        device      : Torch device to place the weight tensor on.

    Returns:
        Tensor of shape (num_classes,) with per-class loss weights.
    """
    train_path = os.path.join(data_dir, "train")
    counts = [
        len(os.listdir(os.path.join(train_path, cls)))
        for cls in class_names
    ]
    total = sum(counts)
    num_classes = len(class_names)

    weights = torch.tensor(
        [total / (num_classes * c) for c in counts],
        dtype=torch.float32,
    ).to(device)

    return weights


# ---------------------------------------------------------------------------
# Training & Evaluation helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one full training epoch and return the average loss."""
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate accuracy on a given DataLoader. Returns accuracy in [0, 1]."""
    model.eval()
    correct = total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, dim=1)
        total   += labels.size(0)
        correct += (predicted == labels).sum().item()

    return correct / total


@torch.no_grad()
def get_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list, list]:
    """Collect all predictions and ground-truth labels from a DataLoader."""
    model.eval()
    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, dim=1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

    return all_preds, all_labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = get_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader, test_loader, class_names = build_loaders(
        args.data_dir, args.batch_size, args.img_size
    )
    print(f"Classes: {class_names}")
    print(
        f"Dataset sizes — "
        f"train: {len(train_loader.dataset)}, "
        f"val: {len(val_loader.dataset)}, "
        f"test: {len(test_loader.dataset)}"
    )

    # Model, loss, optimiser, scheduler
    model     = build_model(num_classes=len(class_names)).to(device)
    weights   = build_class_weights(args.data_dir, class_names, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step, gamma=args.lr_gamma
    )

    # Training loop
    best_val_acc = 0.0
    print("\n--- Training ---")

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc  = evaluate(model, val_loader, device)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{args.epochs}  |  Loss: {avg_loss:.4f}  |  Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.checkpoint)
            print(f"  ✓ New best model saved → {args.checkpoint}")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")

    # Test evaluation
    print("\n--- Test Evaluation ---")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    all_preds, all_labels = get_predictions(model, test_loader, device)
    print(classification_report(all_labels, all_preds, target_names=class_names))


if __name__ == "__main__":
    main()
