import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from tqdm import tqdm

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "USA currency"

@dataclass
class TrainConfig:
    data_dir: str
    out_dir: str
    epochs: int
    batch_size: int
    lr: float
    img_size: int
    val_split: float
    seed: int
    num_workers: int
    device: str


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dataloaders(cfg: TrainConfig) -> Tuple[DataLoader, DataLoader, List[str]]:
    data_path = Path(cfg.data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    # Good augmentations for real-world use (tilt, lighting, partial views)
    train_tfms = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)], p=0.7),
        transforms.RandomRotation(degrees=15),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Case A: user already has train/val folders
    train_dir = data_path / "train"
    val_dir = data_path / "val"
    if train_dir.exists() and val_dir.exists():
        train_ds = datasets.ImageFolder(str(train_dir), transform=train_tfms)
        val_ds = datasets.ImageFolder(str(val_dir), transform=val_tfms)
        class_names = train_ds.classes

    # Case B: single folder of class subfolders; split automatically
    else:
        full_ds_train = datasets.ImageFolder(str(data_path), transform=train_tfms)
        full_ds_val = datasets.ImageFolder(str(data_path), transform=val_tfms)
        class_names = full_ds_train.classes

        n = len(full_ds_train)
        indices = list(range(n))
        random.shuffle(indices)
        val_n = int(cfg.val_split * n)
        val_idx = indices[:val_n]
        train_idx = indices[val_n:]

        train_ds = Subset(full_ds_train, train_idx)
        val_ds = Subset(full_ds_val, val_idx)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )

    return train_loader, val_loader, class_names


def build_model(num_classes: int) -> nn.Module:
    # Lightweight + strong baseline for mobile deployment later
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)

    # Replace classifier head
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)

        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def train(cfg: TrainConfig) -> None:
    seed_everything(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    train_loader, val_loader, class_names = make_dataloaders(cfg)

    device = cfg.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(num_classes=len(class_names)).to(device)

    # Fine-tune entire network (good if dataset is moderate size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_path = Path(cfg.out_dir) / "best_model.pt"

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}", leave=False)
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
            pbar.set_postfix(loss=loss.item())

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        val_loss, val_acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)

    # Save label map for inference
    label_map = {i: name for i, name in enumerate(class_names)}
    with open(Path(cfg.out_dir) / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    print(f"\nSaved best model to: {best_path}")
    print(f"Saved label map to: {Path(cfg.out_dir) / 'label_map.json'}")
    print(f"Best val accuracy: {best_val_acc:.4f}")


def parse_args() -> TrainConfig:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR),
                    help="Path to dataset root. Defaults to the bundled 'USA currency' dataset in the repo.")
    ap.add_argument("--out_dir", type=str, default="out_usd_model")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--val_split", type=float, default=0.2,
                    help="Used only if data_dir does not have train/val folders.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}. Provide --data_dir to point to your dataset."
        )

    return TrainConfig(
        data_dir=str(data_dir),
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        img_size=args.img_size,
        val_split=args.val_split,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
