# arcface_train_cpu.py
"""
CPU-only ArcFace training script (CLEAN pipeline):
- No augmentation or balancing on disk
- Train-only augmentation in transforms
- Train-only class balancing via WeightedRandomSampler (optional flag)
- Mean/std loaded from stats.json computed from TRAIN split on disk (out_dataset/stats.json)

Expected layout (torchvision ImageFolder):
  out_dataset/
    train/class0/*.png
    val/class0/*.png
    test/class0/*.png   (optional; this script can evaluate it)

Run:
  python arcface_train_cpu.py \
    --train_dir out_dataset/train \
    --val_dir out_dataset/val \
    --stats_path out_dataset/stats.json \
    --epochs 30 \
    --balance_train

Optional:
  --test_dir out_dataset/test   (evaluate once at the end)
"""

import math
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from torchvision import transforms
from torchvision.datasets import ImageFolder


# ----------------------------
# Model
# ----------------------------

class ArcMarginProduct(nn.Module):
    """
    ArcFace head:
      - forward(emb_norm, labels) -> margin logits (TRAIN)
      - forward_inference(emb_norm) -> cosine logits (INFER)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float = 30.0,
        m: float = 0.50,
        easy_margin: bool = False,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.s = float(s)
        self.m = float(m)
        self.easy_margin = bool(easy_margin)

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def _cosine(self, emb_norm: torch.Tensor) -> torch.Tensor:
        return F.linear(emb_norm, F.normalize(self.weight, p=2, dim=1))

    def forward(self, emb_norm: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = self._cosine(emb_norm)
        sine = torch.sqrt(torch.clamp(1.0 - cosine * cosine, min=0.0))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        logits = one_hot * phi + (1.0 - one_hot) * cosine
        logits = logits * self.s
        return logits

    @torch.no_grad()
    def forward_inference(self, emb_norm: torch.Tensor) -> torch.Tensor:
        return self._cosine(emb_norm) * self.s


class MatlabNetArcFace(nn.Module):
    def __init__(self, num_classes: int, emb_dim: int = 256, s: float = 30.0, m: float = 0.50):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=True)
        self.bn1 = nn.BatchNorm2d(32, eps=1e-5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1, bias=True)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1, bias=True)
        self.bn3 = nn.BatchNorm2d(128, eps=1e-5)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1, bias=True)
        self.bn4 = nn.BatchNorm2d(256, eps=1e-5)

        # For 61x61: 61 -> 31 -> 16 -> 8
        self.fc1 = nn.Linear(8 * 8 * 256, 512)
        self.fc2 = nn.Linear(512, emb_dim)

        self.arc_head = ArcMarginProduct(
            in_features=emb_dim, out_features=num_classes, s=s, m=m)

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None, return_embedding: bool = False):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn4(x)

        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        emb = F.relu(x)

        emb_norm = F.normalize(emb, p=2, dim=1)

        if labels is not None:
            logits = self.arc_head(emb_norm, labels)
        else:
            logits = self.arc_head.forward_inference(emb_norm)

        if return_embedding:
            return logits, emb
        return logits

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        self.eval()
        logits, emb = self.forward(x, labels=None, return_embedding=True)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        emb_norm = F.normalize(emb, p=2, dim=1)
        return probs, preds, emb, emb_norm


# ----------------------------
# Training utilities
# ----------------------------

@dataclass
class TrainConfig:
    train_dir: str
    val_dir: str
    stats_path: str
    test_dir: Optional[str] = None

    # If enabled, training batches are balanced by sampling with replacement.
    balance_train: bool = False

    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4
    emb_dim: int = 256
    arc_s: float = 30.0
    arc_m: float = 0.50
    device: str = "cpu"
    save_path: str = "arcface_model_cpu.pt"
    seed: int = 42


def set_seed(seed: int):
    torch.manual_seed(seed)


def load_mean_std(stats_path: str):
    d = json.loads(Path(stats_path).read_text(encoding="utf-8"))
    mean = d["mean_rgb_0_1"]
    std = d["std_rgb_0_1"]
    if len(mean) != 3 or len(std) != 3:
        raise ValueError("stats.json must contain 3-channel mean/std arrays")
    return mean, std


def make_loaders(
    train_dir: str,
    val_dir: str,
    stats_path: str,
    batch_size: int,
    num_workers: int,
    balance_train: bool,
    seed: int,
) -> Tuple[DataLoader, DataLoader, List[str]]:

    mean, std = load_mean_std(stats_path)

    # Train-only augmentation (NOT on disk).
    # Note: RandomRotation uses degrees; for "rotations only" (90/180/270),
    # it's cleaner to use RandomChoice of fixed rotations.
    fixed_rotations = [
        transforms.Lambda(lambda img: img),  # 0 deg
        transforms.Lambda(lambda img: img.rotate(90, expand=False)),
        transforms.Lambda(lambda img: img.rotate(180, expand=False)),
        transforms.Lambda(lambda img: img.rotate(270, expand=False)),
    ]

    train_tf = transforms.Compose([
        transforms.Resize((61, 61)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(
            [transforms.RandomChoice(fixed_rotations)], p=0.7),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((61, 61)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_ds = ImageFolder(train_dir, transform=train_tf)
    val_ds = ImageFolder(val_dir, transform=val_tf)

    sampler = None
    shuffle = True

    if balance_train:
        # ---- CLASS BALANCE IS HANDLED HERE ----
        # Weight each sample inversely proportional to its class count.
        targets = torch.tensor(train_ds.targets, dtype=torch.long)
        class_counts = torch.bincount(targets)
        class_weights = 1.0 / torch.clamp(class_counts.float(), min=1.0)
        sample_weights = class_weights[targets]

        # Deterministic sampling across runs (PyTorch uses generator if provided)
        g = torch.Generator()
        g.manual_seed(seed)

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),   # one epoch ~= dataset length
            replacement=True,                  # allows oversampling minority classes
            generator=g,
        )
        shuffle = False  # sampler and shuffle are mutually exclusive

        counts = Counter(train_ds.targets)
        idx_to_class = {v: k for k, v in train_ds.class_to_idx.items()}
        pretty = {idx_to_class[i]: int(c) for i, c in counts.items()}
        print("Train class counts (on disk):", pretty)
        print("Balanced sampling: ON (WeightedRandomSampler).")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )

    return train_loader, val_loader, train_ds.classes


def make_test_loader(
    test_dir: str,
    stats_path: str,
    batch_size: int,
    num_workers: int,
):
    mean, std = load_mean_std(stats_path)
    test_tf = transforms.Compose([
        transforms.Resize((61, 61)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    test_ds = ImageFolder(test_dir, transform=test_tf)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
    return test_loader, test_ds.classes


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    ce = nn.CrossEntropyLoss()

    correct = 0
    total = 0
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images, labels=None, return_embedding=False)
        loss = ce(logits, labels)

        total_loss += float(loss.item()) * images.size(0)
        preds = logits.argmax(dim=1)
        correct += int((preds == labels).sum().item())
        total += int(labels.numel())

    return total_loss / max(total, 1), correct / max(total, 1)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    ce = nn.CrossEntropyLoss()

    correct = 0
    total = 0
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images, labels=labels, return_embedding=False)
        loss = ce(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * images.size(0)

        # training accuracy: use inference logits (no margin)
        with torch.no_grad():
            logits_infer = model(images, labels=None, return_embedding=False)
            preds = logits_infer.argmax(dim=1)
            correct += int((preds == labels).sum().item())
            total += int(labels.numel())

    return total_loss / max(total, 1), correct / max(total, 1)


def copy_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in state_dict.items()}


def save_checkpoint(path: str, model: nn.Module, class_names: List[str]):
    ckpt: Dict[str, Any] = {
        "state_dict": model.state_dict(),
        "class_names": class_names,
    }
    torch.save(ckpt, path)


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--stats_path", type=str, required=True,
                        help="stats.json computed from TRAIN split only")

    parser.add_argument("--test_dir", type=str, default=None,
                        help="Optional test split directory for final eval")

    parser.add_argument("--balance_train", action="store_true",
                        help="Enable class balancing in the DataLoader using WeightedRandomSampler")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--arc_s", type=float, default=30.0)
    parser.add_argument("--arc_m", type=float, default=0.50)
    parser.add_argument("--save_path", type=str,
                        default="arcface_model_cpu.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = TrainConfig(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        stats_path=args.stats_path,
        test_dir=args.test_dir,
        balance_train=args.balance_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        emb_dim=args.emb_dim,
        arc_s=args.arc_s,
        arc_m=args.arc_m,
        save_path=args.save_path,
        seed=args.seed,
        device="cpu",
    )

    set_seed(cfg.seed)
    device = torch.device("cpu")

    train_loader, val_loader, class_names = make_loaders(
        train_dir=cfg.train_dir,
        val_dir=cfg.val_dir,
        stats_path=cfg.stats_path,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        balance_train=cfg.balance_train,
        seed=cfg.seed,
    )
    num_classes = len(class_names)

    model = MatlabNetArcFace(
        num_classes=num_classes, emb_dim=cfg.emb_dim, s=cfg.arc_s, m=cfg.arc_m).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_acc = -1.0
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:03d}/{cfg.epochs} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.4f}"
        )

        if va_acc > best_acc:
            best_acc = va_acc
            best_state = copy_state_dict(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    save_checkpoint(cfg.save_path, model, class_names)
    print(
        f"Saved best model to: {cfg.save_path} (best val acc={best_acc:.4f})")

    # Optional final test evaluation (no training impact)
    if cfg.test_dir:
        test_loader, test_classes = make_test_loader(
            test_dir=cfg.test_dir,
            stats_path=cfg.stats_path,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )
        if test_classes != class_names:
            print(
                "WARNING: test classes differ from train classes order/names. Check folders.")
        te_loss, te_acc = evaluate(model, test_loader, device)
        print(f"Test loss {te_loss:.4f} acc {te_acc:.4f}")


if __name__ == "__main__":
    main()
