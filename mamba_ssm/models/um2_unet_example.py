"""Example training/testing loop for UM^2-UNet.

This script is intentionally minimal and uses placeholder datasets. Replace the
`DummyUltrasoundDataset` with your ultrasound dataset and preprocessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .um2_unet import UM2UNet, compute_um2_losses


@dataclass
class Batch:
    image: torch.Tensor
    mask: torch.Tensor
    boundary: torch.Tensor
    distance: torch.Tensor


class DummyUltrasoundDataset(Dataset):
    """Replace with your dataset that returns image/mask/boundary/distance."""

    def __init__(self, length: int = 16, image_size: int = 256) -> None:
        self.length = length
        self.image_size = image_size

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image = torch.rand(1, self.image_size, self.image_size)
        mask = (torch.rand(1, self.image_size, self.image_size) > 0.5).float()
        boundary = (torch.rand(1, self.image_size, self.image_size) > 0.5).float()
        distance = torch.randn(1, self.image_size, self.image_size)
        return image, mask, boundary, distance


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for image, mask, boundary, distance in loader:
        image = image.to(device)
        mask = mask.to(device)
        boundary = boundary.to(device)
        distance = distance.to(device)

        outputs = model(image)
        losses = compute_um2_losses(outputs, mask, boundary, distance)
        loss = losses["total"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.flatten(1)
    target = target.flatten(1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    return (2.0 * intersection + eps) / (union + eps)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    dice_total = 0.0
    with torch.no_grad():
        for image, mask, _, _ in loader:
            image = image.to(device)
            mask = mask.to(device)
            outputs = model(image)
            dice_total += dice_score(outputs.pred, mask).mean().item()
    return dice_total / max(len(loader), 1)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UM2UNet(base_channels=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    train_set = DummyUltrasoundDataset(length=16)
    val_set = DummyUltrasoundDataset(length=8)

    train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=2)

    for epoch in range(2):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_dice = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_dice={val_dice:.4f}")


if __name__ == "__main__":
    main()
