"""Train/validate UM^2-UNet on BUSI with full metric reporting.

Expected directory layout:
  images_dir: /home/datasets/BC/BUSI/images
  masks_dir:  /home/datasets/BC/BUSI/labels

Masks should share filename stems with images (e.g., xxx.png in both).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage as ndi
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from .um2_unet import UM2UNet, compute_um2_losses


@dataclass
class Busisample:
    image: torch.Tensor
    mask: torch.Tensor
    boundary: torch.Tensor
    distance: torch.Tensor


def load_image(path: Path, size: int) -> torch.Tensor:
    image = Image.open(path).convert("L").resize((size, size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - array.mean()) / (array.std() + 1e-6)
    return torch.from_numpy(array).unsqueeze(0)


def load_mask(path: Path, size: int) -> torch.Tensor:
    mask = Image.open(path).convert("L").resize((size, size), resample=Image.NEAREST)
    array = np.asarray(mask, dtype=np.float32)
    array = (array > 127).astype(np.float32)
    return torch.from_numpy(array).unsqueeze(0)


def morphological_boundary(mask: np.ndarray) -> np.ndarray:
    structure = np.ones((3, 3), dtype=bool)
    dilated = ndi.binary_dilation(mask, structure=structure)
    eroded = ndi.binary_erosion(mask, structure=structure)
    boundary = np.logical_xor(dilated, eroded)
    return boundary.astype(np.float32)


def signed_distance(mask: np.ndarray) -> np.ndarray:
    inside = ndi.distance_transform_edt(mask)
    outside = ndi.distance_transform_edt(~mask)
    return outside - inside


class BUSIDataset(Dataset):
    def __init__(self, images_dir: Path, masks_dir: Path, image_size: int) -> None:
        self.image_size = image_size
        self.items = self._pair_items(images_dir, masks_dir)

    @staticmethod
    def _pair_items(images_dir: Path, masks_dir: Path) -> List[Tuple[Path, Path]]:
        image_files = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
        mask_files = {p.stem: p for p in masks_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}}
        pairs = []
        for image_path in image_files:
            mask_path = mask_files.get(image_path.stem)
            if mask_path is None:
                continue
            pairs.append((image_path, mask_path))
        if not pairs:
            raise ValueError("No image/mask pairs found. Check your BUSI paths.")
        return pairs

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Busisample:
        image_path, mask_path = self.items[idx]
        image = load_image(image_path, self.image_size)
        mask = load_mask(mask_path, self.image_size)
        mask_np = mask.squeeze(0).numpy().astype(bool)
        boundary = morphological_boundary(mask_np)
        distance = signed_distance(mask_np)
        return Busisample(
            image=image,
            mask=mask,
            boundary=torch.from_numpy(boundary).unsqueeze(0),
            distance=torch.from_numpy(distance).unsqueeze(0).float(),
        )


def dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.flatten(1)
    target = target.flatten(1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    return (2.0 * intersection + eps) / (union + eps)


def iou_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.flatten(1)
    target = target.flatten(1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - intersection
    return (intersection + eps) / (union + eps)


def surface_distances(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    pred_surface = pred ^ ndi.binary_erosion(pred)
    gt_surface = gt ^ ndi.binary_erosion(gt)
    gt_dist = ndi.distance_transform_edt(~gt_surface)
    pred_dist = ndi.distance_transform_edt(~pred_surface)
    distances = np.concatenate([gt_dist[pred_surface], pred_dist[gt_surface]])
    if distances.size == 0:
        return np.array([0.0], dtype=np.float32)
    return distances.astype(np.float32)


def hd95_assd(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    distances = surface_distances(pred, gt)
    hd95 = float(np.percentile(distances, 95))
    assd = float(distances.mean())
    return hd95, assd


def boundary_fscore(pred: np.ndarray, gt: np.ndarray, tolerance: int = 2) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    pred_surface = pred ^ ndi.binary_erosion(pred)
    gt_surface = gt ^ ndi.binary_erosion(gt)
    pred_dil = ndi.binary_dilation(pred_surface, iterations=tolerance)
    gt_dil = ndi.binary_dilation(gt_surface, iterations=tolerance)
    tp = np.logical_and(pred_surface, gt_dil).sum()
    fp = np.logical_and(pred_surface, ~gt_dil).sum()
    fn = np.logical_and(gt_surface, ~pred_dil).sum()
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    return float(2 * precision * recall / (precision + recall + 1e-6))


def expected_calibration_error(probs: torch.Tensor, targets: torch.Tensor, bins: int = 10) -> float:
    probs = probs.flatten()
    targets = targets.flatten()
    bin_edges = torch.linspace(0, 1, bins + 1, device=probs.device)
    ece = torch.tensor(0.0, device=probs.device)
    for i in range(bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if mask.any():
            acc = targets[mask].mean()
            conf = probs[mask].mean()
            ece += (mask.float().mean()) * (acc - conf).abs()
    return float(ece.item())


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        image = batch.image.to(device)
        mask = batch.mask.to(device)
        boundary = batch.boundary.to(device)
        distance = batch.distance.to(device)

        outputs = model(image)
        losses = compute_um2_losses(outputs, mask, boundary, distance)
        loss = losses["total"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    metrics = {"dice": 0.0, "iou": 0.0, "hd95": 0.0, "assd": 0.0, "bf": 0.0, "ece": 0.0}
    count = 0
    with torch.no_grad():
        for batch in loader:
            image = batch.image.to(device)
            mask = batch.mask.to(device)
            outputs = model(image)
            pred = outputs.pred
            pred_bin = (pred > 0.5).float()

            metrics["dice"] += dice_score(pred_bin, mask).mean().item()
            metrics["iou"] += iou_score(pred_bin, mask).mean().item()
            metrics["ece"] += expected_calibration_error(pred, mask)

            for i in range(pred.shape[0]):
                pred_np = pred_bin[i, 0].cpu().numpy().astype(bool)
                gt_np = mask[i, 0].cpu().numpy().astype(bool)
                hd95, assd = hd95_assd(pred_np, gt_np)
                bf = boundary_fscore(pred_np, gt_np)
                metrics["hd95"] += hd95
                metrics["assd"] += assd
                metrics["bf"] += bf
                count += 1

    for key in metrics:
        metrics[key] /= max(count, 1)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train UM^2-UNet on BUSI.")
    parser.add_argument("--images-dir", type=Path, default=Path("/home/datasets/BC/BUSI/images"))
    parser.add_argument("--masks-dir", type=Path, default=Path("/home/datasets/BC/BUSI/labels"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = BUSIDataset(args.images_dir, args.masks_dir, args.image_size)
    val_len = int(len(dataset) * args.val_split)
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    model = UM2UNet(base_channels=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        metrics = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch:03d} | "
            f"loss={train_loss:.4f} | "
            f"dice={metrics['dice']:.4f} | "
            f"iou={metrics['iou']:.4f} | "
            f"hd95={metrics['hd95']:.4f} | "
            f"assd={metrics['assd']:.4f} | "
            f"bf={metrics['bf']:.4f} | "
            f"ece={metrics['ece']:.4f}"
        )


if __name__ == "__main__":
    main()
