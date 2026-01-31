"""UM^2-UNet: Uncertainty-modulated & boundary-reset Mamba U-Net.

This module provides a minimal, reproducible PyTorch reference implementation
that mirrors the methodology described in the paper draft. It emphasizes
clarity over kernel-level efficiency and intentionally avoids heavy
abstractions so users can adapt it for research.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.flatten(1)
    target = target.flatten(1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def evidential_uncertainty(evidence: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    alpha = evidence + 1.0
    strength = alpha.sum(dim=1, keepdim=True)
    return num_classes / strength


def evidential_expected_ce(alpha: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # target: (B, 1, H, W) with {0,1}
    target_one_hot = torch.cat([1.0 - target, target], dim=1)
    strength = alpha.sum(dim=1, keepdim=True)
    digamma_sum = torch.digamma(strength)
    digamma_alpha = torch.digamma(alpha)
    ce = (target_one_hot * (digamma_sum - digamma_alpha)).sum(dim=1, keepdim=True)
    return ce.mean()


def dirichlet_kl(alpha: torch.Tensor) -> torch.Tensor:
    num_classes = alpha.shape[1]
    beta = torch.ones_like(alpha)
    sum_alpha = alpha.sum(dim=1, keepdim=True)
    sum_beta = beta.sum(dim=1, keepdim=True)
    lnB_alpha = torch.lgamma(sum_alpha) - torch.lgamma(alpha).sum(dim=1, keepdim=True)
    lnB_beta = torch.lgamma(sum_beta) - torch.lgamma(beta).sum(dim=1, keepdim=True)
    digamma_sum = torch.digamma(sum_alpha)
    digamma_alpha = torch.digamma(alpha)
    kl = (alpha - beta) * (digamma_alpha - digamma_sum)
    kl = kl.sum(dim=1, keepdim=True) + lnB_alpha - lnB_beta
    return kl.mean()


def selective_scan_reset_ref(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    delta_bias: Optional[torch.Tensor] = None,
    delta_softplus: bool = False,
    reset_gate: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reference selective scan with boundary-conditioned reset.

    Args:
        u: (B, D, L)
        delta: (B, D, L)
        A: (D, N)
        B: (B, N, L)
        C: (B, N, L)
        reset_gate: (B, 1, L) or (B, D, L)
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)

    batch, dim, d_state = u.shape[0], A.shape[0], A.shape[1]
    B = B.float()
    C = C.float()
    x = A.new_zeros((batch, dim, d_state))
    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)

    ys: List[torch.Tensor] = []
    for i in range(u.shape[2]):
        if reset_gate is not None:
            gate = 1.0 - reset_gate[..., i]
            if gate.dim() == 2:
                gate = gate.unsqueeze(-1)
            x = gate * x
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        y = torch.einsum("bdn,bn->bd", x, C[:, :, i])
        ys.append(y)

    y = torch.stack(ys, dim=2)
    out = y if D is None else y + u * D.view(1, -1, 1)
    if z is not None:
        out = out * F.silu(z)
    return out.to(dtype=dtype_in)


class UncertaintyMamba1D(nn.Module):
    """Mamba-style 1D module with uncertainty-modulated dt and boundary reset."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int = 16,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A).repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        uncertainty: Optional[torch.Tensor] = None,
        reset_gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)

        x = x.transpose(1, 2)
        x = self.act(self.conv1d(x)[..., :seqlen])
        x = x.transpose(1, 2)

        x_dbl = self.x_proj(x.reshape(-1, self.d_inner))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dt = self.dt_proj(dt).t().reshape(self.d_inner, batch, seqlen).permute(1, 0, 2)
        B = B.reshape(batch, seqlen, self.d_state).permute(0, 2, 1).contiguous()
        C = C.reshape(batch, seqlen, self.d_state).permute(0, 2, 1).contiguous()

        if uncertainty is not None:
            if uncertainty.dim() == 4:
                uncertainty = uncertainty.flatten(2)
            if uncertainty.shape[1] == 1:
                uncertainty = uncertainty.repeat(1, self.d_inner, 1)
            dt = dt * (1.0 - uncertainty)

        if reset_gate is not None and reset_gate.dim() == 4:
            reset_gate = reset_gate.flatten(2)

        A = -torch.exp(self.A_log.float())
        y = selective_scan_reset_ref(
            u=x.transpose(1, 2),
            delta=dt,
            A=A,
            B=B,
            C=C,
            D=self.D.float(),
            z=z.transpose(1, 2),
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            reset_gate=reset_gate,
        )
        y = y.transpose(1, 2)
        return self.out_proj(y)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UM2Block(nn.Module):
    """Local CNN + global Mamba branch with uncertainty and reset gating."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.local = ConvBlock(in_channels, out_channels)
        self.evidence_head = nn.Conv2d(out_channels, 2, kernel_size=1)
        self.global_mamba = UncertaintyMamba1D(out_channels)
        self.fuse = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        reset_gate: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        local_feat = self.local(x)
        evidence = F.softplus(self.evidence_head(local_feat))
        uncertainty = evidential_uncertainty(evidence)

        b, c, h, w = local_feat.shape
        seq = local_feat.flatten(2).transpose(1, 2)
        global_seq = self.global_mamba(seq, uncertainty=uncertainty, reset_gate=reset_gate)
        global_feat = global_seq.transpose(1, 2).reshape(b, c, h, w)

        fused = self.fuse(torch.cat([local_feat, global_feat], dim=1))
        return fused, uncertainty


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = UM2Block(in_channels=out_channels * 2, out_channels=out_channels)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        reset_gate: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x, reset_gate=reset_gate)


@dataclass
class UM2Outputs:
    pred: torch.Tensor
    pred_coarse: torch.Tensor
    boundary: torch.Tensor
    distance: torch.Tensor
    evidence: torch.Tensor
    uncertainties: List[torch.Tensor]
    reset_gate: torch.Tensor


class UM2UNet(nn.Module):
    """UM^2-UNet model with uncertainty gating, boundary reset, and morphology heads."""

    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        ch = base_channels
        self.enc1 = UM2Block(1, ch)
        self.enc2 = UM2Block(ch, ch * 2)
        self.enc3 = UM2Block(ch * 2, ch * 4)
        self.enc4 = UM2Block(ch * 4, ch * 8)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = UM2Block(ch * 8, ch * 16)

        self.dec4 = UpBlock(ch * 16, ch * 8)
        self.dec3 = UpBlock(ch * 8, ch * 4)
        self.dec2 = UpBlock(ch * 4, ch * 2)
        self.dec1 = UpBlock(ch * 2, ch)

        self.seg_head = nn.Conv2d(ch, 1, kernel_size=1)
        self.coarse_head = nn.Conv2d(ch * 2, 1, kernel_size=1)
        self.boundary_head = nn.Conv2d(ch, 1, kernel_size=1)
        self.distance_head = nn.Conv2d(ch, 1, kernel_size=1)
        self.evidence_head = nn.Conv2d(ch, 2, kernel_size=1)

        self.reset_tau = nn.Parameter(torch.tensor(0.5))
        self.reset_gamma = nn.Parameter(torch.tensor(10.0))

    def forward(self, x: torch.Tensor) -> UM2Outputs:
        e1, u1 = self.enc1(x)
        e2, u2 = self.enc2(self.pool(e1))
        e3, u3 = self.enc3(self.pool(e2))
        e4, u4 = self.enc4(self.pool(e3))

        boundary_logits = self.boundary_head(e1)
        boundary = torch.sigmoid(boundary_logits)
        reset_gate = torch.sigmoid(self.reset_gamma * (boundary - self.reset_tau))

        b, _, h, w = boundary.shape
        reset_down2 = F.interpolate(reset_gate, scale_factor=0.5, mode="bilinear", align_corners=False)
        reset_down4 = F.interpolate(reset_gate, scale_factor=0.25, mode="bilinear", align_corners=False)
        reset_down8 = F.interpolate(reset_gate, scale_factor=0.125, mode="bilinear", align_corners=False)

        btm, u5 = self.bottleneck(self.pool(e4), reset_gate=reset_down8)

        d4, u6 = self.dec4(btm, e4, reset_gate=reset_down8)
        d3, u7 = self.dec3(d4, e3, reset_gate=reset_down4)
        d2, u8 = self.dec2(d3, e2, reset_gate=reset_down2)
        d1, u9 = self.dec1(d2, e1, reset_gate=reset_gate)

        pred = torch.sigmoid(self.seg_head(d1))
        pred_coarse = torch.sigmoid(self.coarse_head(d2))
        distance = self.distance_head(d1)
        evidence = F.softplus(self.evidence_head(d1))

        return UM2Outputs(
            pred=pred,
            pred_coarse=pred_coarse,
            boundary=boundary,
            distance=distance,
            evidence=evidence,
            uncertainties=[u1, u2, u3, u4, u5, u6, u7, u8, u9],
            reset_gate=reset_gate,
        )


def compute_um2_losses(
    outputs: UM2Outputs,
    mask: torch.Tensor,
    boundary_gt: torch.Tensor,
    distance_gt: torch.Tensor,
    lambda_b: float = 1.0,
    lambda_e: float = 0.1,
    lambda_m: float = 1.0,
    lambda_c: float = 0.5,
    lambda_r: float = 0.01,
    beta: float = 0.01,
) -> Dict[str, torch.Tensor]:
    pred = outputs.pred
    seg_bce = F.binary_cross_entropy(pred, mask)
    seg_dice = dice_loss(pred, mask)
    loss_seg = seg_bce + seg_dice

    boundary = outputs.boundary
    boundary_bce = F.binary_cross_entropy(boundary, boundary_gt)
    boundary_dice = dice_loss(boundary, boundary_gt)
    loss_boundary = boundary_bce + boundary_dice

    pred_coarse = F.interpolate(outputs.pred_coarse, size=pred.shape[-2:], mode="bilinear", align_corners=False)
    loss_cf = F.kl_div(torch.log(pred.clamp_min(1e-6)), pred_coarse.detach(), reduction="batchmean")

    morph_l1 = torch.abs(outputs.distance - distance_gt)
    uncertainty = outputs.uncertainties[0]
    if uncertainty.dim() == 4:
        uncertainty = uncertainty
    loss_morph = ((1.0 - uncertainty) * morph_l1).mean()

    loss_reset = outputs.reset_gate.mean()

    alpha = outputs.evidence + 1.0
    loss_evi = evidential_expected_ce(alpha, mask) + beta * dirichlet_kl(alpha)

    total = (
        loss_seg
        + lambda_b * loss_boundary
        + lambda_e * loss_evi
        + lambda_m * loss_morph
        + lambda_c * loss_cf
        + lambda_r * loss_reset
    )
    return {
        "total": total,
        "seg": loss_seg,
        "boundary": loss_boundary,
        "evidential": loss_evi,
        "morph": loss_morph,
        "coarse": loss_cf,
        "reset": loss_reset,
    }
