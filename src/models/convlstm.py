"""ConvLSTM cell and stacked ConvLSTM / U-Net+ConvLSTM models for
spatiotemporal forest cover prediction — Deoghar Forest Analysis.

Adapted from ShivekRanjan/bengaluru-lulc-forecast- (Shi et al. 2015).

Input shape: (B, T, C, H, W)
  B = batch, T = time steps, C = channels (bands + indices + static),
  H, W = tile spatial dimensions

Output: (B, n_classes, H, W) logit map for the next-year LULC frame.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell — all 4 gates fused in one convolution."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3, bias: bool = True):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels=in_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

    def forward(self, x: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = state
        combined = torch.cat([x, h_prev], dim=1)
        gates    = self.conv(combined)
        i, f, g, o = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

    def init_state(self, batch: int, height: int, width: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch, self.hidden_channels, height, width, device=device)
        c = torch.zeros(batch, self.hidden_channels, height, width, device=device)
        return h, c


class ConvLSTM(nn.Module):
    """Stacked ConvLSTM encoder + per-pixel classification head.

    Args:
        in_channels: input channels per timestep
            (e.g. 6 S2 bands + NDVI + NBR + NDWI + DEM + slope = 11)
        hidden_channels: list of hidden sizes for stacked layers
        kernel_size: conv kernel for all layers
        n_classes: LULC output classes (6 for Deoghar project)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int] = (32, 64),
        kernel_size: int = 3,
        n_classes: int = 6,
    ) -> None:
        super().__init__()
        self.hidden_channels = list(hidden_channels)
        cells = []
        prev_c = in_channels
        for h in self.hidden_channels:
            cells.append(ConvLSTMCell(prev_c, h, kernel_size))
            prev_c = h
        self.cells = nn.ModuleList(cells)
        # Classification head on the last layer's final hidden state
        self.head = nn.Sequential(
            nn.Conv2d(self.hidden_channels[-1], self.hidden_channels[-1], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.25),   # MC Dropout support
            nn.Conv2d(self.hidden_channels[-1], n_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C, H, W) → logits (B, n_classes, H, W)"""
        b, t, c, h, w = x.shape
        device = x.device
        states = [cell.init_state(b, h, w, device) for cell in self.cells]
        for ti in range(t):
            inp = x[:, ti]
            for li, cell in enumerate(self.cells):
                states[li] = cell(inp, states[li])
                inp = states[li][0]
        return self.head(states[-1][0])


class UNetConvLSTM(nn.Module):
    """U-Net spatial encoder/decoder with ConvLSTM temporal bottleneck.

    Architecture:
      - 3-level U-Net encoder applied independently to each timestep
      - ConvLSTM cell at the bottleneck for temporal state propagation
      - 2-level decoder with skip connections from the LAST timestep encoder
      - Dropout2d in the bottleneck for MC Dropout uncertainty estimation

    Lightweight enough to train on a 4GB GPU at patch=128, batch=1.
    """

    def __init__(
        self,
        in_channels: int,
        n_classes: int = 6,
        base_channels: int = 16,
        bottleneck_hidden: int = 64,
    ) -> None:
        super().__init__()
        c  = base_channels
        # Encoder
        self.enc1 = self._block(in_channels, c)
        self.enc2 = self._block(c,      c * 2)
        self.enc3 = self._block(c * 2,  c * 4)
        self.pool = nn.MaxPool2d(2)
        # Temporal bottleneck
        self.temporal = ConvLSTMCell(c * 4, bottleneck_hidden)
        self.bn_drop  = nn.Dropout2d(p=0.3)  # MC Dropout at bottleneck
        # Decoder
        self.up3  = nn.ConvTranspose2d(bottleneck_hidden, c * 2, 2, stride=2)
        self.dec3 = self._block(c * 4, c * 2)
        self.up2  = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec2 = self._block(c * 2, c)
        self.head = nn.Conv2d(c, n_classes, 1)

    @staticmethod
    def _block(in_c: int, out_c: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def _encode_step(self, x: torch.Tensor):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        return e1, e2, e3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C, H, W) → logits (B, n_classes, H, W)"""
        b, t, c, h, w = x.shape
        device = x.device
        h_bot  = h // 4   # after 2 MaxPool2d(2)
        w_bot  = w // 4
        state  = self.temporal.init_state(b, h_bot, w_bot, device)
        e1_last = e2_last = None
        for ti in range(t):
            e1, e2, e3 = self._encode_step(x[:, ti])
            state       = self.temporal(e3, state)
            e1_last, e2_last = e1, e2
        h_t = self.bn_drop(state[0])
        d3  = self.up3(h_t)
        d3  = self.dec3(torch.cat([d3, e2_last], dim=1))
        d2  = self.up2(d3)
        d2  = self.dec2(torch.cat([d2, e1_last], dim=1))
        return self.head(d2)


if __name__ == "__main__":
    torch.manual_seed(42)
    print("=== Smoke test: 6-class Deoghar models ===")
    x = torch.randn(1, 8, 11, 128, 128)

    m1 = ConvLSTM(in_channels=11, hidden_channels=[32, 64], n_classes=6)
    y1 = m1(x)
    assert y1.shape == (1, 6, 128, 128), f"ConvLSTM shape mismatch: {y1.shape}"
    print(f"ConvLSTM      out={y1.shape}  params={sum(p.numel() for p in m1.parameters()):,}")

    m2 = UNetConvLSTM(in_channels=11, n_classes=6, base_channels=16, bottleneck_hidden=64)
    y2 = m2(x)
    assert y2.shape == (1, 6, 128, 128), f"UNetConvLSTM shape mismatch: {y2.shape}"
    print(f"UNetConvLSTM  out={y2.shape}  params={sum(p.numel() for p in m2.parameters()):,}")
    print("OK")
