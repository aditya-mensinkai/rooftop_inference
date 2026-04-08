"""
model.py — U-Net with ResNet34 Encoder for Binary Rooftop Segmentation
SolarSense Platform | IEEE YESIST12 WePOWER Track 2026

Architecture:
    Encoder : Pretrained ResNet34 (ImageNet weights)
    Decoder : 4 upsampling blocks with skip connections
    Output  : Single-channel logit map (sigmoid applied in loss)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet34_Weights


# ---------------------------------------------------------------------------
# Decoder Block
# ---------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    """
    Single upsampling decoder block for U-Net.

    Pipeline per block:
        Bilinear upsample ×2 → Concat skip → Conv→BN→ReLU → Conv→BN→ReLU

    Args:
        in_channels   : Channels coming from the previous decoder stage.
        skip_channels : Channels from the corresponding encoder skip connection.
        out_channels  : Output channels of this block.
        dropout       : If > 0, adds Dropout2d after the second conv.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Double Conv (Bottleneck)
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """Two consecutive Conv→BN→ReLU blocks used in the bottleneck."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.block(x))


# ---------------------------------------------------------------------------
# U-Net + ResNet34
# ---------------------------------------------------------------------------

class UNetResNet34(nn.Module):
    """
    U-Net with a pretrained ResNet34 encoder for binary rooftop segmentation.

    Encoder skip-connection feature map sizes (for 256×256 input):
        layer0 (stem) : (B, 64,  128, 128)
        layer1        : (B, 64,  64,  64)
        layer2        : (B, 128, 32,  32)
        layer3        : (B, 256, 16,  16)
        layer4        : (B, 512, 8,   8)   ← bottleneck input

    Args:
        pretrained : Load ImageNet weights for the ResNet34 encoder.
        dropout    : Dropout probability applied in the bottleneck and last
                     decoder block only.
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.3) -> None:
        super().__init__()

        # ── Encoder ────────────────────────────────────────────────────────
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.resnet34(weights=weights)

        # Stem: Conv7×7 → BN → ReLU  (no MaxPool — kept separate for skip)
        self.encoder0 = nn.Sequential(base.conv1, base.bn1, base.relu)  # → 64ch, H/2
        self.pool     = base.maxpool                                      # → 64ch, H/4
        self.encoder1 = base.layer1   # → 64ch,  H/4
        self.encoder2 = base.layer2   # → 128ch, H/8
        self.encoder3 = base.layer3   # → 256ch, H/16
        self.encoder4 = base.layer4   # → 512ch, H/32

        # ── Bottleneck ──────────────────────────────────────────────────────
        self.bottleneck = DoubleConv(512, 512, dropout=dropout)

        # ── Decoder ─────────────────────────────────────────────────────────
        # Block 1: up from H/32 → H/16,  skip from encoder3 (256ch)
        self.decoder1 = DecoderBlock(512, 256, 256)
        # Block 2: up from H/16 → H/8,   skip from encoder2 (128ch)
        self.decoder2 = DecoderBlock(256, 128, 128)
        # Block 3: up from H/8  → H/4,   skip from encoder1 (64ch)
        self.decoder3 = DecoderBlock(128, 64,  64)
        # Block 4: up from H/4  → H/2,   skip from encoder0 (64ch)
        self.decoder4 = DecoderBlock(64,  64,  32, dropout=dropout)

        # Final upsample H/2 → H, then 1×1 head
        self.final_upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.head = nn.Conv2d(32, 1, kernel_size=1)

        # ── Weight init for decoder ─────────────────────────────────────────
        self._init_decoder_weights()

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, 3, H, W) normalised RGB image tensor.

        Returns:
            Logit map of shape (B, 1, H, W). Apply sigmoid externally.
        """
        # Encoder
        s0 = self.encoder0(x)          # (B, 64,  H/2,  W/2)
        s1 = self.encoder1(self.pool(s0))  # (B, 64,  H/4,  W/4)
        s2 = self.encoder2(s1)         # (B, 128, H/8,  W/8)
        s3 = self.encoder3(s2)         # (B, 256, H/16, W/16)
        s4 = self.encoder4(s3)         # (B, 512, H/32, W/32)

        # Bottleneck
        b = self.bottleneck(s4)        # (B, 512, H/32, W/32)

        # Decoder
        d1 = self.decoder1(b,  s3)    # (B, 256, H/16, W/16)
        d2 = self.decoder2(d1, s2)    # (B, 128, H/8,  W/8)
        d3 = self.decoder3(d2, s1)    # (B, 64,  H/4,  W/4)
        d4 = self.decoder4(d3, s0)    # (B, 32,  H/2,  W/2)

        out = self.final_upsample(d4)  # (B, 32,  H,    W)
        return self.head(out)          # (B, 1,   H,    W)

    # ── Encoder freeze / unfreeze ──────────────────────────────────────────

    def _encoder_modules(self) -> list[nn.Module]:
        return [self.encoder0, self.encoder1, self.encoder2, self.encoder3, self.encoder4]

    def freeze_encoder(self) -> None:
        """Freeze all encoder parameters (Phase 1 — decoder warmup)."""
        for module in self._encoder_modules():
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Unfreeze all encoder parameters (Phase 2 — full fine-tune)."""
        for module in self._encoder_modules():
            for param in module.parameters():
                param.requires_grad = True

    def get_encoder_params(self) -> list[nn.Parameter]:
        """Return a list of encoder parameters (for differential LR optimiser)."""
        params = []
        for module in self._encoder_modules():
            params.extend(module.parameters())
        return params

    def get_decoder_params(self) -> list[nn.Parameter]:
        """Return a list of decoder + head parameters."""
        encoder_ids = {id(p) for p in self.get_encoder_params()}
        return [p for p in self.parameters() if id(p) not in encoder_ids]

    # ── Weight initialisation ──────────────────────────────────────────────

    def _init_decoder_weights(self) -> None:
        decoder_modules = [
            self.bottleneck,
            self.decoder1, self.decoder2, self.decoder3, self.decoder4,
            self.head,
        ]
        for module in decoder_modules:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# Model summary helper
# ---------------------------------------------------------------------------

def print_model_summary(model: UNetResNet34, input_shape: tuple = (1, 3, 256, 256)) -> None:
    """Print a layer-wise summary table for the given input shape."""

    device = next(model.parameters()).device
    x = torch.zeros(*input_shape, device=device)

    hooks: list = []
    summary: list[dict] = []

    def _hook(name: str):
        def fn(module, inp, out):
            if isinstance(out, torch.Tensor):
                summary.append({
                    "layer": name,
                    "output_shape": tuple(out.shape),
                    "params": sum(p.numel() for p in module.parameters()),
                })
        return fn

    named = [
        ("encoder0",       model.encoder0),
        ("pool",           model.pool),
        ("encoder1",       model.encoder1),
        ("encoder2",       model.encoder2),
        ("encoder3",       model.encoder3),
        ("encoder4",       model.encoder4),
        ("bottleneck",     model.bottleneck),
        ("decoder1",       model.decoder1),
        ("decoder2",       model.decoder2),
        ("decoder3",       model.decoder3),
        ("decoder4",       model.decoder4),
        ("final_upsample", model.final_upsample),
        ("head",           model.head),
    ]

    for name, mod in named:
        hooks.append(mod.register_forward_hook(_hook(name)))

    with torch.no_grad():
        model(x)

    for h in hooks:
        h.remove()

    col = (28, 22, 12)
    sep = "─" * (sum(col) + 6)
    print(f"\n{'MODEL SUMMARY':^{sum(col)+6}}")
    print(sep)
    print(f"{'Layer':<{col[0]}} {'Output Shape':<{col[1]}} {'Params':>{col[2]}}")
    print(sep)
    total_params = 0
    for row in summary:
        print(f"{row['layer']:<{col[0]}} {str(row['output_shape']):<{col[1]}} {row['params']:>{col[2]},}")
        total_params += row["params"]
    print(sep)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params       : {total:>12,}")
    print(f"Trainable params   : {trainable:>12,}")
    print(f"Non-trainable      : {total - trainable:>12,}")
    print(sep)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = UNetResNet34(pretrained=True, dropout=0.3)
    model.eval()
    print_model_summary(model)

    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, 256, 256), f"Unexpected output shape: {out.shape}"
    print(f"\n✓ Forward pass OK — output shape: {out.shape}")

    model.freeze_encoder()
    frozen = sum(1 for p in model.get_encoder_params() if not p.requires_grad)
    print(f"✓ freeze_encoder() — {frozen} encoder params frozen")

    model.unfreeze_encoder()
    unfrozen = sum(1 for p in model.get_encoder_params() if p.requires_grad)
    print(f"✓ unfreeze_encoder() — {unfrozen} encoder params trainable")
