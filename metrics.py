"""
metrics.py — Evaluation Metrics for Rooftop Segmentation
SolarSense Platform | IEEE YESIST12 WePOWER Track 2026

All metric functions accept raw logits and binary targets.
Sigmoid + thresholding is applied internally.
"""

from __future__ import annotations

from collections import defaultdict

import torch


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def iou_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """
    Mean Intersection-over-Union across the batch.

    Args:
        logits    : (B, 1, H, W) raw model output.
        targets   : (B, 1, H, W) binary ground-truth mask.
        threshold : Probability threshold for binarising predictions.

    Returns:
        Mean IoU as a Python float.
    """
    with torch.no_grad():
        preds = (torch.sigmoid(logits) > threshold).float()
        targets = targets.float()

        preds_f   = preds.view(preds.size(0), -1)
        targets_f = targets.view(targets.size(0), -1)

        intersection = (preds_f * targets_f).sum(dim=1)
        union        = preds_f.sum(dim=1) + targets_f.sum(dim=1) - intersection

        iou = (intersection + 1e-6) / (union + 1e-6)
        return iou.mean().item()


def dice_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """
    Mean Dice (F1) coefficient across the batch.

    Args:
        logits    : (B, 1, H, W) raw model output.
        targets   : (B, 1, H, W) binary ground-truth mask.
        threshold : Probability threshold.

    Returns:
        Mean Dice score as a Python float.
    """
    with torch.no_grad():
        preds = (torch.sigmoid(logits) > threshold).float()
        targets = targets.float()

        preds_f   = preds.view(preds.size(0), -1)
        targets_f = targets.view(targets.size(0), -1)

        intersection = (preds_f * targets_f).sum(dim=1)
        dice = (2.0 * intersection + 1e-6) / (preds_f.sum(dim=1) + targets_f.sum(dim=1) + 1e-6)
        return dice.mean().item()


def pixel_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """
    Fraction of correctly classified pixels across the batch.

    Args:
        logits    : (B, 1, H, W) raw model output.
        targets   : (B, 1, H, W) binary ground-truth mask.
        threshold : Probability threshold.

    Returns:
        Pixel accuracy as a Python float in [0, 1].
    """
    with torch.no_grad():
        preds   = (torch.sigmoid(logits) > threshold).float()
        targets = targets.float()
        correct = (preds == targets).float().sum()
        total   = torch.tensor(targets.numel(), dtype=torch.float32)
        return (correct / total).item()


def precision_recall_f1(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute precision, recall, and F1 score for binary segmentation.

    Args:
        logits    : (B, 1, H, W) raw model output.
        targets   : (B, 1, H, W) binary ground-truth mask.
        threshold : Probability threshold.

    Returns:
        Dict with keys 'precision', 'recall', 'f1'.
    """
    with torch.no_grad():
        preds   = (torch.sigmoid(logits) > threshold).float().view(-1)
        targets = targets.float().view(-1)

        tp = (preds * targets).sum().item()
        fp = (preds * (1.0 - targets)).sum().item()
        fn = ((1.0 - preds) * targets).sum().item()

        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)

        return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# MetricTracker
# ---------------------------------------------------------------------------

class MetricTracker:
    """
    Accumulates segmentation metrics over multiple batches and reports means.

    Usage::

        tracker = MetricTracker()
        for logits, targets in loader:
            tracker.update(logits, targets)
        results = tracker.compute()
        tracker.log_summary(epoch=1, phase="val")
        tracker.reset()
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self._storage: dict[str, list[float]] = defaultdict(list)

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Compute metrics for one batch and accumulate.

        Args:
            logits  : (B, 1, H, W) raw model output.
            targets : (B, 1, H, W) binary ground-truth mask.
        """
        t = self.threshold
        self._storage["iou"].append(iou_score(logits, targets, t))
        self._storage["dice"].append(dice_score(logits, targets, t))
        self._storage["pixel_acc"].append(pixel_accuracy(logits, targets, t))

        prf = precision_recall_f1(logits, targets, t)
        for k, v in prf.items():
            self._storage[k].append(v)

    def compute(self) -> dict[str, float]:
        """
        Return mean of each metric accumulated so far.

        Returns:
            Dict with keys: iou, dice, pixel_acc, precision, recall, f1.
        """
        return {k: sum(v) / len(v) for k, v in self._storage.items() if v}

    def reset(self) -> None:
        """Clear all accumulated metric values."""
        self._storage.clear()

    def log_summary(self, epoch: int, phase: str) -> None:
        """
        Pretty-print a metrics summary table.

        Args:
            epoch : Current epoch number.
            phase : One of 'train' or 'val'.
        """
        results = self.compute()
        width = 50
        bar = "─" * width

        print(f"\n┌{bar}┐")
        print(f"│  {phase.upper()} METRICS — Epoch {epoch:<{width - 22}}│")
        print(f"├{bar}┤")
        for key, val in results.items():
            label = key.replace("_", " ").title()
            print(f"│  {label:<20} {val:.4f}{' ' * (width - 28)}│")
        print(f"└{bar}┘")


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    logits  = torch.randn(4, 1, 256, 256)
    targets = torch.randint(0, 2, (4, 1, 256, 256)).float()

    print(f"IoU          : {iou_score(logits, targets):.4f}")
    print(f"Dice         : {dice_score(logits, targets):.4f}")
    print(f"Pixel Acc    : {pixel_accuracy(logits, targets):.4f}")
    print(f"Prec/Rec/F1  : {precision_recall_f1(logits, targets)}")

    tracker = MetricTracker()
    for _ in range(3):
        tracker.update(logits, targets)
    tracker.log_summary(epoch=1, phase="val")
    tracker.reset()
    print("\nMetricTracker tests passed ✓")
