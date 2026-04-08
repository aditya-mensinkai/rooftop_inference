"""
Rooftop Segmentation Inference Script

Predicts rooftop segmentation mask from a single satellite image
and calculates area metrics.

Usage:
    python predict.py --image_path sample.tif --gsd 0.5
    python predict.py --image_path sample.tif --threshold 0.4 --min_area 100 --top_k 20

Output:
    - outputs/mask.png         : Binary segmentation mask
    - outputs/viz.png          : Side-by-side visualization
    - outputs/debug_prob.png   : Debug view (original, probability, cleaned mask)
    - Console                  : Area metrics with post-processing stats
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from area_utils import pixels_to_area
from model import UNetResNet34

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ImageNet normalization constants
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

# Model input size
TARGET_SIZE = 256


def load_model(model_path: str, device: torch.device) -> UNetResNet34:
    """
    Load trained UNetResNet34 model from checkpoint.

    Args:
        model_path: Path to model checkpoint (.pth file)
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    logger.info(f"Loading model from: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    # Initialize model (pretrained=False since we're loading trained weights)
    model = UNetResNet34(pretrained=False, dropout=0.3)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
        logger.info(f"  Loaded from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        # Assume it's just the state dict
        model.load_state_dict(checkpoint)

    # FORCE EVERYTHING TO FLOAT32
    model = model.to(device).float()

    # EXTRA SAFETY: Ensure all parameters are float32
    for param in model.parameters():
        param.data = param.data.float()

    model.eval()

    logger.info(f"  Model loaded successfully on {device}")
    return model


def preprocess(image_path: str, device: torch.device) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Preprocess image for inference.

    Args:
        image_path: Path to input image
        device: Device to move tensor to

    Returns:
        Tuple of (preprocessed tensor [1,3,256,256], original image array)
    """
    logger.info(f"Preprocessing image: {image_path}")

    # Load image using rasterio to match training pipeline (dataset.py)
    import rasterio
    with rasterio.open(image_path) as src:
        if src.count >= 3:
            image = np.dstack([src.read(i) for i in range(1, 4)])
        elif src.count == 1:
            gray = src.read(1)
            image = np.stack([gray] * 3, axis=-1)
        else:
            raise ValueError(f"Unsupported band count: {src.count}")

        # Store original for visualization (before any processing)
        if image.dtype != np.uint8:
            if image.max() > 255:
                original = (image / image.max() * 255).astype(np.uint8)
            else:
                original = image.astype(np.uint8)
        else:
            original = image.copy()

        # Ensure uint8 for consistent processing (matches dataset.py)
        if image.dtype != np.uint8:
            if image.max() > 255:
                image = (image / image.max() * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

    # Resize to target size using PIL (matches dataset.py preprocessing)
    from PIL import Image
    image_pil = Image.fromarray(image)
    image_pil = image_pil.resize((TARGET_SIZE, TARGET_SIZE), Image.BILINEAR)
    image = np.array(image_pil)

    # Convert to float32
    image = image.astype(np.float32)

    # Normalize to [0, 1] (matches training in dataset.py)
    image = image / 255.0

    # Convert to tensor: (H, W, C) -> (C, H, W)
    image_tensor = torch.from_numpy(image).permute(2, 0, 1)

    # Add batch dimension: (1, C, H, W) and ensure float32
    image_tensor = image_tensor.unsqueeze(0).to(device).float()

    logger.info(f"  Image shape: {original.shape} -> tensor {image_tensor.shape}")
    return image_tensor, original


def predict(model: UNetResNet34, image_tensor: torch.Tensor) -> torch.Tensor:
    """
    Run inference on preprocessed image.

    Args:
        model: Loaded UNetResNet34 model
        image_tensor: Preprocessed image tensor [1,3,H,W]

    Returns:
        Probability tensor [H,W] with values in [0, 1]
    """
    logger.info("Running inference...")

    # DEBUG: Print dtypes and input range
    print("  Model dtype:", next(model.parameters()).dtype)
    print("  Input dtype:", image_tensor.dtype)
    print("  Input min:", image_tensor.min().item())
    print("  Input max:", image_tensor.max().item())

    # EXTRA SAFETY: Ensure float32
    image_tensor = image_tensor.float()

    with torch.no_grad():
        # Forward pass (returns logits)
        logits = model(image_tensor)

        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)

        # DEBUG: Print probability range
        print("  Probability range: [{:.4f}, {:.4f}]".format(probs.min().item(), probs.max().item()))

    # Remove batch and channel dimensions -> (H, W)
    probs = probs.squeeze()

    logger.info(f"  Prediction complete. Prob shape: {probs.shape}")
    return probs


def apply_gaussian_smoothing(probs: np.ndarray, kernel_size: Tuple[int, int] = (5, 5), sigma: float = 0) -> np.ndarray:
    """
    Apply Gaussian smoothing to probability map.

    Args:
        probs: Probability map [H,W] with values in [0, 1]
        kernel_size: Gaussian kernel size (default: (5, 5))
        sigma: Gaussian sigma (default: 0 = auto-computed from kernel size)

    Returns:
        Smoothed probability map [H,W]
    """
    smoothed = cv2.GaussianBlur(probs, kernel_size, sigma)
    return smoothed


def apply_threshold(probs: np.ndarray, threshold: float) -> np.ndarray:
    """
    Apply threshold to probability map to get binary mask.

    Args:
        probs: Probability map [H,W]
        threshold: Threshold value (0-1)

    Returns:
        Binary mask [H,W] with values 0 or 1
    """
    mask = (probs > threshold).astype(np.uint8)
    return mask


def apply_morphological_cleaning(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply morphological opening and closing to clean mask.
    Opening removes small noise blobs, closing fills small holes.

    Args:
        mask: Binary mask [H,W] with values 0 or 1
        kernel_size: Size of morphological kernel (default: 3)

    Returns:
        Cleaned binary mask [H,W]
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Opening: erosion followed by dilation (removes small noise)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Closing: dilation followed by erosion (fills small holes)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def remove_small_components(
    mask: np.ndarray,
    min_area: int = 100,
    top_k: Optional[int] = None
) -> Tuple[np.ndarray, int, int]:
    """
    Remove small connected components from binary mask.
    Optionally keep only top-K largest components.

    Args:
        mask: Binary mask [H,W] with values 0 or 1
        min_area: Minimum component area in pixels (default: 100)
        top_k: If specified, keep only top-K largest components (default: None)

    Returns:
        Tuple of (cleaned_mask, total_components, kept_components)
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned_mask = np.zeros_like(mask)

    # Get all component areas (excluding background at index 0)
    areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
    total_components = len(areas)

    if total_components == 0:
        return cleaned_mask, 0, 0

    # Filter by minimum area first
    valid_components = [(i, area) for i, area in areas if area >= min_area]

    # Optional: Keep only top-K largest components
    if top_k is not None and len(valid_components) > top_k:
        valid_components = sorted(valid_components, key=lambda x: x[1], reverse=True)
        valid_components = valid_components[:top_k]

    # Build clean mask
    for i, _ in valid_components:
        cleaned_mask[labels == i] = 1

    kept = len(valid_components)
    return cleaned_mask, total_components, kept


def save_debug_outputs(
    original: np.ndarray,
    prob_heatmap: np.ndarray,
    cleaned_mask: np.ndarray,
    output_dir: str,
) -> str:
    """
    Save debug visualization with original, probability heatmap, and cleaned mask.

    Args:
        original: Original image array [H,W,3]
        prob_heatmap: Probability heatmap [H,W] with values in [0, 1]
        cleaned_mask: Cleaned binary mask [H,W] with values 0 or 1
        output_dir: Directory to save outputs

    Returns:
        Path to saved debug image
    """
    # Resize original to match mask size if needed
    if original.shape[:2] != cleaned_mask.shape:
        original = cv2.resize(original, (cleaned_mask.shape[1], cleaned_mask.shape[0]))

    # Create 3-panel visualization
    viz_height = cleaned_mask.shape[0]
    viz_width = cleaned_mask.shape[1] * 3
    viz_img = Image.new("RGB", (viz_width, viz_height))

    # Panel 1: Original image
    original_pil = Image.fromarray(original)
    viz_img.paste(original_pil, (0, 0))

    # Panel 2: Probability heatmap (convert to colormap)
    prob_normalized = (prob_heatmap * 255).astype(np.uint8)
    prob_colored = cv2.applyColorMap(prob_normalized, cv2.COLORMAP_JET)
    prob_colored = cv2.cvtColor(prob_colored, cv2.COLOR_BGR2RGB)
    prob_pil = Image.fromarray(prob_colored)
    viz_img.paste(prob_pil, (cleaned_mask.shape[1], 0))

    # Panel 3: Cleaned binary mask (white on black)
    mask_rgb = np.zeros((*cleaned_mask.shape, 3), dtype=np.uint8)
    mask_rgb[cleaned_mask == 1] = [255, 255, 255]
    mask_pil = Image.fromarray(mask_rgb)
    viz_img.paste(mask_pil, (cleaned_mask.shape[1] * 2, 0))

    # Save visualization
    debug_path = os.path.join(output_dir, "debug_prob.png")
    viz_img.save(debug_path)
    logger.info(f"  Saved debug visualization: {debug_path}")

    return debug_path


def save_outputs(
    original: np.ndarray,
    binary_mask: np.ndarray,
    output_dir: str,
) -> Tuple[str, str]:
    """
    Save mask and visualization images.

    Args:
        original: Original image array [H,W,3]
        binary_mask: Binary mask array [H,W] with values 0 or 1
        output_dir: Directory to save outputs

    Returns:
        Tuple of (mask_path, viz_path)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save mask as grayscale image (0-255)
    mask_path = os.path.join(output_dir, "mask.png")
    mask_img = (binary_mask * 255).astype(np.uint8)
    Image.fromarray(mask_img, mode="L").save(mask_path)
    logger.info(f"  Saved mask: {mask_path}")

    # Create side-by-side visualization
    # Resize original to match mask size if needed
    if original.shape[:2] != binary_mask.shape:
        original = cv2.resize(original, (binary_mask.shape[1], binary_mask.shape[0]))

    # Convert original to PIL Image
    original_pil = Image.fromarray(original)

    # Create overlay
    mask_rgba = np.zeros((*binary_mask.shape, 4), dtype=np.uint8)
    mask_rgba[binary_mask == 1] = [255, 0, 0, 128]  # Red semi-transparent
    mask_overlay = Image.fromarray(mask_rgba)
    overlay_img = Image.alpha_composite(
        original_pil.convert("RGBA"),
        mask_overlay
    ).convert("RGB")

    # Create side-by-side visualization
    viz_width = binary_mask.shape[1] * 3
    viz_height = binary_mask.shape[0]
    viz_img = Image.new("RGB", (viz_width, viz_height))

    # Original image
    viz_img.paste(original_pil, (0, 0))

    # Predicted mask (grayscale -> RGB)
    mask_rgb = Image.fromarray(mask_img).convert("RGB")
    viz_img.paste(mask_rgb, (binary_mask.shape[1], 0))

    # Overlay
    viz_img.paste(overlay_img, (binary_mask.shape[1] * 2, 0))

    # Save visualization
    viz_path = os.path.join(output_dir, "viz.png")
    viz_img.save(viz_path)
    logger.info(f"  Saved visualization: {viz_path}")

    return mask_path, viz_path


def print_results(
    area_info: dict,
    threshold: float,
    roof_pixels_pct: float,
    components_before: int,
    components_after: int,
) -> None:
    """
    Print formatted prediction results with post-processing stats.

    Args:
        area_info: Output from pixels_to_area()
        threshold: Threshold value used
        roof_pixels_pct: Percentage of pixels predicted as roof
        components_before: Number of components before filtering
        components_after: Number of components after filtering
    """
    print("\n" + "=" * 50)
    print("===== POST-PROCESSING STATS =====")
    print("=" * 50)
    print(f"Threshold used:        {threshold:.2f}")
    print(f"Roof pixels:           {roof_pixels_pct:.2f}%")
    print(f"Components before:     {components_before}")
    print(f"Components after:      {components_after} (removed {components_before - components_after})")

    print("\n" + "=" * 50)
    print("===== PREDICTION RESULTS =====")
    print("=" * 50)
    print(f"Roof pixels:     {area_info['roof_pixels']:,.0f}")
    print(f"Total area:      {area_info['total_roof_area_m2']:,.1f} m²")
    print(f"Usable area:     {area_info['usable_area_m2']:,.1f} m²")
    print("=" * 50 + "\n")


def resolve_gsd(gsd_arg: Optional[float], zoom_arg: Optional[int]) -> float:
    """
    Resolve GSD value from explicit argument or zoom level.

    Priority:
    1. Explicit --gsd value (warns if --zoom also provided)
    2. --zoom level mapping (18 -> 0.3, 19 -> 0.15)
    3. Default fallback (0.3 for SpaceNet dataset compatibility)

    Args:
        gsd_arg: Explicit GSD value from --gsd
        zoom_arg: Zoom level from --zoom

    Returns:
        Resolved GSD in meters per pixel

    Raises:
        ValueError: If gsd <= 0 or unsupported zoom level
    """
    # Priority 1: Explicit GSD provided
    if gsd_arg is not None:
        if gsd_arg <= 0:
            raise ValueError(f"GSD must be positive, got: {gsd_arg}")
        if zoom_arg is not None:
            logger.warning(f"Both --gsd ({gsd_arg}) and --zoom ({zoom_arg}) provided. Using --gsd.")
        return float(gsd_arg)

    # Priority 2: Zoom level mapping
    if zoom_arg is not None:
        zoom_to_gsd = {
            19: 0.15,  # Google Maps zoom 19 (~6 inch resolution)
            18: 0.30,  # Google Maps zoom 18 (~1 foot resolution)
        }
        if zoom_arg not in zoom_to_gsd:
            supported = list(zoom_to_gsd.keys())
            raise ValueError(f"Unsupported zoom level: {zoom_arg}. Supported: {supported}")
        return zoom_to_gsd[zoom_arg]

    # Priority 3: Default fallback for SpaceNet dataset
    logger.info("No --gsd or --zoom provided. Using default GSD: 0.3 m/px (SpaceNet)")
    return 0.3


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Rooftop Segmentation Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py --image_path sample.tif
  python predict.py --image_path sample.tif --gsd 0.3
  python predict.py --image_path sample.tif --threshold 0.4 --min_area 100 --top_k 20
  python predict.py --image_path sample.tif --smooth --threshold 0.35
        """,
    )

    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to input satellite image",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="runs/solarsense/best_model.pth",
        help="Path to trained model checkpoint (default: runs/solarsense/best_model.pth)",
    )

    parser.add_argument(
        "--gsd",
        type=float,
        default=None,
        help="Ground Sampling Distance (meters per pixel). If not provided, use zoom-based default.",
    )

    parser.add_argument(
        "--zoom",
        type=int,
        default=None,
        help="Zoom level if using Google Maps images (e.g., 18 or 19)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory for results (default: outputs)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Auto-detected if not specified.",
    )

    # Post-processing arguments
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Confidence threshold for binary mask (default: 0.4)",
    )

    parser.add_argument(
        "--min_area",
        type=int,
        default=100,
        help="Minimum component area in pixels (default: 100)",
    )

    parser.add_argument(
        "--kernel_size",
        type=int,
        default=3,
        help="Morphological kernel size (default: 3)",
    )

    parser.add_argument(
        "--smooth",
        action="store_true",
        help="Apply Gaussian smoothing before thresholding",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Keep only top-K largest components (default: keep all that pass min_area)",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        # Resolve GSD value from arguments
        gsd = resolve_gsd(args.gsd, args.zoom)
        logger.info(f"Using GSD: {gsd} meters/pixel")
        logger.info(f"Pixel area: {gsd * gsd:.4f} m²")
        # Load model
        model = load_model(args.model_path, device)

        # Preprocess image
        image_tensor, original = preprocess(args.image_path, device)

        # Run prediction (get probabilities)
        probs_tensor = predict(model, image_tensor)

        # Convert to numpy for post-processing
        probs_np = probs_tensor.cpu().numpy()

        # Step 1: Optional Gaussian smoothing
        if args.smooth:
            logger.info("Applying Gaussian smoothing...")
            probs_np = apply_gaussian_smoothing(probs_np, kernel_size=(5, 5), sigma=0)

        # Step 2: Apply threshold
        logger.info(f"Applying threshold: {args.threshold}")
        binary_mask = apply_threshold(probs_np, args.threshold)

        # Calculate roof pixel percentage
        roof_pixels = np.sum(binary_mask)
        total_pixels = binary_mask.size
        roof_pixels_pct = (roof_pixels / total_pixels) * 100
        logger.info(f"  Pixels predicted as roof: {roof_pixels_pct:.2f}%")

        # Step 3: Morphological cleaning (open/close)
        logger.info(f"Applying morphological cleaning (kernel_size={args.kernel_size})...")
        binary_mask = apply_morphological_cleaning(binary_mask, kernel_size=args.kernel_size)

        # Step 4: Remove small components (with optional top-K filtering)
        logger.info(f"Removing small components (min_area={args.min_area}, top_k={args.top_k})...")
        binary_mask, components_before, components_after = remove_small_components(
            binary_mask,
            min_area=args.min_area,
            top_k=args.top_k,
        )
        logger.info(f"  Components: {components_before} -> {components_after} (removed {components_before - components_after})")

        # Calculate area
        logger.info("Calculating area metrics...")
        area_info = pixels_to_area(binary_mask, gsd=gsd)

        # Save outputs
        logger.info(f"Saving outputs to {args.output_dir}/")
        save_outputs(original, binary_mask, args.output_dir)

        # Save debug visualization
        save_debug_outputs(original, probs_np, binary_mask, args.output_dir)

        # Print results with post-processing stats
        print_results(
            area_info,
            args.threshold,
            roof_pixels_pct,
            components_before,
            components_after,
        )

        logger.info("Prediction complete!")
        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    sys.exit(main())
