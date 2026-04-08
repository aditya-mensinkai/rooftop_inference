# 🏠 Rooftop Detection & Area Estimation – Inference Guide

## 📌 Overview

This module performs:

* ✅ Rooftop segmentation from satellite images
* ✅ Area estimation (in m²)

> ⚠️ Note: Solar estimation is handled by a **separate module/service**

---

## 🧠 Pipeline

```text
Input Image (.tif)
        ↓
Segmentation Model (U-Net)
        ↓
Binary Mask
        ↓
Area Calculation (using GSD)
        ↓
Output (JSON + images)
```

---

## 📦 Project Structure

```
rooftop_inference/
│
├── predict.py              # Main inference script
├── model.py                # Model architecture
├── area_utils.py           # Area calculation logic
├── metrics.py              # (optional / can be ignored)
├── requirements.txt        # Dependencies
│
├── runs/
│   └── solarsense/
│       └── best_model.pth  # Trained model (REQUIRED)
│
├── sample_imgs/
│   └── img1.tif            # Sample test image
│
└── outputs/                # Generated outputs
```

---

## ⚙️ Installation

### 1. Create virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage (CLI)

```bash
python predict.py --image_path path/to/image.tif --gsd 0.3
```

---

## 📥 Input Requirements

* Format: `.tif` (GeoTIFF or standard TIFF)
* Channels: RGB (3-channel)
* Resolution: Any (auto-resized internally)

---

## 📏 GSD (Ground Sampling Distance)

### What is GSD?

* Defines real-world size per pixel
* Used for accurate area calculation

### How to use:

#### Option 1 (Recommended for this project):

```bash
--gsd 0.3
```

#### Option 2 (Google Maps):

| Zoom | GSD  |
| ---- | ---- |
| 19   | 0.15 |
| 18   | 0.3  |

---

## 📤 Output

### Console Output

```
===== PREDICTION RESULTS =====

Roof pixels:     XXXX
Total area:      XXXX m²
Usable area:     XXXX m²
```

---

### Output Files

| File                     | Description           |
| ------------------------ | --------------------- |
| `outputs/mask.png`       | Binary rooftop mask   |
| `outputs/viz.png`        | Overlay visualization |
| `outputs/debug_prob.png` | Probability heatmap   |

---

### JSON Output (if using function)

```json
{
  "roof_pixels": 10459,
  "total_area_m2": 941.0,
  "usable_area_m2": 705.7,
  "mask_path": "outputs/mask.png",
  "viz_path": "outputs/viz.png"
}
```

---

## 🔌 Backend Integration

### Option 1 — Import as Function

Add this in `predict.py`:

```python
def run_inference(image_path, gsd=0.3):
    # existing pipeline
    return {
        "roof_pixels": roof_pixels,
        "total_area_m2": total_area,
        "usable_area_m2": usable_area,
        "mask_path": mask_path,
        "viz_path": viz_path
    }
```

---

### Usage in Backend (FastAPI / Flask)

```python
from predict import run_inference

result = run_inference("input.tif", gsd=0.3)
```

---

### Option 2 — API Wrapper (Recommended)

```python
@app.post("/predict")
def predict(file: UploadFile):
    path = save_file(file)
    result = run_inference(path, gsd=0.3)
    return result
```

---

## ⚠️ Important Notes

### 1. Always use correct GSD

* SpaceNet dataset → `0.3`
* Google Maps → depends on zoom

---

### 2. Area Accuracy Depends On:

* Correct GSD ✅
* Good segmentation ✅
* Proper preprocessing ✅

---

### 3. Model Limitations

* May miss small rooftops
* May merge nearby buildings
* Accuracy ~80–85%

---

## 🚫 What is NOT included

* ❌ Training code
* ❌ Dataset
* ❌ Solar estimation (handled separately)

---

## 🚀 Next Steps (Optional)

* Integrate with Solar Estimation API
* Add batch inference support
* Deploy as microservice (FastAPI + Docker)

---

## 👨‍💻 Contact

For issues or improvements, contact [aditya.men2005@gmail.com](mailto:aditya.men2005@gmail.com).
