"""
area_utils.py — Production Solar Area Estimation Utilities
SolarSense Platform | IEEE YESIST12 WePOWER Track 2026

Converts binary rooftop segmentation masks into solar generation estimates
using state-wise GHI data from NASA POWER and MNRE benchmark costs (2024).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# State-wise average GHI (kWh / m² / day) — NASA POWER annual averages
# ---------------------------------------------------------------------------

STATE_GHI: dict[str, float] = {
    "Tamil Nadu":      5.5,
    "Rajasthan":       6.2,
    "Maharashtra":     5.4,
    "Karnataka":       5.3,
    "Gujarat":         5.8,
    "Uttar Pradesh":   4.9,
    "Madhya Pradesh":  5.4,
    "Telangana":       5.6,
    "Andhra Pradesh":  5.5,
    "Punjab":          4.8,
    "Haryana":         4.9,
    "West Bengal":     4.7,
    "Bihar":           4.8,
    "Odisha":          5.1,
    "Kerala":          4.9,
}

# MNRE benchmark cost per kWp (INR) — 2024 rates
MNRE_COST_PER_KWP: dict[str, int] = {
    "residential_1_3kw":      65_000,
    "residential_3_10kw":     55_000,
    "residential_above_10kw": 45_000,
}

# CO₂ emission factor — India grid average (kg CO₂ / kWh), CEA 2023
INDIA_GRID_CO2_KG_PER_KWH: float = 0.716

# m² of roof area required per kWp of solar capacity
M2_PER_KWP: float = 6.5


# ---------------------------------------------------------------------------
# Core area conversion
# ---------------------------------------------------------------------------

def pixels_to_area(
    mask: np.ndarray,
    gsd: float = 0.5,
    usable_factor: float = 0.75,
) -> dict[str, float]:
    """
    Convert a binary rooftop mask to solar-relevant area estimates.

    Args:
        mask          : Binary numpy array (H, W), values 0 or 1.
        gsd           : Ground Sampling Distance in metres per pixel.
                        - Google Maps zoom 19  : ~0.15 m/px
                        - Sentinel-2           : ~10.0 m/px
                        - Default satellite    :  ~0.5 m/px
        usable_factor : Fraction of roof usable for solar panels
                        (accounting for vents, edges, shading).
                        Typical range: 0.70 – 0.80.

    Returns:
        Dict with keys:
            roof_pixels          – number of rooftop pixels
            pixel_area_m2        – area of a single pixel in m²
            total_roof_area_m2   – total detected roof area in m²
            usable_area_m2       – area available for solar panels
            estimated_kw_capacity – system capacity in kWp
            estimated_monthly_kwh – monthly energy (4.5 sun-hrs, 20% eff.)
            estimated_annual_kwh  – annual energy generation estimate
    """
    mask = (mask > 0.5).astype(np.int32)
    roof_pixels     = int(mask.sum())
    pixel_area_m2   = gsd ** 2
    total_area_m2   = roof_pixels * pixel_area_m2
    usable_area_m2  = total_area_m2 * usable_factor
    kw_capacity     = usable_area_m2 / M2_PER_KWP

    # Simplified generation estimate (generic, sensor-agnostic):
    # Uses 4.5 peak sun hours + 20% system efficiency as conservative baseline.
    # For state-specific GHI-accurate estimates, use area_to_solar_metrics().
    daily_kwh       = kw_capacity * 4.5 * 0.20       # 4.5 peak sun hours, 20% system eff.
    monthly_kwh     = daily_kwh * 30
    annual_kwh      = daily_kwh * 365

    return {
        "roof_pixels":           roof_pixels,
        "pixel_area_m2":         round(pixel_area_m2, 6),
        "total_roof_area_m2":    round(total_area_m2, 2),
        "usable_area_m2":        round(usable_area_m2, 2),
        "estimated_kw_capacity": round(kw_capacity, 3),
        "estimated_monthly_kwh": round(monthly_kwh, 2),
        "estimated_annual_kwh":  round(annual_kwh, 2),
    }


# ---------------------------------------------------------------------------
# Full solar metrics from usable area
# ---------------------------------------------------------------------------

def area_to_solar_metrics(
    area_m2: float,
    state: str = "Tamil Nadu",
    panel_efficiency: float = 0.20,
    performance_ratio: float = 0.80,
) -> dict[str, float | list[float]]:
    """
    Convert usable roof area to full solar generation estimates.

    Args:
        area_m2           : Usable roof area in m².
        state             : Indian state name (must be in STATE_GHI).
        panel_efficiency  : Panel conversion efficiency (default 20%).
        performance_ratio : System performance ratio — accounts for wiring,
                            inverter, soiling losses (default 0.80).

    Returns:
        Dict with keys:
            system_kw_capacity        – installed capacity in kWp
            annual_kwh                – estimated annual generation (kWh)
            monthly_kwh_by_month      – list of 12 monthly estimates (kWh)
            estimated_system_cost_inr – MNRE benchmark cost in INR
            co2_offset_kg_per_year    – annual CO₂ offset in kg
    """
    ghi = STATE_GHI.get(state, 5.0)   # default fallback GHI

    system_kw = area_m2 / M2_PER_KWP

    # Annual generation: P_peak × GHI_annual × PR
    # GHI annual (kWh/m²/yr) ≈ daily GHI × 365
    ghi_annual  = ghi * 365                              # kWh/m²/yr
    annual_kwh  = system_kw * ghi_annual * performance_ratio

    # Monthly distribution using typical Indian solar seasonality factors
    # (normalised so they sum to 12.0, i.e. monthly averages relative to mean)
    monthly_factors = [0.90, 0.95, 1.05, 1.10, 1.08, 0.95,
                       0.88, 0.90, 0.97, 1.05, 1.02, 1.05]
    mean_monthly    = annual_kwh / 12.0
    monthly_kwh     = [round(mean_monthly * f, 1) for f in monthly_factors]

    # Cost estimation
    if system_kw <= 3:
        cost_per_kwp = MNRE_COST_PER_KWP["residential_1_3kw"]
    elif system_kw <= 10:
        cost_per_kwp = MNRE_COST_PER_KWP["residential_3_10kw"]
    else:
        cost_per_kwp = MNRE_COST_PER_KWP["residential_above_10kw"]

    system_cost_inr  = system_kw * cost_per_kwp
    co2_offset_kg    = annual_kwh * INDIA_GRID_CO2_KG_PER_KWH

    return {
        "system_kw_capacity":        round(system_kw, 3),
        "annual_kwh":                round(annual_kwh, 1),
        "monthly_kwh_by_month":      monthly_kwh,
        "estimated_system_cost_inr": round(system_cost_inr, 0),
        "co2_offset_kg_per_year":    round(co2_offset_kg, 1),
    }


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Create a synthetic mask with ~30% rooftop coverage
    mask = np.zeros((256, 256), dtype=np.float32)
    mask[60:180, 60:180] = 1.0   # 120×120 pixel square roof

    area_info = pixels_to_area(mask, gsd=0.5, usable_factor=0.75)
    print("pixels_to_area output:")
    for k, v in area_info.items():
        print(f"  {k:<30} : {v}")

    solar = area_to_solar_metrics(area_info["usable_area_m2"], state="Tamil Nadu")
    print("\narea_to_solar_metrics output:")
    for k, v in solar.items():
        if k == "monthly_kwh_by_month":
            print(f"  {k:<30} : {v}")
        else:
            print(f"  {k:<30} : {v}")

    assert area_info["total_roof_area_m2"] > 0, "Area must be positive"
    assert area_info["estimated_kw_capacity"] > 0, "Capacity must be positive"
    print("\narea_utils tests passed ✓")
