"""
ARGO Float Dataset Generator
Generates a realistic synthetic NetCDF dataset (argo_data.cdf) for testing FloatChat AI.

Variables included:
  TEMP         - Sea water temperature (°C)
  SAL          - Practical salinity (PSU)
  OXYGEN       - Dissolved oxygen (ml/L)
  CHLOROPHYLL  - Chlorophyll-a concentration (mg/m³)
  PH           - pH (total scale)
  NITRATE      - Nitrate (µmol/kg)

Dimensions:
  TAXIS  - time  (180 daily steps, Jan–Jun 2023)
  ZAX    - depth (12 standard levels, 10–2000 m)
  YAXIS  - latitude  (Indian Ocean band: -40° to 30°)
  XAXIS  - longitude (Indian Ocean band:  40° to 120°)

Run:
  python generate_argo_dataset.py
"""

import numpy as np
import pandas as pd
import xarray as xr

OUTPUT_FILE = "argo_data.cdf"

# ── Axes ─────────────────────────────────────────────────────────────────────
TIME  = pd.date_range("2023-01-01", periods=180, freq="D")
DEPTH = np.array([10.0, 25.0, 50.0, 100.0, 200.0, 300.0,
                  500.0, 750.0, 1000.0, 1500.0, 1750.0, 2000.0])
LAT   = np.linspace(-40.0, 30.0, 70)      # 1° spacing approx
LON   = np.linspace(40.0, 120.0, 80)      # 1° spacing approx

NT, NZ, NY, NX = len(TIME), len(DEPTH), len(LAT), len(LON)

rng = np.random.default_rng(42)


def _thermocline_temp(depth_arr, lat_arr):
    """
    Realistic temperature field:
      - Warm surface (~28 °C near equator, ~22 °C at ±30°)
      - Strong thermocline 50–300 m
      - Cold deep water (~2–4 °C below 1000 m)
    """
    # Surface temperature: warmer near equator
    t_surface = 28.0 - 0.15 * np.abs(lat_arr)          # shape (NY,)

    # Thermocline decay with depth
    scale = np.where(depth_arr < 500, 200.0, 600.0)     # shape (NZ,)
    t_deep = 2.5                                         # asymptote

    # Broadcast: (NZ, NY)
    temp_2d = t_deep + (t_surface[np.newaxis, :] - t_deep) * np.exp(
        -depth_arr[:, np.newaxis] / scale[:, np.newaxis]
    )
    return temp_2d                                       # (NZ, NY)


def _halocline_sal(depth_arr, lat_arr):
    """
    Realistic salinity:
      - Slight surface freshening near equator (ITCZ rain)
      - Salinity maximum ~100–200 m (subtropical cells)
      - Relatively fresh in deep water
    """
    s_surface = 34.8 + 0.05 * np.abs(lat_arr)           # (NY,)
    s_max     = 35.5 + 0.03 * np.abs(lat_arr)           # (NY,)

    depth_norm = depth_arr / 150.0                       # (NZ,)
    # Gaussian peak at ~150 m
    sal_2d = s_max[np.newaxis, :] * np.exp(
        -0.5 * (depth_norm[:, np.newaxis] - 1.0) ** 2 / 0.8
    )
    # Blend surface + subsurface max
    alpha = np.exp(-depth_arr[:, np.newaxis] / 30.0)    # (NZ, NY)
    sal_2d = alpha * s_surface[np.newaxis, :] + (1 - alpha) * sal_2d
    # Deep water 34.65 asymptote
    beta = np.exp(-depth_arr[:, np.newaxis] / 800.0)
    sal_2d = beta * sal_2d + (1 - beta) * 34.65

    return sal_2d                                        # (NZ, NY)


def _oxygen(depth_arr):
    """
    Dissolved oxygen (ml/L):
      - High at surface (~6.5)
      - Oxygen minimum zone 200–800 m (~1.5–3.0)
      - Slight recovery in deep water (~4.5)
    """
    o2 = np.where(
        depth_arr < 50,   6.5 - depth_arr / 80.0,
        np.where(
            depth_arr < 800, 3.5 - (depth_arr - 50) / 800.0,
            4.2 + (depth_arr - 800) / 4000.0
        )
    )
    return np.clip(o2, 0.5, 7.5)                        # (NZ,)


def _chlorophyll(depth_arr):
    """
    Chlorophyll-a (mg/m³):
      - Deep chlorophyll maximum ~50–80 m
      - Near zero below 200 m
    """
    chl = 1.2 * np.exp(-0.5 * ((depth_arr - 60) / 35) ** 2)
    chl = np.where(depth_arr > 200, chl * np.exp(-(depth_arr - 200) / 80), chl)
    return np.clip(chl, 0.0, None)                      # (NZ,)


def _ph(depth_arr):
    """
    pH (total scale):
      - Surface ~8.10–8.12
      - Decreases to ~7.85 at depth (more CO2)
    """
    return 8.12 - 0.00013 * depth_arr                   # (NZ,)


def _nitrate(depth_arr):
    """
    Nitrate (µmol/kg):
      - Near zero at surface (consumed by phytoplankton)
      - Increases rapidly below the euphotic zone
    """
    return np.clip(0.5 + depth_arr / 60.0, 0.0, 40.0)  # (NZ,)


def generate():
    print(f"Generating ARGO dataset — {NT} days x {NZ} depths x {NY} lats x {NX} lons")
    print(f"Total grid cells: {NT * NZ * NY * NX:,}")

    # ── Base 2-D fields (NZ x NY) ──────────────────────────────────────────
    temp_base = _thermocline_temp(DEPTH, LAT)      # (NZ, NY)
    sal_base  = _halocline_sal(DEPTH, LAT)         # (NZ, NY)
    oxy_base  = _oxygen(DEPTH)                     # (NZ,)
    chl_base  = _chlorophyll(DEPTH)                # (NZ,)
    ph_base   = _ph(DEPTH)                         # (NZ,)
    nit_base  = _nitrate(DEPTH)                    # (NZ,)

    # ── Broadcast to full 4-D (NT x NZ x NY x NX) ─────────────────────────
    shape = (NT, NZ, NY, NX)

    # Seasonal cycle: ±1 °C, period 365 days
    day_of_year = np.array([(t - pd.Timestamp("2023-01-01")).days for t in TIME])
    seasonal = np.sin(2 * np.pi * day_of_year / 365.0)  # (NT,)

    TEMP = (
        temp_base[np.newaxis, :, :, np.newaxis]          # (1,NZ,NY,1)
        + 0.8 * seasonal[:, np.newaxis, np.newaxis, np.newaxis]  # seasonal
        + rng.normal(0, 0.25, shape)                      # random noise
    )

    SAL = (
        sal_base[np.newaxis, :, :, np.newaxis]
        + 0.05 * seasonal[:, np.newaxis, np.newaxis, np.newaxis]
        + rng.normal(0, 0.04, shape)
    )

    OXYGEN = (
        oxy_base[np.newaxis, :, np.newaxis, np.newaxis]
        + rng.normal(0, 0.12, shape)
    )

    CHLOROPHYLL = (
        chl_base[np.newaxis, :, np.newaxis, np.newaxis]
        * (1 + 0.4 * seasonal[:, np.newaxis, np.newaxis, np.newaxis])
        + rng.exponential(0.03, shape)
    )

    PH = (
        ph_base[np.newaxis, :, np.newaxis, np.newaxis]
        + rng.normal(0, 0.008, shape)
    )

    NITRATE = (
        nit_base[np.newaxis, :, np.newaxis, np.newaxis]
        + rng.normal(0, 0.8, shape)
    )

    # Clip to physically plausible ranges
    TEMP        = np.clip(TEMP,        -2.0,  35.0)
    SAL         = np.clip(SAL,         30.0,  40.0)
    OXYGEN      = np.clip(OXYGEN,       0.0,   8.5)
    CHLOROPHYLL = np.clip(CHLOROPHYLL,  0.0,   5.0)
    PH          = np.clip(PH,           7.6,   8.3)
    NITRATE     = np.clip(NITRATE,      0.0,  45.0)

    # ── Build xarray Dataset ───────────────────────────────────────────────
    dims = ["TAXIS", "ZAX", "YAXIS", "XAXIS"]

    ds = xr.Dataset(
        {
            "TEMP": (dims, TEMP.astype("float32"), {
                "long_name": "Sea water temperature",
                "units": "degree_Celsius",
                "valid_min": -2.0,
                "valid_max": 35.0,
            }),
            "SAL": (dims, SAL.astype("float32"), {
                "long_name": "Practical salinity",
                "units": "PSU",
                "valid_min": 30.0,
                "valid_max": 40.0,
            }),
            "OXYGEN": (dims, OXYGEN.astype("float32"), {
                "long_name": "Dissolved oxygen",
                "units": "ml/L",
                "valid_min": 0.0,
                "valid_max": 8.5,
            }),
            "CHLOROPHYLL": (dims, CHLOROPHYLL.astype("float32"), {
                "long_name": "Chlorophyll-a concentration",
                "units": "mg/m3",
                "valid_min": 0.0,
                "valid_max": 5.0,
            }),
            "PH": (dims, PH.astype("float32"), {
                "long_name": "pH (total scale)",
                "units": "1",
                "valid_min": 7.6,
                "valid_max": 8.3,
            }),
            "NITRATE": (dims, NITRATE.astype("float32"), {
                "long_name": "Nitrate concentration",
                "units": "umol/kg",
                "valid_min": 0.0,
                "valid_max": 45.0,
            }),
        },
        coords={
            "TAXIS": TIME,
            "ZAX":   DEPTH,
            "YAXIS": LAT,
            "XAXIS": LON,
        },
        attrs={
            "title":       "FloatChat AI — Synthetic ARGO Float Dataset",
            "institution": "FloatChat AI Project",
            "source":      "Synthetic data generated for testing",
            "history":     f"Created by generate_argo_dataset.py",
            "region":      "Indian Ocean (40-120E, 40S-30N)",
            "time_coverage_start": "2023-01-01",
            "time_coverage_end":   "2023-06-29",
            "n_time_steps": NT,
            "n_depth_levels": NZ,
            "depth_range": f"{DEPTH.min():.0f}–{DEPTH.max():.0f} m",
            "Conventions": "CF-1.8",
        },
    )

    # ── Write ──────────────────────────────────────────────────────────────
    encoding = {
        var: {"zlib": True, "complevel": 4}
        for var in ["TEMP", "SAL", "OXYGEN", "CHLOROPHYLL", "PH", "NITRATE"]
    }
    ds.to_netcdf(OUTPUT_FILE, encoding=encoding)
    print(f"\nDataset saved to: {OUTPUT_FILE}")
    print(f"  Time steps  : {NT}  ({TIME[0].date()} to {TIME[-1].date()})")
    print(f"  Depth levels: {NZ}  ({DEPTH.min():.0f} – {DEPTH.max():.0f} m)")
    print(f"  Latitude    : {NY}  ({LAT.min():.1f}° to {LAT.max():.1f}°)")
    print(f"  Longitude   : {NX}  ({LON.min():.1f}° to {LON.max():.1f}°)")
    print(f"  Variables   : TEMP, SAL, OXYGEN, CHLOROPHYLL, PH, NITRATE")

    import os
    size_mb = os.path.getsize(OUTPUT_FILE) / (1024 ** 2)
    print(f"  File size   : {size_mb:.1f} MB")
    print("\nNext steps:")
    print("  1. python argo_float_processor.py   <- loads .cdf into PostgreSQL")
    print("  2. python data_chroma_floats.py     <- indexes into ChromaDB")
    print("  3. uvicorn main:app --reload        <- start backend")
    print("  4. streamlit run streamlit_app.py   <- start frontend")


if __name__ == "__main__":
    generate()
