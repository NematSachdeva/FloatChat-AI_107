"""Ingest real Argo GDAC data (referenced by Seanoe DOI 10.17882/42182) into PostgreSQL."""

import csv
import glob
import io
import os
import random
import sys
import tempfile
from datetime import date
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import requests
import xarray as xr
from sqlalchemy import create_engine, text

import config
from pipeline.argo_float_processor import clear_existing_data, create_argo_tables


engine = create_engine(config.DATABASE_URL)


def _decode_platform_number(raw_value) -> str:
    if raw_value is None:
        return "UNKNOWN"

    if hasattr(raw_value, "values"):
        raw_value = raw_value.values

    if hasattr(raw_value, "dtype") and str(raw_value.dtype).startswith("|S"):
        try:
            raw_value = b"".join(raw_value.tolist()).decode("utf-8", errors="ignore")
        except Exception:  # noqa: BLE001
            raw_value = str(raw_value)

    value = str(raw_value).strip()
    value = value.replace("'", "").replace("b", "")
    return value or "UNKNOWN"


def _safe_float(value):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:  # noqa: BLE001
        return None


def _safe_datetime(value):
    try:
        if pd.isna(value):
            return None
        return pd.to_datetime(value).to_pydatetime()
    except Exception:  # noqa: BLE001
        return None


def fetch_profile_index(max_profiles: int) -> List[str]:
    index_url = f"{config.ARGO_GDAC_HTTP_BASE.rstrip('/')}/{config.ARGO_INDEX_PATH.lstrip('/')}"
    print(f"Fetching profile index from: {index_url}")

    response = requests.get(index_url, timeout=config.ARGO_PROFILE_TIMEOUT_SECONDS)
    response.raise_for_status()

    content = response.text
    records: List[str] = []

    for row in csv.reader(io.StringIO(content)):
        if not row:
            continue
        if row[0].startswith("#"):
            continue
        # argo synthetic profile index: first column is relative file path
        path = row[0].strip()
        if path.endswith(".nc"):
            records.append(path)

    if not records:
        raise RuntimeError("No NetCDF profile paths found in Argo index")

    if max_profiles and max_profiles > 0:
        random.seed(42)
        if len(records) > max_profiles:
            records = random.sample(records, max_profiles)

    print(f"Selected {len(records)} profile files from index")
    return records


def fetch_local_snapshot_paths(snapshot_dir: str, max_profiles: int) -> List[str]:
    """Collect local *_Sprof.nc files from a downloaded Argo snapshot folder."""
    pattern = os.path.join(snapshot_dir, config.ARGO_LOCAL_GLOB)
    files = sorted(glob.glob(pattern, recursive=True))

    if not files:
        raise RuntimeError(f"No local files found with pattern: {pattern}")

    if max_profiles and max_profiles > 0:
        random.seed(42)
        if len(files) > max_profiles:
            files = random.sample(files, max_profiles)

    print(f"Selected {len(files)} local profile files from snapshot")
    return files


def _extract_profile_rows(ds: xr.Dataset, source_path: str, start_profile_id: int) -> Tuple[List[Dict], List[Dict], List[Dict], int]:
    floats: Dict[str, Dict] = {}
    profiles: List[Dict] = []
    measurements: List[Dict] = []

    if "N_PROF" not in ds.dims:
        return [], [], [], start_profile_id

    n_prof = int(ds.dims["N_PROF"])

    platform_var = ds.get("PLATFORM_NUMBER")
    cycle_var = ds.get("CYCLE_NUMBER")
    juld_var = ds.get("JULD")
    lat_var = ds.get("LATITUDE")
    lon_var = ds.get("LONGITUDE")

    pres_var = ds.get("PRES")
    temp_var = ds.get("TEMP")
    psal_var = ds.get("PSAL")
    doxy_var = ds.get("DOXY")
    ph_var = ds.get("PH_IN_SITU_TOTAL")
    if ph_var is None:
        ph_var = ds.get("PH_IN_SITU_TOTAL_ADJUSTED")

    chl_var = ds.get("CHLA")
    if chl_var is None:
        chl_var = ds.get("CHLA_ADJUSTED")

    nitrate_var = ds.get("NITRATE")
    if nitrate_var is None:
        nitrate_var = ds.get("NITRATE_ADJUSTED")

    profile_id = start_profile_id

    for i in range(n_prof):
        platform_raw = platform_var.isel(N_PROF=i) if platform_var is not None else None
        platform_number = _decode_platform_number(platform_raw)
        float_id = f"ARGO_{platform_number}"

        cycle_number = int(cycle_var.isel(N_PROF=i).item()) if cycle_var is not None else i + 1
        profile_time = _safe_datetime(juld_var.isel(N_PROF=i).item()) if juld_var is not None else None
        lat = _safe_float(lat_var.isel(N_PROF=i).item()) if lat_var is not None else None
        lon = _safe_float(lon_var.isel(N_PROF=i).item()) if lon_var is not None else None

        if float_id not in floats:
            wmo_digits = "".join(ch for ch in platform_number if ch.isdigit())
            wmo_id = int(wmo_digits) if wmo_digits else None
            floats[float_id] = {
                "float_id": float_id,
                "wmo_id": wmo_id,
                "deployment_date": profile_time.date() if profile_time else date.today(),
                "deployment_lat": lat,
                "deployment_lon": lon,
                "status": "ACTIVE",
                "last_contact": profile_time.date() if profile_time else date.today(),
            }

        n_levels = 0
        if pres_var is not None and "N_LEVELS" in pres_var.dims:
            profile_pres = pres_var.isel(N_PROF=i).values
            n_levels = int(pd.Series(profile_pres).notna().sum())

        profiles.append(
            {
                "profile_id": profile_id,
                "float_id": float_id,
                "cycle_number": cycle_number,
                "profile_date": profile_time,
                "profile_lat": lat,
                "profile_lon": lon,
                "n_levels": n_levels,
            }
        )

        if pres_var is None or temp_var is None or psal_var is None:
            profile_id += 1
            continue

        pres_values = pres_var.isel(N_PROF=i).values
        temp_values = temp_var.isel(N_PROF=i).values
        sal_values = psal_var.isel(N_PROF=i).values
        oxy_values = doxy_var.isel(N_PROF=i).values if doxy_var is not None else None
        ph_values = ph_var.isel(N_PROF=i).values if ph_var is not None else None
        chl_values = chl_var.isel(N_PROF=i).values if chl_var is not None else None
        nitrate_values = nitrate_var.isel(N_PROF=i).values if nitrate_var is not None else None

        for level_idx in range(len(pres_values)):
            pressure = _safe_float(pres_values[level_idx])
            temperature = _safe_float(temp_values[level_idx])
            salinity = _safe_float(sal_values[level_idx])
            if pressure is None or temperature is None or salinity is None:
                continue

            measurements.append(
                {
                    "profile_id": profile_id,
                    "float_id": float_id,
                    "time": profile_time,
                    "lat": lat,
                    "lon": lon,
                    "depth": pressure,
                    "pressure": pressure,
                    "temperature": temperature,
                    "salinity": salinity,
                    "oxygen": _safe_float(oxy_values[level_idx]) if oxy_values is not None else None,
                    "ph": _safe_float(ph_values[level_idx]) if ph_values is not None else None,
                    "chlorophyll": _safe_float(chl_values[level_idx]) if chl_values is not None else None,
                    "nitrate": _safe_float(nitrate_values[level_idx]) if nitrate_values is not None else None,
                    "backscatter": None,
                    "cdom": None,
                    "downwelling_par": None,
                }
            )

        profile_id += 1

    return list(floats.values()), profiles, measurements, profile_id


def ingest_from_seanoe(max_profiles: int = None):
    max_profiles = max_profiles or config.ARGO_MAX_PROFILES

    print("Using Seanoe source:", config.ARGO_DATA_SOURCE_URL)
    print("Clearing existing ARGO data and rebuilding schema")
    clear_existing_data()
    create_argo_tables()

    profile_paths = fetch_profile_index(max_profiles=max_profiles)

    _ingest_profile_paths(profile_paths=profile_paths, source_mode="remote")


def ingest_from_local_snapshot(snapshot_dir: str, max_profiles: int = None):
    """Ingest data from a locally downloaded Argo snapshot directory."""
    max_profiles = max_profiles or config.ARGO_MAX_PROFILES

    print(f"Using local snapshot folder: {snapshot_dir}")
    print("Clearing existing ARGO data and rebuilding schema")
    clear_existing_data()
    create_argo_tables()

    profile_paths = fetch_local_snapshot_paths(snapshot_dir=snapshot_dir, max_profiles=max_profiles)

    _ingest_profile_paths(profile_paths=profile_paths, source_mode="local")


def _ingest_profile_paths(profile_paths: List[str], source_mode: str):
    if source_mode not in {"remote", "local"}:
        raise ValueError("source_mode must be 'remote' or 'local'")

    all_floats: Dict[str, Dict] = {}
    all_profiles: List[Dict] = []
    all_measurements: List[Dict] = []

    profile_id = 1
    processed = 0

    for path_value in profile_paths:
        file_url = path_value
        if source_mode == "remote":
            file_url = f"{config.ARGO_GDAC_HTTP_BASE.rstrip('/')}/{path_value.lstrip('/')}"

        tmp_path = None
        try:
            if source_mode == "remote":
                response = requests.get(file_url, timeout=config.ARGO_PROFILE_TIMEOUT_SECONDS)
                response.raise_for_status()

                with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
                    tmp.write(response.content)
                    tmp_path = tmp.name

                open_path = tmp_path
            else:
                open_path = path_value

            ds = xr.open_dataset(open_path, decode_times=True)
            floats, profiles, measurements, profile_id = _extract_profile_rows(ds, path_value, profile_id)
            ds.close()

            for item in floats:
                all_floats[item["float_id"]] = item
            all_profiles.extend(profiles)
            all_measurements.extend(measurements)

            processed += 1
            if processed % 20 == 0:
                print(f"Processed {processed}/{len(profile_paths)} profile files")

        except Exception as exc:  # noqa: BLE001
            print(f"Skipping {file_url}: {exc}")
            continue
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    if not all_floats or not all_profiles or not all_measurements:
        raise RuntimeError("No valid Argo records were extracted from selected files")

    floats_df = pd.DataFrame(all_floats.values())
    profiles_df = pd.DataFrame(all_profiles)
    measurements_df = pd.DataFrame(all_measurements)

    print(f"Inserting {len(floats_df)} floats, {len(profiles_df)} profiles, {len(measurements_df)} measurements")

    floats_df.to_sql("floats", engine, if_exists="append", index=False, method="multi")
    profiles_df.to_sql("profiles", engine, if_exists="append", index=False, method="multi")

    batch_size = 5000
    for i in range(0, len(measurements_df), batch_size):
        batch = measurements_df.iloc[i : i + batch_size]
        batch.to_sql("measurements", engine, if_exists="append", index=False, method="multi")

    with engine.connect() as conn:
        summary = conn.execute(
            text(
                """
                SELECT
                    (SELECT COUNT(*) FROM floats) AS floats,
                    (SELECT COUNT(*) FROM profiles) AS profiles,
                    (SELECT COUNT(*) FROM measurements) AS measurements
                """
            )
        ).mappings().first()

    print("Ingestion complete:", dict(summary))


if __name__ == "__main__":
    if config.ARGO_LOCAL_SNAPSHOT_DIR:
        ingest_from_local_snapshot(snapshot_dir=config.ARGO_LOCAL_SNAPSHOT_DIR)
    else:
        ingest_from_seanoe()
