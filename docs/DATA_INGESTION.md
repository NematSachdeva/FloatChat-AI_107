# Data Ingestion Guide for FloatChat AI

## Overview

This guide explains how to ingest ARGO float data into FloatChat AI for semantic search and analysis.

## Data Ingestion Process

### Step 1: Prepare Your Data

FloatChat AI supports multiple data formats:

- **NetCDF Files** (.nc) - Standard oceanographic format
- **CSV Files** - Tabular data with proper column mapping
- **PostgreSQL Direct** - Import from existing databases

### Step 2: Required Data Fields

Your data must include these core fields:

```
Core Fields:
- float_id: Unique float identifier
- profile_id: Profile cycle number
- time: Measurement timestamp (ISO 8601 format)
- latitude: Geographic latitude (-90 to 90)
- longitude: Geographic longitude (-180 to 180)
- depth: Measurement depth in meters

Physical Parameters:
- temperature: Water temperature in °C
- salinity: Practical salinity in PSU (Practical Salinity Units)
- pressure: Water pressure in dbar (optional)

Biogeochemical Parameters (Optional):
- oxygen: Dissolved oxygen in μmol/kg
- ph: pH level
- chlorophyll: Chlorophyll-a concentration in mg/m³
- nitrate: Nitrate concentration in μmol/kg
- backscatter: Optical backscatter
```

### Step 3: Data Ingestion Methods

#### Method 1: Using ARGO Float Processor (Recommended)

```bash
python argo_float_processor.py
```

#### Method 2: Direct PostgreSQL Import

```bash
python data_postgresql.py
```

#### Method 3: ChromaDB Vector Embedding

```bash
python data_chroma_floats.py
```

## Troubleshooting

- Check PostgreSQL: `psql -U nematsachdeva -d argo -c "SELECT COUNT(*) FROM measurements;"`
- Verify ChromaDB: Query the backend API
- Check logs for detailed error messages

## Next Steps

After data ingestion, test queries in the frontend and verify results.
