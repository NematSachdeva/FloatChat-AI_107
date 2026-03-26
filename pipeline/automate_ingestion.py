"""One-command ingestion runner for DB + vector indexing."""

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automate FloatChat ingestion")
    parser.add_argument(
        "--local-snapshot-dir",
        default="",
        help="Path to downloaded Argo snapshot folder (example: /Users/.../202601-BgcArgoSprof)",
    )
    parser.add_argument(
        "--local-glob",
        default="dac/**/*_Sprof.nc",
        help="Glob pattern under snapshot folder for Sprof files",
    )
    parser.add_argument(
        "--max-profiles",
        type=int,
        default=0,
        help="Max profile files to ingest; 0 means all discovered files",
    )
    parser.add_argument(
        "--database-url",
        default="",
        help="Override DATABASE_URL for this run",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Only ingest to PostgreSQL; skip Chroma indexing",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.database_url:
        os.environ["DATABASE_URL"] = args.database_url

    if args.local_snapshot_dir:
        os.environ["ARGO_LOCAL_SNAPSHOT_DIR"] = args.local_snapshot_dir
    if args.local_glob:
        os.environ["ARGO_LOCAL_GLOB"] = args.local_glob

    os.environ["ARGO_MAX_PROFILES"] = str(args.max_profiles)

    # Import after environment overrides are set.
    from pipeline.ingest_seanoe_argo import ingest_from_local_snapshot, ingest_from_seanoe

    local_dir = os.environ.get("ARGO_LOCAL_SNAPSHOT_DIR", "").strip()
    max_profiles = int(os.environ.get("ARGO_MAX_PROFILES", "0"))

    if local_dir:
        print(f"Starting local ingestion from: {local_dir}")
        ingest_from_local_snapshot(snapshot_dir=local_dir, max_profiles=max_profiles)
    else:
        print("Starting remote GDAC ingestion")
        ingest_from_seanoe(max_profiles=max_profiles)

    if args.skip_index:
        print("Skipping Chroma indexing as requested.")
        return 0

    print("Building Chroma index from PostgreSQL measurements...")
    from pipeline.data_chroma_floats import create_float_aware_embeddings

    create_float_aware_embeddings()
    print("Ingestion automation completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
