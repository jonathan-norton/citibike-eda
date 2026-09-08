"""
Citi Bike trip data loader.

Handles the two mechanical traps in this dataset:
  1. Months over 1M trips are split across multiple CSVs (_1, _2, _3 ...).
     Every part must be read or you silently lose trips.
  2. Station IDs look numeric but are decimal-coded strings ('5406.02').
     Reading them as float mangles trailing zeros.

This module does INGESTION ONLY. It does not clean.
Cleaning decisions belong in your notebook where you can justify them.
"""

import glob
import os
import pandas as pd

DTYPES = {
    "ride_id": "string",
    "rideable_type": "string",
    "start_station_name": "string",
    "start_station_id": "string",
    "end_station_name": "string",
    "end_station_id": "string",
    "member_casual": "string",
}

TIMESTAMPS = ["started_at", "ended_at"]

EXPECTED_COLUMNS = [
    "ride_id", "rideable_type", "started_at", "ended_at",
    "start_station_name", "start_station_id",
    "end_station_name", "end_station_id",
    "start_lat", "start_lng", "end_lat", "end_lng",
    "member_casual",
]


def load_month(data_dir, yyyymm):
    """Read every CSV part for one month and return a single DataFrame."""
    pattern = os.path.join(data_dir, f"{yyyymm}-citibike-tripdata*.csv")
    paths = sorted(glob.glob(pattern))

    if not paths:
        raise FileNotFoundError(f"No files matched {pattern}")

    frames = []
    for p in paths:
        df = pd.read_csv(p, dtype=DTYPES, parse_dates=TIMESTAMPS)

        if list(df.columns) != EXPECTED_COLUMNS:
            raise ValueError(
                f"Unexpected columns in {os.path.basename(p)}:\n"
                f"  got      {list(df.columns)}\n"
                f"  expected {EXPECTED_COLUMNS}"
            )

        df["source_file"] = os.path.basename(p)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out["year_month"] = yyyymm
    return out


def load_range(data_dir, months):
    """Load several months. `months` is a list like ['202101', '202102']."""
    frames = []
    for m in months:
        df = load_month(data_dir, m)
        print(f"  {m}: {len(df):>10,} trips from {df.source_file.nunique()} file(s)")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def add_derived_columns(df):
    """
    Add fields the raw data does not ship with.

    Duration is derived because the current schema has no duration column.
    Nothing here removes or alters a row: derivation only.
    """
    df = df.copy()
    df["duration_min"] = (df.ended_at - df.started_at).dt.total_seconds() / 60
    df["start_hour"] = df.started_at.dt.hour
    df["start_dow"] = df.started_at.dt.dayofweek        # Monday = 0
    df["start_date"] = df.started_at.dt.date
    df["is_weekend"] = df.start_dow >= 5
    return df


def quality_report(df):
    """Quantify problems. Reports only, changes nothing."""
    n = len(df)
    print(f"rows: {n:,}   columns: {df.shape[1]}")

    print("\nmissing values:")
    miss = df.isna().sum()
    miss = miss[miss > 0]
    if miss.empty:
        print("  none")
    for c, v in miss.items():
        print(f"  {c:<22} {v:>9,}  ({v / n * 100:.2f}%)")

    print("\nduplicates:")
    print(f"  duplicate ride_id       {df.ride_id.duplicated().sum():>9,}")
    print(f"  fully duplicate rows    {df.duplicated().sum():>9,}")

    if "duration_min" in df:
        d = df.duration_min
        print("\nduration (minutes):")
        for label, val in [
            ("min", d.min()), ("median", d.median()), ("mean", d.mean()),
            ("p99", d.quantile(0.99)), ("max", d.max()),
        ]:
            print(f"  {label:<10} {val:>14,.2f}")
        print(f"  negative        {(d < 0).sum():>9,}")
        print(f"  under 1 min     {(d < 1).sum():>9,}")
        print(f"  over 24 hours   {(d > 1440).sum():>9,}")

    print("\nstation identifier cardinality:")
    print(f"  unique start_station_id    {df.start_station_id.nunique():>7,}")
    print(f"  unique start_station_name  {df.start_station_name.nunique():>7,}")


# ---------------------------------------------------------------------------
# CLEANING: your decisions go here.
#
# Task 3 asks you to explain and justify rather than simply deleting, so each
# of these needs a reason you can defend, not just a threshold.
#
#   1. Trips over 24 hours. In Jan 2021 there were 720 of them out of 1.1M,
#      and they moved the mean from 45.4 to 14.2 minutes. Decide: drop,
#      cap, or flag and keep? What does each choice do to your later charts?
#
#   2. Rows missing end station. These are not random. Their median duration
#      was 70.8 min against 9.7 for normal trips, so they look like bikes
#      that were never docked. Decide whether they are a separate phenomenon
#      worth analysing or noise to remove.
#
#   3. Station name vs ID as the grouping key. Jan 2021 had 1,224 IDs and
#      1,231 names, so names are the less stable key. Find the collisions
#      before you pick one.
#
#   4. Coordinate precision ranges from 2 to 8 decimal places. Two decimals
#      is about 1 km of error. Decide whether that matters for what you plot.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    DATA_DIR = "data/raw"
    MONTHS = [f"2021{m:02d}" for m in range(1, 13)]

    print("loading...")
    trips = load_range(DATA_DIR, MONTHS)
    trips = add_derived_columns(trips)

    print(f"\ntotal: {len(trips):,} trips\n")
    quality_report(trips)

    os.makedirs("data/interim", exist_ok=True)
    trips.to_parquet("data/interim/trips_2021_raw.parquet", index=False)
    print("\nwrote data/interim/trips_2021_raw.parquet")
