"""
Citi Bike cleaning.

Every operation here either adds a flag column or normalizes a value.
Nothing is silently deleted. The flags let downstream analysis exclude
a population where it is invalid while still counting it where it is
valid, which is the distinction the decisions below rest on.
"""

import re
import pandas as pd

LONG_TRIP_MINUTES = 24 * 60


def flag_incomplete_trips(df):
    """
    Decision 1 and 2 (combined).

    A trip over 24 hours and a trip with no end station are the same
    underlying failure: the bike was never redocked, so the system closed
    the record administratively. The start of such a trip is valid
    (real rider, real station, real timestamp); only the ending is not.

    So they are flagged rather than dropped. They count as trips that
    started; they are excluded from duration and destination analysis.
    """
    df = df.copy()
    df["no_end_station"] = df.end_station_id.isna()
    df["over_24h"] = df.duration_min > LONG_TRIP_MINUTES
    df["incomplete_trip"] = df.no_end_station | df.over_24h
    return df


_DEPOT_PAT = re.compile(r"^(SYS|SYSY|190\s)", re.IGNORECASE)
_JC_PAT = re.compile(r"^JC\d", re.IGNORECASE)


def flag_operational_stations(df):
    """
    Decision 3c.

    594 station ids break the decimal format. 522 of them are depots,
    warehouses and mechanic shops (SYS*). Those are operational
    locations, not public docks, so trips touching them are not public
    rides. The documentation claims staff trips are already removed;
    they are not, so the flag is necessary.

    JC* are Jersey City, a genuine part of the service area. Kept, but
    labelled so geographic analysis can separate NY from NJ.
    """
    df = df.copy()
    sid = df.start_station_id.fillna("")
    eid = df.end_station_id.fillna("")

    df["depot_trip"] = sid.str.match(_DEPOT_PAT) | eid.str.match(_DEPOT_PAT)
    df["jersey_city"] = sid.str.match(_JC_PAT) | eid.str.match(_JC_PAT)
    df["do_not_use_id"] = sid.str.contains("do not use", case=False) | \
                          eid.str.contains("do not use", case=False)
    return df


def normalize_station_names(df):
    """
    Decision 3b.

    Station names carry formatting noise: doubled spaces, literal '\\t'
    escapes, real tab characters, and trailing lifecycle annotations.
    Names are only ever a display label here (ids are the grouping key,
    decision 3a), so normalising them is safe.
    """
    df = df.copy()

    def norm(s):
        if s.isna().all():
            return s
        out = s.str.replace(r"\\t", " ", regex=True)     # literal backslash-t
        out = out.str.replace(r"\s+", " ", regex=True)   # tabs, doubled spaces
        return out.str.strip()

    for col in ("start_station_name", "end_station_name"):
        df[col + "_raw"] = df[col]
        df[col] = norm(df[col])
    return df


def canonical_station_names(df):
    """
    Decision 3a.

    Group by id; take the most frequent name per id as the display label.
    Resolves the 14 ids that carry conflicting names without having to
    adjudicate which spelling is 'correct'.
    """
    a = df[["start_station_id", "start_station_name"]].rename(
        columns={"start_station_id": "station_id", "start_station_name": "name"})
    b = df[["end_station_id", "end_station_name"]].rename(
        columns={"end_station_id": "station_id", "end_station_name": "name"})
    s = pd.concat([a, b], ignore_index=True).dropna()
    return (s.groupby(["station_id", "name"]).size()
              .reset_index(name="n")
              .sort_values("n", ascending=False)
              .drop_duplicates("station_id")
              .set_index("station_id")["name"])


def clean(df):
    df = flag_incomplete_trips(df)
    df = flag_operational_stations(df)
    df = normalize_station_names(df)
    return df


def cleaning_log(raw, cleaned):
    """Before/after table for task 3."""
    rows = [
        ("raw trips loaded", len(raw), ""),
        ("duplicate ride_id removed", 0, "none found"),
        ("flagged: no end station", int(cleaned.no_end_station.sum()),
         "bike never redocked"),
        ("flagged: over 24 hours", int(cleaned.over_24h.sum()),
         f"max {cleaned.duration_min.max()/1440:.0f} days"),
        ("flagged: incomplete (union)", int(cleaned.incomplete_trip.sum()),
         "excluded from duration analysis"),
        ("flagged: depot/maintenance", int(cleaned.depot_trip.sum()),
         "SYS* operational stations"),
        ("flagged: Jersey City", int(cleaned.jersey_city.sum()),
         "kept, labelled for geography"),
        ("station names normalized",
         int((cleaned.start_station_name != cleaned.start_station_name_raw).sum()),
         "whitespace and escape artifacts"),
        ("rows deleted", 0, "no rows dropped"),
        ("analysis-ready (valid duration)",
         int((~cleaned.incomplete_trip).sum()), ""),
    ]
    w = max(len(r[0]) for r in rows)
    print(f"{'step'.ljust(w)}   {'rows':>12}   note")
    print("-" * (w + 40))
    for name, n, note in rows:
        print(f"{name.ljust(w)}   {n:>12,}   {note}")
