#!/usr/bin/env python3
"""
Smart Vibration Alarms
======================
Calculates data-driven alarm thresholds from a rolling baseline and writes a
combined CSV that sits alongside the existing ISO alarm levels — the source
table is never modified.

Output CSV columns
  Metadata  : client, area, machine, component, bearing, point, parameter,
               type, unit, test_point_name
  ISO alarms: iso_pre_alarm, iso_alarm, iso_danger  (existing limits, unchanged)
  Smart      : baseline_avg, smart_warning, smart_alarm
  Audit      : stable, stability_note, slope_per_month_pct, trend_p_value,
               months_of_data, total_readings, computed_at

Usage
  # Demo (synthetic history built from the sample CSV):
  python3 smart_alarms.py --demo

  # Live database (edit DB_CONFIG below first):
  python3 smart_alarms.py --client DSM

  # All options:
  python3 smart_alarms.py --client DSM --months 6 --warning 1.3 --alarm 1.5
                          --batch-size 50 --out results.csv
"""

import argparse
import csv
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from scipy import stats as _scipy_stats

# ═══════════════════════════════════════════════════════════════════════════════
#  DATABASE CONFIG — fill in your details here
#  You can also override any field via environment variable (shown in comments).
# ═══════════════════════════════════════════════════════════════════════════════

DB_CONFIG = {
    "host":         os.environ.get("DB_HOST",         "localhost"),         # DB_HOST
    "port":         int(os.environ.get("DB_PORT",     "5432")),             # DB_PORT
    "dbname":       os.environ.get("DB_NAME",         "your_database"),     # DB_NAME
    "user":         os.environ.get("DB_USER",         "your_username"),     # DB_USER
    "password":     os.environ.get("DB_PASSWORD",     "your_password"),     # DB_PASSWORD
    "schema":       os.environ.get("DB_SCHEMA",       "public"),            # DB_SCHEMA
    "table":        os.environ.get("DB_TABLE",        "falcon_scalars"),    # DB_TABLE  (source — read only)
    "output_table": os.environ.get("DB_OUTPUT_TABLE", "smart_alarm_results"), # DB_OUTPUT_TABLE (written by this script)
}

# ═══════════════════════════════════════════════════════════════════════════════
#  ALGORITHM DEFAULTS  (all overridable via CLI flags)
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_CLIENT       = "DSM"
DEFAULT_MONTHS       = 6
DEFAULT_WARNING_MULT = 1.3
DEFAULT_ALARM_MULT   = 1.5
DEFAULT_BATCH_SIZE   = 50    # number of machines processed per database query

MIN_MONTHS_REQUIRED  = 3     # points with fewer months are skipped
SLOPE_THRESHOLD      = 0.05  # normalised slope > 5 %/month → unstable
P_VALUE_ALPHA        = 0.10  # significance level for trend test


# ─────────────────────────────────────────────────────────────────────────────
#  STABILITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

def is_stable(monthly_values: list[float]) -> tuple[bool, float | None, float | None]:
    """
    Returns (stable, slope_normalised, p_value).
    Both the slope threshold AND statistical significance must be exceeded to
    flag a point as unstable — one alone is not sufficient.
    """
    n = len(monthly_values)
    if n < 2:
        return False, None, None

    mean_val = sum(monthly_values) / n
    if mean_val <= 0:
        return True, 0.0, 1.0

    result     = _scipy_stats.linregress(range(n), monthly_values)
    slope_norm = result.slope / mean_val
    p_value    = result.pvalue

    trending_up = (p_value < P_VALUE_ALPHA) and (slope_norm > SLOPE_THRESHOLD)
    return not trending_up, round(slope_norm, 5), round(p_value, 5)


# ─────────────────────────────────────────────────────────────────────────────
#  CORE CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_smart_alarms(
    monthly_data: dict,
    latest_meta: dict,
    warning_mult: float,
    alarm_mult: float,
    computed_at: str,
) -> list[dict]:
    """
    monthly_data : {(machine_guid, point, parameter, type):
                     [(month_str, avg_value, n_readings), ...]}
    latest_meta  : {same key: metadata dict}
    Returns list of result dicts, one per measurement point.
    """
    results = []

    for key, months_list in monthly_data.items():
        months_list  = sorted(months_list, key=lambda r: r[0])
        values       = [r[1] for r in months_list]
        n_months     = len(values)
        total_reads  = sum(r[2] for r in months_list)
        meta         = latest_meta.get(key, {})

        row = {
            # ── metadata ──────────────────────────────────────────────────────
            "client":          meta.get("client", ""),
            "area":            meta.get("area", ""),
            "machine_guid":    key[0],
            "machine":         meta.get("machine", ""),
            "component":       meta.get("component", ""),
            "bearing":         meta.get("bearing", ""),
            "point":           key[1],
            "parameter":       key[2],
            "type":            key[3],
            "unit":            meta.get("unit", ""),
            "test_point_name": meta.get("test_point_name", ""),
            # ── existing ISO alarms (untouched) ───────────────────────────────
            "iso_pre_alarm":   meta.get("pal_plus"),
            "iso_alarm":       meta.get("al_plus"),
            "iso_danger":      meta.get("dg_plus"),
            # ── smart alarm outputs ───────────────────────────────────────────
            "baseline_avg":        None,
            "smart_warning":       None,
            "smart_alarm":         None,
            # ── stability audit ───────────────────────────────────────────────
            "stable":              None,
            "stability_note":      "",
            "slope_per_month_pct": None,
            "trend_p_value":       None,
            "months_of_data":      n_months,
            "total_readings":      total_reads,
            "computed_at":         computed_at,
        }

        if n_months < MIN_MONTHS_REQUIRED:
            row["stability_note"] = (
                f"Insufficient data: {n_months} month(s), need {MIN_MONTHS_REQUIRED}"
            )
            results.append(row)
            continue

        stable, slope_norm, p_val = is_stable(values)
        baseline_avg = sum(values) / len(values)

        row["baseline_avg"]         = round(baseline_avg, 6)
        row["stable"]               = stable
        row["slope_per_month_pct"]  = round(slope_norm * 100, 2) if slope_norm is not None else None
        row["trend_p_value"]        = p_val

        if stable:
            row["smart_warning"]  = round(baseline_avg * warning_mult, 4)
            row["smart_alarm"]    = round(baseline_avg * alarm_mult, 4)
            row["stability_note"] = "Stable baseline"
        else:
            direction = "increasing" if (slope_norm or 0) > 0 else "decreasing"
            row["stability_note"] = (
                f"Unstable ({direction} trend, "
                f"slope={slope_norm * 100:.1f}%/month, p={p_val:.3f}) — "
                "ISO alarms remain active"
            )

        results.append(row)

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  OUTPUT TABLE  — created automatically, never touches the source table
# ─────────────────────────────────────────────────────────────────────────────

SQL_CREATE_OUTPUT_TABLE = """
CREATE TABLE IF NOT EXISTS {schema}.{output_table} (
    -- identity / primary key
    client              TEXT         NOT NULL,
    machine_guid        TEXT         NOT NULL,
    point               TEXT         NOT NULL,
    parameter           TEXT         NOT NULL,
    type                TEXT         NOT NULL,
    -- metadata
    area                TEXT,
    machine             TEXT,
    component           TEXT,
    bearing             TEXT,
    unit                TEXT,
    test_point_name     TEXT,
    -- existing ISO alarms (copied from source, never modified there)
    iso_pre_alarm       NUMERIC,
    iso_alarm           NUMERIC,
    iso_danger          NUMERIC,
    -- smart alarm thresholds
    baseline_avg        NUMERIC,
    smart_warning       NUMERIC,
    smart_alarm         NUMERIC,
    -- stability audit
    stable              BOOLEAN,
    stability_note      TEXT,
    slope_per_month_pct NUMERIC,
    trend_p_value       NUMERIC,
    months_of_data      INTEGER,
    total_readings      INTEGER,
    computed_at         TIMESTAMPTZ,
    PRIMARY KEY (client, machine_guid, point, parameter, type)
);
"""

SQL_UPSERT_ROW = """
INSERT INTO {schema}.{output_table} (
    client, machine_guid, point, parameter, type,
    area, machine, component, bearing, unit, test_point_name,
    iso_pre_alarm, iso_alarm, iso_danger,
    baseline_avg, smart_warning, smart_alarm,
    stable, stability_note, slope_per_month_pct, trend_p_value,
    months_of_data, total_readings, computed_at
) VALUES (
    :client, :machine_guid, :point, :parameter, :type,
    :area, :machine, :component, :bearing, :unit, :test_point_name,
    :iso_pre_alarm, :iso_alarm, :iso_danger,
    :baseline_avg, :smart_warning, :smart_alarm,
    :stable, :stability_note, :slope_per_month_pct, :trend_p_value,
    :months_of_data, :total_readings, :computed_at
)
ON CONFLICT (client, machine_guid, point, parameter, type) DO UPDATE SET
    area                = EXCLUDED.area,
    machine             = EXCLUDED.machine,
    component           = EXCLUDED.component,
    bearing             = EXCLUDED.bearing,
    unit                = EXCLUDED.unit,
    test_point_name     = EXCLUDED.test_point_name,
    iso_pre_alarm       = EXCLUDED.iso_pre_alarm,
    iso_alarm           = EXCLUDED.iso_alarm,
    iso_danger          = EXCLUDED.iso_danger,
    baseline_avg        = EXCLUDED.baseline_avg,
    smart_warning       = EXCLUDED.smart_warning,
    smart_alarm         = EXCLUDED.smart_alarm,
    stable              = EXCLUDED.stable,
    stability_note      = EXCLUDED.stability_note,
    slope_per_month_pct = EXCLUDED.slope_per_month_pct,
    trend_p_value       = EXCLUDED.trend_p_value,
    months_of_data      = EXCLUDED.months_of_data,
    total_readings      = EXCLUDED.total_readings,
    computed_at         = EXCLUDED.computed_at;
"""


def write_results_to_db(engine, cfg: dict, results: list[dict], t0: float) -> None:
    """Creates the output table if needed, then upserts all results."""
    import sqlalchemy as sa

    schema       = cfg["schema"]
    output_table = cfg["output_table"]

    print(f"[{_elapsed(t0)}]  Writing {len(results)} rows to {schema}.{output_table}...")

    with engine.begin() as conn:
        conn.execute(sa.text(
            SQL_CREATE_OUTPUT_TABLE.format(schema=schema, output_table=output_table)
        ))

        upsert = sa.text(SQL_UPSERT_ROW.format(schema=schema, output_table=output_table))
        for row in results:
            conn.execute(upsert, {
                "client":              row.get("client") or "",
                "machine_guid":        row.get("machine_guid") or "",
                "point":               row.get("point") or "",
                "parameter":           row.get("parameter") or "",
                "type":                row.get("type") or "",
                "area":                row.get("area"),
                "machine":             row.get("machine"),
                "component":           row.get("component"),
                "bearing":             row.get("bearing"),
                "unit":                row.get("unit"),
                "test_point_name":     row.get("test_point_name"),
                "iso_pre_alarm":       row.get("iso_pre_alarm"),
                "iso_alarm":           row.get("iso_alarm"),
                "iso_danger":          row.get("iso_danger"),
                "baseline_avg":        row.get("baseline_avg"),
                "smart_warning":       row.get("smart_warning"),
                "smart_alarm":         row.get("smart_alarm"),
                "stable":              row.get("stable"),
                "stability_note":      row.get("stability_note"),
                "slope_per_month_pct": row.get("slope_per_month_pct"),
                "trend_p_value":       row.get("trend_p_value"),
                "months_of_data":      row.get("months_of_data"),
                "total_readings":      row.get("total_readings"),
                "computed_at":         row.get("computed_at"),
            })

    print(f"[{_elapsed(t0)}]  Database write complete.")


# ─────────────────────────────────────────────────────────────────────────────
#  DATABASE MODE
# ─────────────────────────────────────────────────────────────────────────────

def _make_db_url(cfg: dict) -> str:
    return (
        f"postgresql+psycopg2://{cfg['user']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['dbname']}"
    )


def _table(cfg: dict) -> str:
    return f"{cfg['schema']}.{cfg['table']}"


SQL_MACHINES = """
SELECT DISTINCT machine_guid, machine
FROM {table}
WHERE client = :client
  AND date_meas >= NOW() - INTERVAL '{months} months'
ORDER BY machine_guid
"""

SQL_MONTHLY_BATCH = """
SELECT
    machine_guid,
    point,
    parameter,
    type,
    TO_CHAR(DATE_TRUNC('month', date_meas), 'YYYY-MM') AS month,
    AVG(value::float)  AS avg_value,
    COUNT(*)           AS reading_count
FROM {table}
WHERE
    client       = :client
    AND machine_guid IN :guids
    AND date_meas    >= NOW() - INTERVAL '{months} months'
    AND date_meas    <  NOW()
    AND status_code::int IN (2, 3, 4, 5)
    AND value IS NOT NULL
    AND value::float > 0
GROUP BY machine_guid, point, parameter, type,
         DATE_TRUNC('month', date_meas)
ORDER BY machine_guid, point, parameter, type, month
"""

SQL_META_BATCH = """
SELECT DISTINCT ON (machine_guid, point, parameter, type)
    machine_guid,
    point,
    parameter,
    type,
    unit,
    client,
    machine,
    component,
    bearing,
    test_point_name,
    COALESCE(area_sub_1, '') AS area,
    pal_plus,
    al_plus,
    dg_plus
FROM {table}
WHERE
    client       = :client
    AND machine_guid IN :guids
    AND date_meas >= NOW() - INTERVAL '30 days'
ORDER BY machine_guid, point, parameter, type, date_meas DESC
"""


def run_db_mode(
    cfg: dict,
    client: str,
    months: int,
    warning_mult: float,
    alarm_mult: float,
    batch_size: int,
    write_db: bool = True,
) -> list[dict]:
    try:
        import sqlalchemy as sa
    except ImportError:
        sys.exit("sqlalchemy is required: pip install sqlalchemy psycopg2-binary")

    db_url  = _make_db_url(cfg)
    tbl     = _table(cfg)
    engine  = sa.create_engine(db_url)
    run_ts  = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    t_total = time.perf_counter()

    with engine.connect() as conn:
        # ── 1. Get full machine list ──────────────────────────────────────────
        print(f"[{_elapsed(t_total)}]  Fetching machine list for client='{client}'...")
        machine_rows = conn.execute(
            sa.text(SQL_MACHINES.format(table=tbl, months=months)),
            {"client": client},
        ).fetchall()

    machine_guids = [r.machine_guid for r in machine_rows]
    total_machines = len(machine_guids)
    batches = [
        machine_guids[i: i + batch_size]
        for i in range(0, total_machines, batch_size)
    ]
    n_batches = len(batches)
    print(f"[{_elapsed(t_total)}]  {total_machines} machines → {n_batches} batch(es) of ≤{batch_size}\n")

    all_monthly: dict = defaultdict(list)
    all_meta:    dict = {}

    # ── 2. Process in batches ─────────────────────────────────────────────────
    batch_times: list[float] = []
    for i, batch in enumerate(batches, start=1):
        t_batch = time.perf_counter()
        guid_tuple = tuple(batch)

        with engine.connect() as conn:
            monthly_rows = conn.execute(
                sa.text(SQL_MONTHLY_BATCH.format(table=tbl, months=months)).bindparams(
                    sa.bindparam("guids", expanding=True)
                ),
                {"client": client, "guids": list(guid_tuple)},
            ).fetchall()

            meta_rows = conn.execute(
                sa.text(SQL_META_BATCH.format(table=tbl)).bindparams(
                    sa.bindparam("guids", expanding=True)
                ),
                {"client": client, "guids": list(guid_tuple)},
            ).fetchall()

        for row in monthly_rows:
            key = (row.machine_guid, row.point, row.parameter, row.type)
            all_monthly[key].append((row.month, float(row.avg_value), int(row.reading_count)))

        for row in meta_rows:
            key = (row.machine_guid, row.point, row.parameter, row.type)
            all_meta[key] = {
                "client":          row.client,
                "area":            row.area,
                "machine":         row.machine,
                "component":       row.component,
                "bearing":         row.bearing,
                "unit":            row.unit,
                "test_point_name": row.test_point_name,
                "pal_plus":        row.pal_plus,
                "al_plus":         row.al_plus,
                "dg_plus":         row.dg_plus,
            }

        elapsed_batch = time.perf_counter() - t_batch
        batch_times.append(elapsed_batch)
        avg_batch = sum(batch_times) / len(batch_times)
        remaining = avg_batch * (n_batches - i)
        pct = i / n_batches * 100
        print(
            f"[{_elapsed(t_total)}]  Batch {i:>{len(str(n_batches))}}/{n_batches} "
            f"({pct:5.1f}%)  {len(batch)} machines  "
            f"{elapsed_batch:.1f}s/batch  ETA {_fmt_seconds(remaining)}"
        )

    # ── 3. Compute smart alarms ───────────────────────────────────────────────
    print(f"\n[{_elapsed(t_total)}]  Computing smart alarm thresholds...")
    results = compute_smart_alarms(all_monthly, all_meta, warning_mult, alarm_mult, run_ts)

    # ── 4. Write results back to the database ─────────────────────────────────
    if write_db:
        write_results_to_db(engine, cfg, results, t_total)
    else:
        print(f"[{_elapsed(t_total)}]  Skipping database write (--no-db-write).")

    print(f"[{_elapsed(t_total)}]  Done.  Total runtime: {_elapsed(t_total)}\n")
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  DEMO MODE  (synthetic history from sample CSV)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_float(v):
    try:
        return float(v) if v and v.upper() != "NULL" else None
    except ValueError:
        return None


def run_demo_mode(
    csv_path: str,
    client: str,
    months: int,
    warning_mult: float,
    alarm_mult: float,
    batch_size: int,
) -> list[dict]:
    """Generates synthetic monthly history so every part of the algorithm runs."""
    print(f"Demo mode — building synthetic {months}-month history from {csv_path}")
    print(f"Client filter: '{client}'\n")
    random.seed(42)
    t_total = time.perf_counter()
    run_ts  = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    # Apply client filter
    rows = [r for r in rows if r.get("client", "").strip() == client]
    if not rows:
        sys.exit(f"No rows found for client='{client}' in {csv_path}")

    snapshots: dict = {}
    for row in rows:
        if row.get("status_code") not in {"2", "3", "4", "5"}:
            continue
        val = _parse_float(row.get("value"))
        if val is None or val <= 0:
            continue
        key = (row["machine_guid"], row["point"], row["parameter"], row["type"])
        snapshots[key] = row

    keys = list(snapshots.keys())
    n    = len(keys)
    random.shuffle(keys)
    stable_keys = set(keys[: int(n * 0.70)])
    rising_keys = set(keys[int(n * 0.70): int(n * 0.90)])

    now = datetime.now(tz=timezone.utc)

    # Batch the machine_guids just like the real DB path
    machine_guids = list({k[0] for k in snapshots})
    batches       = [machine_guids[i: i + batch_size] for i in range(0, len(machine_guids), batch_size)]
    n_batches     = len(batches)
    print(f"  {len(machine_guids)} machines → {n_batches} batch(es) of ≤{batch_size}\n")

    all_monthly: dict = defaultdict(list)
    all_meta:    dict = {}
    batch_times: list[float] = []

    for b_idx, batch in enumerate(batches, start=1):
        t_batch  = time.perf_counter()
        batch_set = set(batch)

        for key, row in snapshots.items():
            if key[0] not in batch_set:
                continue

            base_val = _parse_float(row["value"])
            if base_val is None or base_val <= 0:
                continue

            if key in stable_keys:
                n_m, trend, noise = months, 0.0,  0.08
            elif key in rising_keys:
                n_m, trend, noise = months, 0.08, 0.05
            else:
                n_m, trend, noise = random.randint(1, MIN_MONTHS_REQUIRED - 1), 0.0, 0.12

            for i in range(n_m):
                offset    = n_m - 1 - i
                month_str = (now - timedelta(days=30 * offset)).strftime("%Y-%m")
                val       = max(base_val * (1 + trend * i) * (1 + noise * (random.random() * 2 - 1)), 1e-6)
                all_monthly[key].append((month_str, val, random.randint(3, 12)))

            all_meta[key] = {
                "client":          row.get("client", ""),
                "area":            row.get("area_sub_1", ""),
                "machine":         row.get("machine", ""),
                "component":       row.get("component", ""),
                "bearing":         row.get("bearing", ""),
                "unit":            row.get("unit", ""),
                "test_point_name": row.get("test_point_name", ""),
                "pal_plus":        _parse_float(row.get("pal_plus")),
                "al_plus":         _parse_float(row.get("al_plus")),
                "dg_plus":         _parse_float(row.get("dg_plus")),
            }

        elapsed_batch = time.perf_counter() - t_batch
        batch_times.append(elapsed_batch)
        avg_b     = sum(batch_times) / len(batch_times)
        remaining = avg_b * (n_batches - b_idx)
        pct       = b_idx / n_batches * 100
        print(
            f"[{_elapsed(t_total)}]  Batch {b_idx:>{len(str(n_batches))}}/{n_batches} "
            f"({pct:5.1f}%)  {len(batch)} machines  "
            f"{elapsed_batch:.3f}s/batch  ETA {_fmt_seconds(remaining)}"
        )

    print(f"\n[{_elapsed(t_total)}]  Computing smart alarm thresholds...")
    results = compute_smart_alarms(all_monthly, all_meta, warning_mult, alarm_mult, run_ts)
    print(f"[{_elapsed(t_total)}]  Done.  Total runtime: {_elapsed(t_total)}\n")
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  TIMING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

_T0: float = 0.0

def _elapsed(t0: float) -> str:
    s = time.perf_counter() - t0
    return f"{int(s // 60):02d}:{s % 60:05.2f}"

def _fmt_seconds(s: float) -> str:
    if s < 60:
        return f"{s:.0f}s"
    return f"{int(s // 60)}m {int(s % 60)}s"


# ─────────────────────────────────────────────────────────────────────────────
#  OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_FIELDS = [
    # metadata
    "client", "area", "machine_guid", "machine", "component", "bearing",
    "point", "parameter", "type", "unit", "test_point_name",
    # existing ISO alarms
    "iso_pre_alarm", "iso_alarm", "iso_danger",
    # smart alarm outputs
    "baseline_avg", "smart_warning", "smart_alarm",
    # stability audit
    "stable", "stability_note", "slope_per_month_pct", "trend_p_value",
    "months_of_data", "total_readings", "computed_at",
]


def write_csv(results: list[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def print_summary(results: list[dict]) -> None:
    total        = len(results)
    if total == 0:
        print("No results.")
        return
    has_smart    = sum(1 for r in results if r["smart_warning"] is not None)
    unstable     = sum(1 for r in results if r["stable"] is False)
    insufficient = sum(1 for r in results if r["stable"] is None)

    print(f"{'='*65}")
    print(f"  Smart Alarm Summary")
    print(f"{'='*65}")
    print(f"  Total measurement points  : {total}")
    print(f"  Smart alarms computed     : {has_smart}  ({has_smart/total*100:.0f}%)")
    print(f"  Unstable — ISO kept active: {unstable}")
    print(f"  Insufficient data (skipped): {insufficient}")
    print(f"{'='*65}\n")

    sample = [r for r in results if r["smart_warning"] is not None][:6]
    if sample:
        print("  Sample — stable points:")
        for r in sample:
            iso_a = r["iso_alarm"] if r["iso_alarm"] is not None else "N/A"
            print(
                f"    {str(r['point']):12s}  {str(r['parameter']):26s}"
                f"  avg={r['baseline_avg']:.4f} {str(r['unit']):5s}"
                f"  smart_warn={r['smart_warning']:.4f}"
                f"  smart_alarm={r['smart_alarm']:.4f}"
                f"  (ISO alert={iso_a})"
            )

    bad = [r for r in results if r["stable"] is False][:3]
    if bad:
        print("\n  Sample — unstable points (ISO alarms remain active):")
        for r in bad:
            print(
                f"    {str(r['point']):12s}  {str(r['parameter']):26s}"
                f"  slope={r['slope_per_month_pct']:+.1f}%/mo"
                f"  p={r['trend_p_value']:.3f}"
            )
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute smart vibration alarm thresholds",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--demo",       action="store_true",
                        help="Demo mode: use sample CSV with synthetic history")
    parser.add_argument("--csv",        default="falcon_scalars_example_data_1000_rows.csv",
                        help="Sample CSV path (demo mode only)")
    parser.add_argument("--client",     default=DEFAULT_CLIENT,
                        help="Client name to filter on")
    parser.add_argument("--months",     type=int,   default=DEFAULT_MONTHS,
                        help="Lookback window in months")
    parser.add_argument("--warning",    type=float, default=DEFAULT_WARNING_MULT,
                        help="Warning threshold multiplier")
    parser.add_argument("--alarm",      type=float, default=DEFAULT_ALARM_MULT,
                        help="Alarm threshold multiplier")
    parser.add_argument("--batch-size",  type=int,  default=DEFAULT_BATCH_SIZE,
                        help="Machines per database query batch")
    parser.add_argument("--out",         default="smart_alarm_results.csv",
                        help="Output CSV path")
    parser.add_argument("--no-db-write", action="store_true",
                        help="Skip writing results back to the database (CSV only)")
    args = parser.parse_args()

    if args.demo:
        results = run_demo_mode(
            args.csv, args.client, args.months,
            args.warning, args.alarm, args.batch_size,
        )
    else:
        results = run_db_mode(
            DB_CONFIG, args.client, args.months,
            args.warning, args.alarm, args.batch_size,
            write_db=not args.no_db_write,
        )

    print_summary(results)
    write_csv(results, args.out)
    print(f"Results written to: {args.out}")


if __name__ == "__main__":
    main()
