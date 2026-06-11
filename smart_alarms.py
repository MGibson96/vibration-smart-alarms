#!/usr/bin/env python3
"""
Smart Vibration Alarms
======================
Calculates data-driven alarm thresholds from a rolling baseline and writes a
combined table that sits alongside the existing ISO alarm levels — the source
table is never modified.

Database: Microsoft SQL Server (T-SQL)

Output table columns
  Metadata  : client, area, machine, component, bearing, point, parameter,
               type, unit, test_point_name
  ISO alarms: iso_pre_alarm, iso_alarm, iso_danger  (existing limits, unchanged)
  Smart      : baseline_avg, smart_warning, smart_alarm
  Audit      : stable, stability_note, slope_per_month_pct, trend_p_value,
               months_of_data, total_readings, computed_at

Usage
  # Test connections only:
  python3 smart_alarms.py --test

  # Demo (synthetic history from sample CSV, no DB needed):
  python3 smart_alarms.py --demo

  # Live database:
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

from dotenv import load_dotenv
from scipy import stats as _scipy_stats

# Load .env from the same directory as this script
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# ═══════════════════════════════════════════════════════════════════════════════
#  DATABASE CONFIG  — edit .env, not this file
# ═══════════════════════════════════════════════════════════════════════════════

READ_CONFIG = {
    "host":     os.environ.get("READ_DB_HOST", ""),
    "port":     int(os.environ.get("READ_DB_PORT", "1433")),
    "dbname":   os.environ.get("READ_DB_NAME", ""),
    "user":     os.environ.get("READ_DB_USER", ""),
    "password": os.environ.get("READ_DB_PASSWORD", ""),
    "schema":   os.environ.get("DB_SCHEMA", "dbo"),
    "table":    os.environ.get("READ_DB_TABLE", "falcon_scalars"),
}

WRITE_CONFIG = {
    "host":         os.environ.get("WRITE_DB_HOST", ""),
    "port":         int(os.environ.get("WRITE_DB_PORT", "1433")),
    "dbname":       os.environ.get("WRITE_DB_NAME", ""),
    "user":         os.environ.get("WRITE_DB_USER", ""),
    "password":     os.environ.get("WRITE_DB_PASSWORD", ""),
    "schema":       os.environ.get("DB_SCHEMA", "dbo"),
    "output_table": os.environ.get("WRITE_DB_OUTPUT_TABLE", "smart_alarm_results"),
}

# ═══════════════════════════════════════════════════════════════════════════════
#  ALGORITHM DEFAULTS  (all overridable via CLI flags)
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_CLIENT       = "DSM"
DEFAULT_MONTHS       = 6
DEFAULT_WARNING_MULT = 1.3
DEFAULT_ALARM_MULT   = 1.5
DEFAULT_BATCH_SIZE   = 50

MIN_MONTHS_REQUIRED  = 3
SLOPE_THRESHOLD      = 0.05   # normalised slope > 5 %/month → unstable
P_VALUE_ALPHA        = 0.10


# ─────────────────────────────────────────────────────────────────────────────
#  STABILITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

def is_stable(monthly_values: list[float]) -> tuple[bool, float | None, float | None]:
    """
    Returns (stable, slope_normalised, p_value).
    Both the slope threshold AND statistical significance must be exceeded
    to flag a point as unstable — one alone is not sufficient.
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
    results = []

    for key, months_list in monthly_data.items():
        months_list = sorted(months_list, key=lambda r: r[0])
        values      = [r[1] for r in months_list]
        n_months    = len(values)
        total_reads = sum(r[2] for r in months_list)
        meta        = latest_meta.get(key, {})

        row = {
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
            "iso_pre_alarm":   meta.get("pal_plus"),
            "iso_alarm":       meta.get("al_plus"),
            "iso_danger":      meta.get("dg_plus"),
            "baseline_avg":        None,
            "smart_warning":       None,
            "smart_alarm":         None,
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
#  SQL SERVER HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _make_db_url(cfg: dict) -> str:
    """mssql+pymssql connection URL."""
    return (
        f"mssql+pymssql://{cfg['user']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['dbname']}"
    )


def _source_table(cfg: dict) -> str:
    return f"[{cfg['schema']}].[{cfg['table']}]"


def _output_table_name(cfg: dict) -> str:
    return f"[{cfg['schema']}].[{cfg['output_table']}]"


def _guid_list(guids: list[str]) -> str:
    """Format a list of GUIDs for SQL IN clause (UUIDs are safe to inline)."""
    return ", ".join(f"'{g}'" for g in guids)


# ─────────────────────────────────────────────────────────────────────────────
#  T-SQL QUERIES  (SQL Server syntax)
# ─────────────────────────────────────────────────────────────────────────────

SQL_MACHINES = """
SELECT DISTINCT machine_guid, machine
FROM {table}
WHERE client = :client
  AND date_meas >= DATEADD(month, -{months}, GETUTCDATE())
ORDER BY machine_guid
"""

SQL_MONTHLY_BATCH = """
SELECT
    machine_guid,
    point,
    parameter,
    type,
    FORMAT(DATEADD(month, DATEDIFF(month, 0, date_meas), 0), 'yyyy-MM') AS month,
    AVG(CAST(value AS FLOAT))  AS avg_value,
    COUNT(*)                   AS reading_count
FROM {table}
WHERE
    client = :client
    AND machine_guid IN ({guids})
    AND date_meas >= DATEADD(month, -{months}, GETUTCDATE())
    AND date_meas <  GETUTCDATE()
    AND CAST(status_code AS INT) IN (2, 3, 4, 5)
    AND value IS NOT NULL
    AND CAST(value AS FLOAT) > 0
GROUP BY
    machine_guid, point, parameter, type,
    DATEADD(month, DATEDIFF(month, 0, date_meas), 0)
ORDER BY machine_guid, point, parameter, type, month
"""

SQL_META_BATCH = """
WITH ranked AS (
    SELECT
        machine_guid, point, parameter, type, unit, client,
        machine, component, bearing, test_point_name,
        COALESCE(area_sub_1, '') AS area,
        pal_plus, al_plus, dg_plus,
        ROW_NUMBER() OVER (
            PARTITION BY machine_guid, point, parameter, type
            ORDER BY date_meas DESC
        ) AS rn
    FROM {table}
    WHERE
        client = :client
        AND machine_guid IN ({guids})
        AND date_meas >= DATEADD(day, -30, GETUTCDATE())
)
SELECT * FROM ranked WHERE rn = 1
"""

SQL_CREATE_OUTPUT_TABLE = """
IF NOT EXISTS (
    SELECT 1 FROM INFORMATION_SCHEMA.TABLES
    WHERE TABLE_SCHEMA = '{schema}' AND TABLE_NAME = '{output_table}'
)
CREATE TABLE [{schema}].[{output_table}] (
    client              NVARCHAR(255)  NOT NULL,
    machine_guid        NVARCHAR(255)  NOT NULL,
    point               NVARCHAR(255)  NOT NULL,
    parameter           NVARCHAR(255)  NOT NULL,
    type                NVARCHAR(255)  NOT NULL,
    area                NVARCHAR(500),
    machine             NVARCHAR(255),
    component           NVARCHAR(255),
    bearing             NVARCHAR(255),
    unit                NVARCHAR(50),
    test_point_name     NVARCHAR(500),
    iso_pre_alarm       FLOAT,
    iso_alarm           FLOAT,
    iso_danger          FLOAT,
    baseline_avg        FLOAT,
    smart_warning       FLOAT,
    smart_alarm         FLOAT,
    stable              BIT,
    stability_note      NVARCHAR(MAX),
    slope_per_month_pct FLOAT,
    trend_p_value       FLOAT,
    months_of_data      INT,
    total_readings      INT,
    computed_at         DATETIME2,
    CONSTRAINT PK_smart_alarm_results
        PRIMARY KEY (client, machine_guid, point, parameter, type)
)
"""

SQL_MERGE = """
MERGE [{schema}].[{output_table}] WITH (HOLDLOCK) AS target
USING (SELECT
    :client              AS client,
    :machine_guid        AS machine_guid,
    :point               AS point,
    :parameter           AS parameter,
    :type                AS type,
    :area                AS area,
    :machine             AS machine,
    :component           AS component,
    :bearing             AS bearing,
    :unit                AS unit,
    :test_point_name     AS test_point_name,
    :iso_pre_alarm       AS iso_pre_alarm,
    :iso_alarm           AS iso_alarm,
    :iso_danger          AS iso_danger,
    :baseline_avg        AS baseline_avg,
    :smart_warning       AS smart_warning,
    :smart_alarm         AS smart_alarm,
    :stable              AS stable,
    :stability_note      AS stability_note,
    :slope_per_month_pct AS slope_per_month_pct,
    :trend_p_value       AS trend_p_value,
    :months_of_data      AS months_of_data,
    :total_readings      AS total_readings,
    :computed_at         AS computed_at
) AS source
ON  target.client       = source.client
AND target.machine_guid = source.machine_guid
AND target.point        = source.point
AND target.parameter    = source.parameter
AND target.type         = source.type
WHEN MATCHED THEN UPDATE SET
    area                = source.area,
    machine             = source.machine,
    component           = source.component,
    bearing             = source.bearing,
    unit                = source.unit,
    test_point_name     = source.test_point_name,
    iso_pre_alarm       = source.iso_pre_alarm,
    iso_alarm           = source.iso_alarm,
    iso_danger          = source.iso_danger,
    baseline_avg        = source.baseline_avg,
    smart_warning       = source.smart_warning,
    smart_alarm         = source.smart_alarm,
    stable              = source.stable,
    stability_note      = source.stability_note,
    slope_per_month_pct = source.slope_per_month_pct,
    trend_p_value       = source.trend_p_value,
    months_of_data      = source.months_of_data,
    total_readings      = source.total_readings,
    computed_at         = source.computed_at
WHEN NOT MATCHED THEN INSERT (
    client, machine_guid, point, parameter, type,
    area, machine, component, bearing, unit, test_point_name,
    iso_pre_alarm, iso_alarm, iso_danger,
    baseline_avg, smart_warning, smart_alarm,
    stable, stability_note, slope_per_month_pct, trend_p_value,
    months_of_data, total_readings, computed_at
) VALUES (
    source.client, source.machine_guid, source.point, source.parameter, source.type,
    source.area, source.machine, source.component, source.bearing, source.unit,
    source.test_point_name, source.iso_pre_alarm, source.iso_alarm, source.iso_danger,
    source.baseline_avg, source.smart_warning, source.smart_alarm,
    source.stable, source.stability_note, source.slope_per_month_pct, source.trend_p_value,
    source.months_of_data, source.total_readings, source.computed_at
);
"""


# ─────────────────────────────────────────────────────────────────────────────
#  CONNECTION TEST
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_cfg(cfg: dict) -> str:
    """Human-readable connection summary — never shows the password."""
    tbl = cfg.get("table") or cfg.get("output_table", "")
    return (
        f"{cfg['user']}@{cfg['host']}:{cfg['port']}"
        f"/{cfg['dbname']}  schema={cfg['schema']}  table={tbl}"
    )


def test_connections(read_cfg: dict, write_cfg: dict) -> bool:
    """
    Tests both connections with a lightweight query.
    READ  — verifies the source table is reachable.
    WRITE — verifies the write user can connect (table created on first run).
    Returns True if both pass.
    """
    try:
        import sqlalchemy as sa
    except ImportError:
        sys.exit("sqlalchemy is required: pip install sqlalchemy pymssql")

    all_ok = True
    checks = [
        ("READ ", read_cfg,  f"SELECT TOP 1 1 FROM {_source_table(read_cfg)}"),
        ("WRITE", write_cfg, "SELECT 1"),
    ]

    print("\n── Connection test ──────────────────────────────────────────")
    for label, cfg, probe_sql in checks:
        if not cfg.get("host"):
            print(f"  [{label}]  SKIP  (no host set in .env)")
            all_ok = False
            continue
        desc = _fmt_cfg(cfg)
        try:
            engine = sa.create_engine(_make_db_url(cfg))
            with engine.connect() as conn:
                conn.execute(sa.text(probe_sql))
            print(f"  [{label}]  OK    {desc}")
        except Exception as e:
            print(f"  [{label}]  FAIL  {desc}")
            # Print just the first line of the error to keep output readable
            print(f"           {str(e).splitlines()[0]}")
            all_ok = False

    print("─────────────────────────────────────────────────────────────\n")
    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
#  DATABASE MODE
# ─────────────────────────────────────────────────────────────────────────────

def run_db_mode(
    read_cfg: dict,
    write_cfg: dict,
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
        sys.exit("sqlalchemy is required: pip install sqlalchemy pymssql")

    run_ts  = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    t_total = time.perf_counter()

    if not test_connections(read_cfg, write_cfg):
        sys.exit("Connection test failed — fix the errors above before running.")

    read_engine = sa.create_engine(_make_db_url(read_cfg))
    tbl         = _source_table(read_cfg)

    # ── 1. Fetch machine list ─────────────────────────────────────────────────
    print(f"[{_elapsed(t_total)}]  Fetching machine list for client='{client}'...")
    with read_engine.connect() as conn:
        machine_rows = conn.execute(
            sa.text(SQL_MACHINES.format(table=tbl, months=months)),
            {"client": client},
        ).fetchall()

    machine_guids  = [r.machine_guid for r in machine_rows]
    total_machines = len(machine_guids)
    batches        = [machine_guids[i: i + batch_size]
                      for i in range(0, total_machines, batch_size)]
    n_batches      = len(batches)
    print(f"[{_elapsed(t_total)}]  {total_machines} machines → {n_batches} batch(es) of ≤{batch_size}\n")

    all_monthly: dict = defaultdict(list)
    all_meta:    dict = {}
    batch_times: list[float] = []

    # ── 2. Batch queries ──────────────────────────────────────────────────────
    for i, batch in enumerate(batches, start=1):
        t_batch   = time.perf_counter()
        guid_str  = _guid_list(batch)

        with read_engine.connect() as conn:
            monthly_rows = conn.execute(sa.text(
                SQL_MONTHLY_BATCH.format(table=tbl, months=months, guids=guid_str)
            ), {"client": client}).fetchall()

            meta_rows = conn.execute(sa.text(
                SQL_META_BATCH.format(table=tbl, guids=guid_str)
            ), {"client": client}).fetchall()

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
        remaining = (sum(batch_times) / len(batch_times)) * (n_batches - i)
        pct       = i / n_batches * 100
        print(
            f"[{_elapsed(t_total)}]  Batch {i:>{len(str(n_batches))}}/{n_batches} "
            f"({pct:5.1f}%)  {len(batch)} machines  "
            f"{elapsed_batch:.1f}s/batch  ETA {_fmt_seconds(remaining)}"
        )

    # ── 3. Compute ────────────────────────────────────────────────────────────
    print(f"\n[{_elapsed(t_total)}]  Computing smart alarm thresholds...")
    results = compute_smart_alarms(all_monthly, all_meta, warning_mult, alarm_mult, run_ts)

    # ── 4. Write back ─────────────────────────────────────────────────────────
    if write_db:
        write_engine = sa.create_engine(_make_db_url(write_cfg))
        write_results_to_db(write_engine, write_cfg, results, t_total)
    else:
        print(f"[{_elapsed(t_total)}]  Skipping database write (--no-db-write).")

    print(f"[{_elapsed(t_total)}]  Done.  Total runtime: {_elapsed(t_total)}\n")
    return results


def write_results_to_db(engine, write_cfg: dict, results: list[dict], t0: float) -> None:
    """Creates the output table if needed, then merges all results."""
    import sqlalchemy as sa

    schema       = write_cfg["schema"]
    output_table = write_cfg["output_table"]

    print(f"[{_elapsed(t0)}]  Writing {len(results)} rows to [{schema}].[{output_table}]...")

    create_sql = SQL_CREATE_OUTPUT_TABLE.format(schema=schema, output_table=output_table)
    merge_sql  = sa.text(SQL_MERGE.format(schema=schema, output_table=output_table))

    with engine.begin() as conn:
        conn.execute(sa.text(create_sql))
        for row in results:
            conn.execute(merge_sql, {
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
                "stable":              1 if row.get("stable") else (0 if row.get("stable") is False else None),
                "stability_note":      row.get("stability_note"),
                "slope_per_month_pct": row.get("slope_per_month_pct"),
                "trend_p_value":       row.get("trend_p_value"),
                "months_of_data":      row.get("months_of_data"),
                "total_readings":      row.get("total_readings"),
                "computed_at":         row.get("computed_at"),
            })

    print(f"[{_elapsed(t0)}]  Database write complete.")


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
    print(f"Demo mode — building synthetic {months}-month history from {csv_path}")
    print(f"Client filter: '{client}'\n")
    random.seed(42)
    t_total = time.perf_counter()
    run_ts  = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

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

    now           = datetime.now(tz=timezone.utc)
    machine_guids = list({k[0] for k in snapshots})
    batches       = [machine_guids[i: i + batch_size]
                     for i in range(0, len(machine_guids), batch_size)]
    n_batches     = len(batches)
    print(f"  {len(machine_guids)} machines → {n_batches} batch(es) of ≤{batch_size}\n")

    all_monthly: dict = defaultdict(list)
    all_meta:    dict = {}
    batch_times: list[float] = []

    for b_idx, batch in enumerate(batches, start=1):
        t_batch   = time.perf_counter()
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
        remaining = (sum(batch_times) / len(batch_times)) * (n_batches - b_idx)
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
    "client", "area", "machine_guid", "machine", "component", "bearing",
    "point", "parameter", "type", "unit", "test_point_name",
    "iso_pre_alarm", "iso_alarm", "iso_danger",
    "baseline_avg", "smart_warning", "smart_alarm",
    "stable", "stability_note", "slope_per_month_pct", "trend_p_value",
    "months_of_data", "total_readings", "computed_at",
]


def write_csv(results: list[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def print_summary(results: list[dict]) -> None:
    total = len(results)
    if total == 0:
        print("No results.")
        return
    has_smart    = sum(1 for r in results if r["smart_warning"] is not None)
    unstable     = sum(1 for r in results if r["stable"] is False)
    insufficient = sum(1 for r in results if r["stable"] is None)

    print(f"{'='*65}")
    print(f"  Smart Alarm Summary")
    print(f"{'='*65}")
    print(f"  Total measurement points   : {total}")
    print(f"  Smart alarms computed      : {has_smart}  ({has_smart/total*100:.0f}%)")
    print(f"  Unstable — ISO kept active : {unstable}")
    print(f"  Insufficient data (skipped): {insufficient}")
    print(f"{'='*65}\n")

    for r in [r for r in results if r["smart_warning"] is not None][:6]:
        iso_a = r["iso_alarm"] if r["iso_alarm"] is not None else "N/A"
        print(
            f"  {str(r['point']):12s}  {str(r['parameter']):26s}"
            f"  avg={r['baseline_avg']:.4f} {str(r['unit']):5s}"
            f"  warn={r['smart_warning']:.4f}  alarm={r['smart_alarm']:.4f}"
            f"  (ISO={iso_a})"
        )

    bad = [r for r in results if r["stable"] is False][:3]
    if bad:
        print("\n  Unstable (ISO alarms remain active):")
        for r in bad:
            print(
                f"  {str(r['point']):12s}  {str(r['parameter']):26s}"
                f"  slope={r['slope_per_month_pct']:+.1f}%/mo  p={r['trend_p_value']:.3f}"
            )
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute smart vibration alarm thresholds (SQL Server)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--test",        action="store_true",
                        help="Test DB connections then exit")
    parser.add_argument("--demo",        action="store_true",
                        help="Demo mode: sample CSV with synthetic history, no DB needed")
    parser.add_argument("--csv",         default="falcon_scalars_example_data_1000_rows.csv",
                        help="Sample CSV path (demo mode only)")
    parser.add_argument("--client",      default=DEFAULT_CLIENT)
    parser.add_argument("--months",      type=int,   default=DEFAULT_MONTHS)
    parser.add_argument("--warning",     type=float, default=DEFAULT_WARNING_MULT)
    parser.add_argument("--alarm",       type=float, default=DEFAULT_ALARM_MULT)
    parser.add_argument("--batch-size",  type=int,   default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--out",         default="smart_alarm_results.csv")
    parser.add_argument("--no-db-write", action="store_true",
                        help="Skip writing to the database (CSV output only)")
    args = parser.parse_args()

    if args.test:
        ok = test_connections(READ_CONFIG, WRITE_CONFIG)
        sys.exit(0 if ok else 1)

    if args.demo:
        results = run_demo_mode(
            args.csv, args.client, args.months,
            args.warning, args.alarm, args.batch_size,
        )
    else:
        if not READ_CONFIG["host"]:
            sys.exit("READ_DB_HOST is not set. Copy .env.example to .env and fill in your details.")
        results = run_db_mode(
            READ_CONFIG, WRITE_CONFIG, args.client, args.months,
            args.warning, args.alarm, args.batch_size,
            write_db=not args.no_db_write,
        )

    print_summary(results)
    write_csv(results, args.out)
    print(f"Results written to: {args.out}")


if __name__ == "__main__":
    main()
