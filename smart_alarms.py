#!/usr/bin/env python3
"""
Smart Vibration Alarms
======================
Calculates data-driven alarm thresholds from 6-month rolling baseline.

Logic:
  - Baseline = mean of monthly averages over the last 6 months
  - Warning  = baseline × 1.3
  - Alarm    = baseline × 1.5
  - If the baseline is trending upward (significant positive slope), the
    smart alarm is suppressed — ISO alarms remain the active limit.

Usage:
  # Against a real database:
  DATABASE_URL=postgresql://user:pass@host/db python3 smart_alarms.py

  # Demo mode (generates synthetic history from the sample CSV):
  python3 smart_alarms.py --demo

  # Override lookback window and multipliers:
  python3 smart_alarms.py --demo --months 6 --warning 1.3 --alarm 1.5

Output: smart_alarm_results.csv
"""

import argparse
import csv
import os
import random
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from scipy import stats as _scipy_stats

# ── constants ────────────────────────────────────────────────────────────────

DEFAULT_MONTHS       = 6
DEFAULT_WARNING_MULT = 1.3
DEFAULT_ALARM_MULT   = 1.5
MIN_MONTHS_REQUIRED  = 3   # minimum distinct months needed to compute a baseline
VALID_STATUS_CODES   = {2, 3, 4, 5}  # Ok, PreAlarm, Alarm, Danger (exclude Unknown=1)

# Trend thresholds (applied together — both must be true to flag as unstable)
SLOPE_THRESHOLD = 0.05   # normalised slope > 5 % per month = unstable
P_VALUE_ALPHA   = 0.10   # trend must be statistically significant at this level


# ── stability check ───────────────────────────────────────────────────────────

def is_stable(monthly_values: list[float]) -> tuple[bool, float | None, float | None]:
    """
    Returns (stable, slope_normalised, p_value).
    Stable means no significant upward trend detected.
    """
    n = len(monthly_values)
    if n < 2:
        return False, None, None

    mean_val = sum(monthly_values) / n
    if mean_val <= 0:
        return True, 0.0, 1.0

    x = list(range(n))
    result = _scipy_stats.linregress(x, monthly_values)
    slope_norm = result.slope / mean_val  # fraction of mean per month
    p_value    = result.pvalue

    trending_up = (p_value < P_VALUE_ALPHA) and (slope_norm > SLOPE_THRESHOLD)
    return not trending_up, round(slope_norm, 5), round(p_value, 5)


# ── smart alarm calculation ───────────────────────────────────────────────────

def compute_smart_alarms(
    monthly_data: dict,         # {point_key: [(month_str, avg_value, n_readings), ...]}
    latest_iso: dict,           # {point_key: {pal_plus, al_plus, dg_plus, unit, ...}}
    warning_mult: float,
    alarm_mult: float,
) -> list[dict]:
    """
    Core calculation. Returns list of result dicts, one per measurement point.
    monthly_data values are sorted oldest→newest.
    """
    results = []

    for key, months_list in monthly_data.items():
        months_list = sorted(months_list, key=lambda r: r[0])
        values  = [r[1] for r in months_list]
        n_months = len(values)
        total_readings = sum(r[2] for r in months_list)
        meta = latest_iso.get(key, {})

        base = {
            "machine_guid": key[0],
            "point":        key[1],
            "parameter":    key[2],
            "type":         key[3],
            "unit":         meta.get("unit", ""),
            "machine":      meta.get("machine", ""),
            "component":    meta.get("component", ""),
            "bearing":      meta.get("bearing", ""),
            "months_of_data":  n_months,
            "total_readings":  total_readings,
            "baseline_avg":    None,
            "smart_warning":   None,
            "smart_alarm":     None,
            "stable":          None,
            "stability_note":  "",
            "slope_per_month_pct": None,
            "trend_p_value":   None,
            "iso_pre_alarm":   meta.get("pal_plus"),
            "iso_alarm":       meta.get("al_plus"),
            "iso_danger":      meta.get("dg_plus"),
        }

        if n_months < MIN_MONTHS_REQUIRED:
            base["stability_note"] = (
                f"Insufficient data: {n_months} months (need {MIN_MONTHS_REQUIRED})"
            )
            results.append(base)
            continue

        stable, slope_norm, p_val = is_stable(values)
        baseline_avg = sum(values) / len(values)

        base["baseline_avg"]         = round(baseline_avg, 6)
        base["stable"]               = stable
        base["slope_per_month_pct"]  = round(slope_norm * 100, 2) if slope_norm is not None else None
        base["trend_p_value"]        = p_val

        if stable:
            base["smart_warning"] = round(baseline_avg * warning_mult, 4)
            base["smart_alarm"]   = round(baseline_avg * alarm_mult, 4)
            base["stability_note"] = "Stable baseline"
        else:
            direction = "increasing" if (slope_norm or 0) > 0 else "decreasing"
            base["stability_note"] = (
                f"Unstable ({direction} trend, "
                f"slope={slope_norm*100:.1f}%/month, p={p_val:.3f}) — "
                "ISO alarms remain active"
            )

        results.append(base)

    return results


# ── database mode ─────────────────────────────────────────────────────────────

SQL_MONTHLY = """
SELECT
    machine_guid,
    point,
    parameter,
    type,
    TO_CHAR(DATE_TRUNC('month', date_meas), 'YYYY-MM') AS month,
    AVG(value::float)   AS avg_value,
    COUNT(*)            AS reading_count
FROM falcon_scalars
WHERE
    date_meas   >= NOW() - INTERVAL '{months} months'
    AND date_meas < NOW()
    AND status_code::int IN (2, 3, 4, 5)
    AND value IS NOT NULL
    AND value::float > 0
GROUP BY machine_guid, point, parameter, type,
         DATE_TRUNC('month', date_meas)
ORDER BY machine_guid, point, parameter, type, month
"""

SQL_LATEST_ISO = """
SELECT DISTINCT ON (machine_guid, point, parameter, type)
    machine_guid,
    point,
    parameter,
    type,
    unit,
    machine,
    component,
    bearing,
    pal_plus,
    al_plus,
    dg_plus
FROM falcon_scalars
WHERE date_meas >= NOW() - INTERVAL '30 days'
ORDER BY machine_guid, point, parameter, type, date_meas DESC
"""


def run_db_mode(db_url: str, months: int, warning_mult: float, alarm_mult: float) -> list[dict]:
    try:
        import sqlalchemy as sa
    except ImportError:
        sys.exit("sqlalchemy is required for database mode: pip install sqlalchemy")

    engine = sa.create_engine(db_url)
    with engine.connect() as conn:
        print(f"Querying monthly aggregates for last {months} months...")
        monthly_rows = conn.execute(sa.text(SQL_MONTHLY.format(months=months))).fetchall()

        print("Querying latest ISO alarm levels...")
        iso_rows = conn.execute(sa.text(SQL_LATEST_ISO)).fetchall()

    monthly_data: dict = defaultdict(list)
    for row in monthly_rows:
        key = (row.machine_guid, row.point, row.parameter, row.type)
        monthly_data[key].append((row.month, float(row.avg_value), int(row.reading_count)))

    latest_iso: dict = {}
    for row in iso_rows:
        key = (row.machine_guid, row.point, row.parameter, row.type)
        latest_iso[key] = {
            "unit":      row.unit,
            "machine":   row.machine,
            "component": row.component,
            "bearing":   row.bearing,
            "pal_plus":  row.pal_plus,
            "al_plus":   row.al_plus,
            "dg_plus":   row.dg_plus,
        }

    return compute_smart_alarms(monthly_data, latest_iso, warning_mult, alarm_mult)


# ── demo mode ─────────────────────────────────────────────────────────────────

def _parse_float(v):
    try:
        return float(v) if v and v.upper() != "NULL" else None
    except ValueError:
        return None


def run_demo_mode(csv_path: str, months: int, warning_mult: float, alarm_mult: float) -> list[dict]:
    """
    Builds synthetic 6-month monthly history from a single-snapshot CSV.
    Each unique (machine_guid, point, parameter, type) gets realistic month-
    by-month variation so the stability and threshold logic can be exercised.
    """
    print(f"Demo mode: building synthetic {months}-month history from {csv_path}")
    random.seed(42)

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    # Collect latest snapshot per unique point
    snapshots: dict = {}
    for row in rows:
        if row.get("status_code") not in {"2", "3", "4", "5"}:
            continue
        val = _parse_float(row.get("value"))
        if val is None or val <= 0:
            continue
        key = (row["machine_guid"], row["point"], row["parameter"], row["type"])
        snapshots[key] = row  # last row wins (doesn't matter for demo)

    # Assign behaviours: ~70% stable, ~20% slow-riser, ~10% erratic
    keys  = list(snapshots.keys())
    n     = len(keys)
    random.shuffle(keys)
    stable_keys  = set(keys[: int(n * 0.70)])
    rising_keys  = set(keys[int(n * 0.70): int(n * 0.90)])
    # remaining → erratic / insufficient data

    now   = datetime.now(tz=timezone.utc)
    monthly_data: dict = defaultdict(list)

    for key, row in snapshots.items():
        base_val = _parse_float(row["value"])
        if base_val is None or base_val <= 0:
            continue

        if key in stable_keys:
            n_months   = months
            trend_frac = 0.0      # no trend
            noise_frac = 0.08     # ±8% random noise
        elif key in rising_keys:
            n_months   = months
            trend_frac = 0.08     # +8% per month → clearly unstable
            noise_frac = 0.05
        else:
            n_months   = random.randint(1, MIN_MONTHS_REQUIRED - 1)  # too few months
            trend_frac = 0.0
            noise_frac = 0.12

        for i in range(n_months):
            m_offset  = n_months - 1 - i        # 0 = most recent month
            month_str = (now - timedelta(days=30 * m_offset)).strftime("%Y-%m")
            trend_val = base_val * (1 + trend_frac * i)
            noise     = trend_val * noise_frac * (random.random() * 2 - 1)
            avg_val   = max(trend_val + noise, 0.0001)
            n_readings = random.randint(3, 12)
            monthly_data[key].append((month_str, avg_val, n_readings))

    latest_iso: dict = {}
    for key, row in snapshots.items():
        latest_iso[key] = {
            "unit":      row.get("unit", ""),
            "machine":   row.get("machine", ""),
            "component": row.get("component", ""),
            "bearing":   row.get("bearing", ""),
            "pal_plus":  _parse_float(row.get("pal_plus")),
            "al_plus":   _parse_float(row.get("al_plus")),
            "dg_plus":   _parse_float(row.get("dg_plus")),
        }

    return compute_smart_alarms(monthly_data, latest_iso, warning_mult, alarm_mult)


# ── output ────────────────────────────────────────────────────────────────────

OUTPUT_FIELDS = [
    "machine_guid", "machine", "component", "bearing", "point",
    "parameter", "type", "unit",
    "months_of_data", "total_readings",
    "baseline_avg",
    "smart_warning", "smart_alarm",
    "stable", "stability_note",
    "slope_per_month_pct", "trend_p_value",
    "iso_pre_alarm", "iso_alarm", "iso_danger",
]


def write_csv(results: list[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def print_summary(results: list[dict]) -> None:
    total        = len(results)
    has_smart    = sum(1 for r in results if r["smart_warning"] is not None)
    unstable     = sum(1 for r in results if r["stable"] is False)
    insufficient = sum(1 for r in results if r["stable"] is None)

    print(f"\n{'='*60}")
    print(f"  Smart Alarm Results")
    print(f"{'='*60}")
    print(f"  Total measurement points : {total}")
    print(f"  Smart alarms computed    : {has_smart}  ({has_smart/total*100:.0f}%)")
    print(f"  Unstable (suppressed)    : {unstable}")
    print(f"  Insufficient data        : {insufficient}")
    print(f"{'='*60}\n")

    # Show a handful of examples
    sample = [r for r in results if r["smart_warning"] is not None][:5]
    if sample:
        print("  Sample stable points with smart alarms:")
        for r in sample:
            iso_a = r["iso_alarm"] or "N/A"
            print(
                f"    {r['point']:12s} | {r['parameter']:25s} | "
                f"avg={r['baseline_avg']:.4f} {r['unit']:5s} | "
                f"warn={r['smart_warning']:.4f}  alarm={r['smart_alarm']:.4f}  "
                f"(ISO alert={iso_a})"
            )

    unstable_sample = [r for r in results if r["stable"] is False][:3]
    if unstable_sample:
        print("\n  Sample unstable points (ISO alarms remain active):")
        for r in unstable_sample:
            print(
                f"    {r['point']:12s} | {r['parameter']:25s} | "
                f"slope={r['slope_per_month_pct']:+.1f}%/mo  p={r['trend_p_value']:.3f}"
            )
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compute smart vibration alarm thresholds")
    parser.add_argument("--demo",    action="store_true",
                        help="Run in demo mode using the sample CSV")
    parser.add_argument("--csv",     default="falcon_scalars_example_data_1000_rows.csv",
                        help="Path to sample CSV (demo mode only)")
    parser.add_argument("--months",  type=int,   default=DEFAULT_MONTHS,
                        help="Lookback window in months (default 6)")
    parser.add_argument("--warning", type=float, default=DEFAULT_WARNING_MULT,
                        help="Warning multiplier (default 1.3)")
    parser.add_argument("--alarm",   type=float, default=DEFAULT_ALARM_MULT,
                        help="Alarm multiplier (default 1.5)")
    parser.add_argument("--out",     default="smart_alarm_results.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    db_url = os.environ.get("DATABASE_URL")

    if args.demo or not db_url:
        if not args.demo and not db_url:
            print("No DATABASE_URL set — running in demo mode.")
            print("Set DATABASE_URL=postgresql://... to run against your database.\n")
        results = run_demo_mode(args.csv, args.months, args.warning, args.alarm)
    else:
        results = run_db_mode(db_url, args.months, args.warning, args.alarm)

    print_summary(results)
    write_csv(results, args.out)
    print(f"Results written to: {args.out}")


if __name__ == "__main__":
    main()
