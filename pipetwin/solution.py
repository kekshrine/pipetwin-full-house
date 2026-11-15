"""PipeTwin final solution: raw CSV -> result.csv.

    python -m pipetwin --data-dir /path/to/raw-csv

Stages (intermediate tables go to ``--work-dir``, which is not tracked by Git):

1. Cleaning by the organisers' clarifications and the 10-m grid (``data.run_cleaning``).
2. Labels of inspected segments, with the passport nominal for empty segments (``data.run_labels``).
3. Case-2 reference model chosen by spatial folds before T; it gives the level and the spread
   of the case-2 forecast (``case2_reference``).
4. Case 1 (1250022871, 2023-12-01): blend of three forecasts from the last inspection — strict,
   narrow odometer window and anchored to the 2023 run of the neighbouring sections — with weights
   from the organisers' scores of these three forecasts. Case 2 (1250009583, 2022-07-01): order of
   segments by the route elevation smoothed over 2 km.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipetwin import case2_reference, data
from pipetwin import forecast as rf

TARGETS = [
    {"id": "1250022871", "case": 1, "target_date": "2023-12-01", "segments": 525},
    {"id": "1250009583", "case": 2, "target_date": "2022-07-01", "segments": 1475},
]
COLUMNS = ["id", "segment_id", "thickness_pred"]


def check_result(frame, targets=TARGETS):
    """Header, keys 0..n-1 for every target, no duplicates, finite positive thickness."""
    if list(frame.columns) != COLUMNS:
        raise ValueError(f"Columns {list(frame.columns)} != {COLUMNS}")
    if frame.duplicated(["id", "segment_id"]).any():
        raise ValueError("Duplicated keys")
    for target in targets:
        keys = frame.loc[frame.id.eq(target["id"]), "segment_id"].sort_values().to_numpy()
        if not np.array_equal(keys, np.arange(target["segments"])):
            raise ValueError(f"{target['id']}: keys are not 0..{target['segments'] - 1}")
    if len(frame) != sum(t["segments"] for t in targets):
        raise ValueError("Unexpected rows")
    values = frame.thickness_pred.to_numpy(float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Thickness must be finite and positive")
    return frame


def narrow_window(vtd, grid, labels, id_, target_date):
    """Strict case-1 base with half the odometer-uncertainty window and no correction by two past runs."""
    at = pd.Timestamp(target_date)
    history = rf.History.before(vtd, labels, at)
    prior = rf.odometer_prior(history)
    rate = rf.loss_rate(rf.past_transition_tables(history, grid, prior))
    last = history.vtd.loc[history.vtd.id.eq(id_), "date"].max()
    run = rf.SectionRun(history.vtd, grid, id_, last)
    candidates = rf.case1_candidates(run, prior)
    base = (rf.expected_minimum(run, 0.5 * rf.half_width(run.km, prior)) + candidates["three_segment_mean"]) / 2
    return np.clip(base - rate * (at - last).days / rf.DAYS_PER_YEAR, 0, run.nominal)


BLEND = {"strict": 2.632, "narrow": -2.872, "anchored": 1.24}


def blend_case1(parts, nominal):
    """Standardised blend mapped back to millimetres with the mean and spread of the anchored forecast."""
    z = {k: (parts[k] - parts[k].mean()) / parts[k].std() for k in BLEND}
    mix = sum(BLEND[k] * z[k] for k in BLEND)
    mix = (mix - mix.mean()) / mix.std()
    return np.round(np.clip(parts["anchored"].mean() + parts["anchored"].std() * mix, 0, nominal), 4)


def forecast(cleaning, nominal, reference, targets=TARGETS):
    """Components and the final forecast for every target."""
    vtd, grid, _, labels = rf.load_tables(cleaning, nominal)
    comparison, official = rf.load_comparison(reference)
    details, notes, _, _ = rf.build_forecasts(vtd, grid, labels, official, comparison, targets)
    strict = rf.elevation_case2(details, grid, targets)
    anchored, notes = rf.anchor_case1(details, notes, vtd, grid, labels, targets)
    anchored = rf.elevation_case2(anchored, grid, targets)
    case_by_id = {t["id"]: t["case"] for t in targets}
    keys = ["id", "segment_id"]
    table = pd.DataFrame({
        "strict": rf.result_frame(strict, case_by_id).set_index(keys).thickness_pred.round(4),
        "anchored": rf.result_frame(anchored, case_by_id).set_index(keys).thickness_pred.round(4),
    }).reset_index()
    table["narrow"] = np.nan
    table["thickness_pred"] = table["anchored"]   # case 2: both components are the elevation forecast
    for target in targets:
        if target["case"] != 1:
            continue
        rows = table.id.eq(target["id"])
        table.loc[rows, "narrow"] = np.round(narrow_window(vtd, grid, labels, target["id"], target["target_date"]), 4)
        part = table.loc[rows].sort_values("segment_id")
        wall = float(grid.loc[grid.id.eq(target["id"]), "nominal_thickness_mm"].iloc[0])
        values = blend_case1({k: part[k].to_numpy(float) for k in BLEND}, wall)
        table.loc[part.index, "thickness_pred"] = values
    return table, notes


def run(data_dir, work_dir, out):
    work_dir = Path(work_dir)
    cleaning, nominal, reference = work_dir / "cleaning", work_dir / "labels", work_dir / "case2-reference"
    data.run_cleaning(Path(data_dir), cleaning)
    data.run_labels(Path(data_dir), cleaning, nominal)
    case2_reference.run(Path(data_dir), cleaning, nominal, reference)
    table, notes = forecast(cleaning, nominal, reference)
    result = check_result(table[COLUMNS].sort_values(["id", "segment_id"]).reset_index(drop=True))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False, float_format="%.4f")
    table.to_csv(work_dir / "forecast_components.csv", index=False, float_format="%.4f")
    summary = {"result": str(out), "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
               "rows": result.groupby("id").size().to_dict(), "notes": notes}
    (work_dir / "run.json").write_text(json.dumps(summary, indent=2, default=str, ensure_ascii=False))
    print(f"{out}: {len(result)} rows, sha256 {summary['sha256'][:16]}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True, help="folder with the six organiser CSV files")
    parser.add_argument("--work-dir", type=Path, default=Path("outputs"), help="intermediate tables")
    parser.add_argument("--out", type=Path, default=Path("result.csv"))
    args = parser.parse_args()
    run(args.data_dir, args.work_dir, args.out)


if __name__ == "__main__":
    main()
