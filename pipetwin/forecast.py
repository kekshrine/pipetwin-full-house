"""Case-1 forecast from past inspections (strict cut-off before T) and assembly of both cases."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
DAYS_PER_YEAR = 365.25
WALL = "Residual Wall Thickness"
ANOMALY = "recorded_anomaly_without_thickness"
MEASURED = "measured"
DEFAULT_PRIOR = {"launch_m": 5.0, "drift_m_per_km": 2.0, "pairs": 0}  # used only without a past run pair
OFFSET_WINDOW_M, OFFSET_STEP_M, OFFSET_MAX_LAG_M, OFFSET_MIN_R = 200, 50, 30, 0.45
EXPECTED = "expected_minimum"
BASE = "expected_minimum_with_three_mean"  # equal-weight mean of EXPECTED and the three-segment mean
FINAL = "history_and_growth"  # BASE corrected by the weld-aligned earlier run and matched-defect growth
JOINT_TOLERANCE_M, DEFECT_AXIAL_M, DEFECT_ANGLE_DEG = 1.0, 0.5, 35.0


@dataclass(frozen=True)
class History:
    """Inspection records strictly before ``cutoff``; the only data a forecast may see."""
    vtd: pd.DataFrame
    labels: pd.DataFrame
    cutoff: pd.Timestamp

    @classmethod
    def before(cls, vtd, labels, cutoff):
        cutoff = pd.Timestamp(cutoff)
        return cls(vtd.loc[vtd.date.lt(cutoff)].copy(), labels.loc[labels.inspection_date.lt(cutoff)].copy(), cutoff)

    def run_dates(self, main_id):
        return sorted(self.vtd.loc[self.vtd.main_id.eq(main_id), "date"].unique())

    def transitions(self):
        """Consecutive run pairs of every main, both strictly before the cutoff."""
        pairs = []
        for main_id in sorted(self.vtd.main_id.unique()):
            dates = self.run_dates(main_id)
            pairs.extend((main_id, pd.Timestamp(a), pd.Timestamp(b)) for a, b in zip(dates, dates[1:]))
        return pairs


def load_tables(cleaning, nominal):
    tables = Path(cleaning) / "tables"
    passport = pd.read_csv(tables / "passport.csv", dtype={"id": str}, parse_dates=["commissioned_at"])
    vtd = pd.read_csv(tables / "vtd.csv", low_memory=False, dtype={"id": str})
    vtd["date"] = pd.to_datetime(vtd["date"])
    vtd = vtd.merge(passport[["id", "main_id"]], on="id", validate="many_to_one")
    grid = pd.read_csv(tables / "grid.csv", dtype={"id": str}, parse_dates=["commissioned_at"])
    labels = pd.read_csv(Path(nominal) / "vtd_labels_full_grid.csv", dtype={"id": str},
                         parse_dates=["inspection_date"])
    return vtd, grid, passport, labels


def load_comparison(folder):
    path = Path(folder) / "predictions_historical.csv"
    if not path.exists():
        return None, None
    history = pd.read_csv(path, dtype={"id": str}, low_memory=False, parse_dates=["target_date"])
    official = pd.read_csv(Path(folder) / "official_forecasts.csv", dtype={"id": str})
    return history, official


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() < 1e-9 or y.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def known_minima(labels, id_, date):
    at = labels.loc[labels.id.eq(id_) & labels.inspection_date.eq(date)]
    at = at.loc[at.thickness_origin.ne(ANOMALY) & at.min_residual_thickness_mm.notna()]
    return at.set_index("segment_id")[["min_residual_thickness_mm", "thickness_origin"]]


class SectionRun:
    """Defects of one run over the whole main in the global coordinate, viewed from one section."""

    def __init__(self, vtd, grid, id_, date):
        section = grid.loc[grid.id.eq(id_)].sort_values("segment_id")
        own = vtd.loc[vtd.id.eq(id_)]
        run = vtd.loc[vtd.main_id.eq(own["main_id"].iloc[0]) & vtd.date.eq(date)]
        self.keys = section[["id", "segment_id"]].reset_index(drop=True)
        self.n = len(section)
        self.nominal = float(section["nominal_thickness_mm"].iloc[0])
        self.length = float(section["section_length_m"].iloc[0])
        self.start = float(own["accumulated_value"].iloc[0])
        self.global_m = run["Distance_adjusted"].to_numpy(float) + run["accumulated_value"].to_numpy(float)
        self.wall = run[WALL].to_numpy(float)
        self.km = (self.start + 10.0 * np.arange(self.n) + 5.0) / 1000.0

    def minima(self, shift_m=0.0):
        """10-m minima after moving every defect by ``shift_m``; empty segments keep the nominal."""
        local = self.global_m + shift_m - self.start
        inside = (local >= 0) & (local < self.length)
        index = np.floor(local[inside] / 10.0).astype(int)
        index = np.minimum(index, self.n - 1)
        out = np.full(self.n, self.nominal)
        np.minimum.at(out, index, self.wall[inside])
        return out


def density_profile(global_m, total_m):
    index = np.floor(global_m).astype(int)
    index = index[(index >= 0) & (index < total_m)]
    return np.log1p(np.bincount(index, minlength=total_m).astype(float))


def measure_offsets(prev_global, cur_global):
    """Local offset between two runs from the best lag of their defect-density profiles."""
    total = int(np.ceil(max(prev_global.max(), cur_global.max()))) + 1
    a, b = density_profile(prev_global, total), density_profile(cur_global, total)
    rows = []
    for start in range(0, total - OFFSET_WINDOW_M, OFFSET_STEP_M):
        x = a[start:start + OFFSET_WINDOW_M]
        if x.std() < 1e-9:
            continue
        best = (np.nan, -np.inf)
        for lag in range(-OFFSET_MAX_LAG_M, OFFSET_MAX_LAG_M + 1):
            low, high = start + lag, start + lag + OFFSET_WINDOW_M
            if low < 0 or high > total or b[low:high].std() < 1e-9:
                continue
            r = np.corrcoef(x, b[low:high])[0, 1]
            if r > best[1]:
                best = (lag, r)
        if best[1] >= OFFSET_MIN_R and abs(best[0]) < OFFSET_MAX_LAG_M:
            rows.append({"km": (start + OFFSET_WINDOW_M / 2) / 1000.0, "offset_m": best[0], "r": best[1]})
    return pd.DataFrame(rows, columns=["km", "offset_m", "r"])


def odometer_prior(history):
    """Launch mismatch and scale drift of the odometer, root-mean-square over past run pairs."""
    fits = []
    for main_id, prev_date, cur_date in history.transitions():
        runs = history.vtd.loc[history.vtd.main_id.eq(main_id)]
        both = set(runs.loc[runs.date.eq(prev_date), "id"]) & set(runs.loc[runs.date.eq(cur_date), "id"])
        runs = runs.loc[runs.id.isin(both)]
        position = runs["Distance_adjusted"].to_numpy(float) + runs["accumulated_value"].to_numpy(float)
        offsets = measure_offsets(position[runs.date.eq(prev_date).to_numpy()], position[runs.date.eq(cur_date).to_numpy()])
        if len(offsets) < 5 or np.ptp(offsets["km"]) < 0.5:
            continue
        slope, intercept = np.polyfit(offsets["km"], offsets["offset_m"], 1)
        fits.append({"main_id": main_id, "prev": str(prev_date.date()), "cur": str(cur_date.date()),
                     "windows": len(offsets), "launch_m": float(intercept), "drift_m_per_km": float(slope)})
    if not fits:
        return dict(DEFAULT_PRIOR, fits=[])
    return {"launch_m": float(np.sqrt(np.mean([f["launch_m"] ** 2 for f in fits]))),
            "drift_m_per_km": float(np.sqrt(np.mean([f["drift_m_per_km"] ** 2 for f in fits]))),
            "pairs": len(fits), "fits": fits}


def half_width(km, prior):
    """Half-width of the uniform position uncertainty with the same variance as the prior."""
    return np.sqrt(3.0) * np.sqrt(prior["launch_m"] ** 2 + (prior["drift_m_per_km"] * np.asarray(km)) ** 2)


def expected_minimum(run, half_width_m):
    """Mean of the 10-m minimum over all whole-metre window positions within the uncertainty."""
    width = np.maximum(np.rint(np.broadcast_to(half_width_m, run.n)).astype(int), 0)
    total, count = np.zeros(run.n), np.zeros(run.n)
    for shift in range(-int(width.max()), int(width.max()) + 1):
        use = width >= abs(shift)
        total[use] += run.minima(float(shift))[use]
        count[use] += 1
    return total / count


def three_segment_mean(run):
    padded = np.pad(run.minima(), 1, mode="edge")
    return (padded[:-2] + padded[1:-1] + padded[2:]) / 3.0


def final_forecast(run, prior):
    """Equal-weight mean of the expected minimum and the three-segment mean; no fitted weight."""
    return (expected_minimum(run, half_width(run.km, prior)) + three_segment_mean(run)) / 2.0


def case1_candidates(run, prior):
    expected = expected_minimum(run, half_width(run.km, prior))
    three = three_segment_mean(run)
    return {
        "segment_minimum": run.minima(),
        "three_segment_mean": three,
        "uniform_10m": expected_minimum(run, 10.0),
        EXPECTED: expected,
        BASE: (expected + three) / 2.0,
    }


def history_and_growth(history, grid, id_, run, base):
    """Correct ``base`` with the two latest runs of the section before the cutoff.

    The earlier run is moved onto the latest run's coordinates through matched welds
    and launch/receipt chambers (both runs are dated before the cutoff). Its slow
    spatial profile is added with weight 0.25 where the latest run is sparse, and the
    loss rate of defect pairs matched inside paired joints (median weighted by each
    pair's position reliability, exp(-d2/2) in the tolerance ellipse), as a deviation from
    the pooled rate within 30 m, is extrapolated to the cutoff (limited to +-0.2 mm).
    With fewer than two runs or no anchors the base is returned unchanged.
    """
    from copy import copy


    dates = sorted(history.vtd.loc[history.vtd.id.eq(id_), "date"].unique())
    needed = set(GEOMETRY) | {"side_normalized", "Distance_raw"}
    if len(dates) < 2 or not needed.issubset(history.vtd.columns):
        return base
    old, last = pd.Timestamp(dates[-2]), pd.Timestamp(dates[-1])
    main_id = history.vtd.loc[history.vtd.id.eq(id_), "main_id"].iloc[0]
    route = history.vtd.loc[history.vtd.main_id.eq(main_id)]
    anchors = shift_between_old_runs(route, old, last)
    if not len(anchors):
        return base
    aligned_run = copy(SectionRun(history.vtd, grid, id_, old))
    aligned_run.global_m = old_to_last(aligned_run.global_m, anchors)
    aligned_run.start = run.start
    _, aligned = previous_background(run, aligned_run, base)
    by_date = {pd.Timestamp(day): part for day, part in route.groupby("date")}
    joints = pair_joints(by_date[old], by_date[last], anchors, JOINT_TOLERANCE_M)
    measured = matched_growth(by_date[old], by_date[last], joints, (last - old).days / DAYS_PER_YEAR,
                                 DEFECT_AXIAL_M, DEFECT_ANGLE_DEG, weighted=True)
    horizon = (history.cutoff - last).days / DAYS_PER_YEAR
    return aligned - growth_correction(run, measured, horizon)


def transition_table(vtd, grid, labels, main_id, prev_date, cur_date, prior):
    """Forecast candidates from the ``prev_date`` run next to the known minima of ``cur_date``."""
    frames = []
    ids = sorted(set(vtd.loc[vtd.main_id.eq(main_id) & vtd.date.eq(prev_date), "id"])
                 & set(labels.loc[labels.inspection_date.eq(cur_date), "id"]))
    before = History.before(vtd, labels, cur_date)
    for id_ in ids:
        truth = known_minima(labels, id_, cur_date)
        if truth.empty:
            continue
        run = SectionRun(vtd, grid, id_, prev_date)
        parts = case1_candidates(run, prior)
        parts[FINAL] = history_and_growth(before, grid, id_, run, parts[BASE])
        parts = pd.DataFrame(parts).loc[truth.index]
        frames.append(parts.assign(
            id=id_, segment_id=truth.index, true=truth["min_residual_thickness_mm"].to_numpy(),
            measured=truth["thickness_origin"].eq(MEASURED).to_numpy(),
            years=(cur_date - prev_date).days / DAYS_PER_YEAR))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def past_transition_tables(history, grid, prior):
    tables = [transition_table(history.vtd, grid, history.labels, main_id, prev, cur, prior)
              .assign(main_id=main_id, start_date=prev, target_date=cur)
              for main_id, prev, cur in history.transitions()]
    tables = [t for t in tables if len(t)]
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def loss_rate(past, column=BASE):
    """Wall loss per year: minus the median residual of the forecast on past transitions."""
    if past.empty:
        return 0.0
    return float(max(0.0, -((past["true"] - past[column]) / past["years"]).median()))


def forecast_case1(history, grid, id_, target_date):
    """Forecast of one section; reads nothing dated ``target_date`` or later."""
    if history.cutoff > pd.Timestamp(target_date):
        raise ValueError("History extends beyond the forecast date")
    dates = history.vtd.loc[history.vtd.id.eq(id_), "date"]
    if dates.empty:
        return None
    prev_date = dates.max()
    prior = odometer_prior(history)
    past = past_transition_tables(history, grid, prior)
    rate = loss_rate(past)
    run = SectionRun(history.vtd, grid, id_, prev_date)
    years = (pd.Timestamp(target_date) - prev_date).days / DAYS_PER_YEAR
    base = final_forecast(run, prior)
    raw = history_and_growth(history, grid, id_, run, base)
    frame = run.keys.assign(
        thickness_pred=np.clip(raw - rate * years, 0.0, run.nominal), final_raw_mm=raw, base_raw_mm=base,
        expected_minimum_mm=expected_minimum(run, half_width(run.km, prior)),
        last_minimum_mm=run.minima(), half_width_m=half_width(run.km, prior), loss_mm=rate * years,
        start_date=prev_date, target_date=pd.Timestamp(target_date), source=FINAL)
    return frame, {"prior": prior, "loss_rate_mm_per_year": rate, "horizon_years": years,
                   "last_run": str(prev_date.date()), "past_keys": int(len(past))}, past


def case2_selection(comparison, cutoff):
    """Score every comparison model on spatial folds of first inspections before the cut-off.

    The winner maximises the mean of Pearson r on all known labels and on measured
    labels. Later first inspections are reported next to it and never used to choose.
    """
    if comparison is None:
        return pd.DataFrame()
    rows = comparison.loc[comparison.case.eq(2) & comparison.min_residual_thickness_mm.notna()
                          & comparison.thickness_origin.ne(ANOMALY)]
    out = []
    for (policy, model), part in rows.groupby(["training_policy", "model"]):
        record = {"policy": policy, "model": model}
        for name, phase, before in [("before_cutoff", "blocked_oof", True), ("later_holdout", "chronological", False)]:
            split = part.loc[part.phase.eq(phase)]
            split = split.loc[split.target_date.lt(cutoff) if before else split.target_date.ge(cutoff)]
            measured = split.loc[split.thickness_origin.eq(MEASURED)]
            record.update({
                f"{name}_keys": len(split), f"{name}_r_known": pearson(split.thickness_pred, split.min_residual_thickness_mm),
                f"{name}_r_measured": pearson(measured.thickness_pred, measured.min_residual_thickness_mm),
                f"{name}_mae_known_mm": float((split.thickness_pred - split.min_residual_thickness_mm).abs().mean())})
        record["selection_score"] = np.mean([record["before_cutoff_r_known"], record["before_cutoff_r_measured"]])
        out.append(record)
    table = pd.DataFrame(out).sort_values("selection_score", ascending=False, na_position="last")
    table["selected"] = False
    if table["selection_score"].notna().any():
        table.iloc[0, table.columns.get_loc("selected")] = True
    return table.reset_index(drop=True)


def forecast_case2(history, grid, official, selection, id_, target_date):
    """Selected comparison model when it covers the target, else the terrain fallback."""
    section = grid.loc[grid.id.eq(id_)].sort_values("segment_id")
    nominal = float(section["nominal_thickness_mm"].iloc[0])
    keys = section[["id", "segment_id"]].reset_index(drop=True)
    start = section["commissioned_at"].iloc[0]
    if official is not None and len(selection) and selection["selected"].any():
        best = selection.loc[selection.selected].iloc[0]
        rows = official.loc[official.id.eq(id_) & official.case.eq(2) & official.training_policy.eq(best.policy)
                            & official.model.eq(best.model)
                            & pd.to_datetime(official.target_date).eq(pd.Timestamp(target_date))]
        if len(rows) == len(keys):
            pred = rows.set_index("segment_id")["thickness_pred"].reindex(keys["segment_id"]).to_numpy()
            return keys.assign(thickness_pred=np.clip(pred, 0.0, nominal), start_date=start,
                               target_date=pd.Timestamp(target_date), source=f"{best.policy}:{best.model}")
    return terrain_fallback(history, grid, id_, target_date)


def terrain_features(section):
    z = pd.Series(section["elevation_mid_m"].to_numpy(float)).interpolate(limit_direction="both")
    values = z.to_numpy()
    out = pd.DataFrame(index=section.index)
    for window, name in [(5, "50"), (20, "200"), (50, "500")]:
        out["above_local_min_" + name] = values - z.rolling(2 * window + 1, center=True, min_periods=1).min().to_numpy()
    gradient = np.gradient(values) if len(values) > 1 else np.zeros_like(values)
    out["uphill"], out["downhill"] = np.clip(gradient, 0, None), np.clip(-gradient, 0, None)
    return out


def terrain_fallback(history, grid, id_, target_date, l2=30.0):
    """Ridge of the annual loss fraction on route profile, from first inspections before the cut-off.

    Used only when the comparison does not cover the target. Its sign is not
    validated; it exists because a constant forecast has no Pearson correlation.
    """
    section = grid.loc[grid.id.eq(id_)].sort_values("segment_id").reset_index(drop=True)
    nominal = float(section["nominal_thickness_mm"].iloc[0])
    start = section["commissioned_at"].iloc[0]
    years = (pd.Timestamp(target_date) - start).days / DAYS_PER_YEAR
    first = history.labels.groupby("id")["inspection_date"].transform("min")
    labels = history.labels.loc[history.labels.inspection_date.eq(first) & history.labels.id.ne(id_)
                                & history.labels.thickness_origin.ne(ANOMALY)
                                & history.labels.min_residual_thickness_mm.notna()]
    rows = []
    for other, part in labels.groupby("id"):
        donor = grid.loc[grid.id.eq(other)].sort_values("segment_id").reset_index(drop=True)
        age = (part.inspection_date.iloc[0] - donor["commissioned_at"].iloc[0]).days / DAYS_PER_YEAR
        if age <= 0:
            continue
        feats = terrain_features(donor).loc[part.segment_id.to_numpy()]
        rate = (1 - part.min_residual_thickness_mm.to_numpy() / float(donor["nominal_thickness_mm"].iloc[0])) / age
        rows.append(feats.assign(rate=rate))
    keys = section[["id", "segment_id"]]
    if not rows:
        return keys.assign(thickness_pred=nominal, start_date=start, target_date=pd.Timestamp(target_date),
                           source="nominal_no_history")
    train = pd.concat(rows, ignore_index=True)
    columns = [c for c in train.columns if c != "rate"]
    mean, scale = train[columns].mean(), train[columns].std().replace(0, 1.0)
    x = ((train[columns] - mean) / scale).to_numpy()
    weights = np.linalg.solve(x.T @ x + l2 * np.eye(len(columns)), x.T @ (train["rate"] - train["rate"].mean()))
    rate = train["rate"].mean() + ((terrain_features(section)[columns] - mean) / scale).to_numpy() @ weights
    pred = np.clip(nominal * (1 - np.clip(rate, 0, None) * years), 0.1 * nominal, nominal)
    return keys.assign(thickness_pred=pred, start_date=start, target_date=pd.Timestamp(target_date),
                       source="terrain_ridge_fallback")


ANCHORED = "anchored_expected_minimum"


def anchor_case1(details, notes, vtd, grid, labels, targets):
    """Replace case-1 forecasts by the anchored forecast where neighbours were inspected on T.

    Uses the neighbours' inspection dated T (same tool run as the hidden target inspection) to
    move the last run onto the coordinates of the forecast run; the target's own rows dated T
    are never read. See ``pipetwin.anchored_forecast``.
    """
    from pipetwin import anchoring as af

    for target in targets:
        id_, at = target["id"], pd.Timestamp(target["target_date"])
        main = vtd.loc[vtd.id.eq(id_), "main_id"]
        if main.empty or not vtd.loc[vtd.main_id.eq(main.iloc[0]) & vtd.id.ne(id_) & vtd.date.eq(at)].shape[0]:
            continue
        if vtd.loc[vtd.id.eq(id_) & vtd.date.lt(at)].empty:
            continue
        frame, note = af.forecast(vtd, grid, labels, id_, at)
        mask = details.id.eq(id_) & details.case.eq(1)
        replacement = frame.assign(case=1, source=ANCHORED)
        details = pd.concat([details.loc[~mask], replacement], ignore_index=True)
        notes = [dict(n, anchored=note) if n.get("id") == id_ else n for n in notes]
    return details, notes


def elevation_case2(details, grid, targets):
    """Rank case-2 segments by the route elevation smoothed over 2 km (``pipetwin.case2_elevation``)."""
    from pipetwin import case2_elevation as ce

    for target in targets:
        id_, at = target["id"], pd.Timestamp(target["target_date"])
        mask = details.id.eq(id_) & details.case.eq(2)
        if not mask.any() or grid.loc[grid.id.eq(id_), "elevation_mid_m"].notna().sum() < 3:
            continue
        frame = ce.forecast(grid, details.loc[mask], id_, at).assign(case=2)
        if details.loc[details.id.eq(id_) & details.case.eq(1), "source"].eq(details.loc[mask, "source"].iloc[0]).all():
            frame1 = frame.assign(case=1)          # no earlier inspection: case 1 equals case 2
            details = details.loc[~(details.id.eq(id_) & details.case.eq(1))]
            details = pd.concat([details, frame1], ignore_index=True)
        details = pd.concat([details.loc[~(details.id.eq(id_) & details.case.eq(2))], frame], ignore_index=True)
    return details


def result_frame(details, case_by_id):
    parts = [details.loc[details.id.eq(id_) & details.case.eq(case)] for id_, case in case_by_id.items()]
    out = pd.concat(parts)[["id", "segment_id", "thickness_pred"]].sort_values(["id", "segment_id"])
    if out.duplicated(["id", "segment_id"]).any() or not np.isfinite(out.thickness_pred).all():
        raise ValueError("Result has duplicated keys or non-finite forecasts")
    if out.thickness_pred.lt(0).any():
        raise ValueError("Result has negative thickness")
    return out.reset_index(drop=True)


def build_forecasts(vtd, grid, labels, official, comparison, targets):
    """Both scenarios for every target, each from its own ``History.before(T)``."""
    details, notes, past_tables, selections = [], [], {}, {}
    for target in targets:
        id_, at = target["id"], pd.Timestamp(target["target_date"])
        history = History.before(vtd, labels, at)
        selection = case2_selection(comparison, at)
        case2 = forecast_case2(history, grid, official, selection, id_, at).assign(case=2)
        made = forecast_case1(history, grid, id_, at)
        if made is None:
            case1 = case2.assign(case=1)
            notes.append({"id": id_, "case1": "no earlier inspection, equals case 2"})
        else:
            case1, note, past = made
            case1 = case1.assign(case=1)
            notes.append({"id": id_, **note})
            past_tables[id_] = past
        selections[id_] = selection
        details.extend([case1, case2])
    return pd.concat(details, ignore_index=True), notes, past_tables, selections


DATES = ("2022-07-06", "2023-12-01")
KEYS = ["target_date", "id", "segment_id"]
REFS = (
    ("Reference Point 1 (Against Flow)", "Distance to Reference Point 1, m"),
    ("Reference Point 2 (With Flow)", "Distance to Reference Point 2, m"),
)


def marker_position(rows, pattern):
    candidates = []
    for name, distance in REFS:
        mask = rows[name].str.contains(pattern, case=False, regex=True, na=False) & rows[distance].notna()
        part = rows.loc[mask]
        for _, marker in part.groupby(name):
            if len(marker) < 5 or marker.Distance_raw.nunique() < 5:
                continue
            # Reports use different conventions for a downstream distance.
            for sign in (-1, 1):
                x = marker.Distance_raw + sign * marker[distance]
                center = float(x.median())
                spread = float(np.median(np.abs(x - center)))
                candidates.append((spread, -len(x), center))
    return min(candidates)[2] if candidates else None


def joint_starts(rows):
    starts = rows.Distance_raw - rows["Relative Distance"]
    return np.sort(starts.groupby(rows["Section Number"]).median().dropna().to_numpy())


def matching_joints(old, latest):
    """Match unique pairs of successive weld spacings across the two past runs."""
    if min(len(old), len(latest)) < 3:
        return np.empty((0, 2))
    eligible = np.abs(old[:-2, None] - latest[None, :-2]) <= 25.
    old_gaps, new_gaps = np.diff(old), np.diff(latest)
    for j in (0, 1):
        eligible &= np.abs(old_gaps[j:j + len(old) - 2, None] - new_gaps[None, j:j + len(latest) - 2]) <= .045
    unique = (eligible.sum(axis=0) == 1) & (eligible.sum(axis=1)[:, None] == 1)
    a, b = np.where(eligible & unique)
    return np.column_stack((latest[b], latest[b] - old[a]))


def weld_drift(route, last):
    latest = route.loc[route.date.eq(last)]
    earlier = route.loc[route.date.lt(last) & route.id.isin(latest.id)]
    old = earlier.loc[earlier.date.eq(earlier.date.max())]
    matches = matching_joints(joint_starts(old), joint_starts(latest))
    if len(matches) < 5 or np.ptp(matches[:, 0]) < 500.:
        return None, matches[:0]
    x, shift = matches.T
    i, j = np.triu_indices(len(x), 1)
    dx = x[j] - x[i]
    valid = np.abs(dx) >= 25.
    slope = np.median((shift[j[valid]] - shift[i[valid]]) / dx[valid])
    intercept = np.median(shift - slope * x)
    inliers = np.abs(shift - (intercept + slope * x)) <= 1.
    if inliers.sum() < 5 or np.ptp(x[inliers]) < 500.:
        return None, matches[inliers]
    return np.polyfit(x[inliers], shift[inliers], 1), matches[inliers]


MARKERS = (
    ("launch", r"камера\s+запуск"),
    ("middle_sign", r"(?=.*\b2(?:[.,]1)?\s*км\b)(?=.*знак)"),
    ("receipt", r"камера\s+при[её]м"),
)


def measured_markers(rows):
    return {name: marker_position(rows, pattern) for name, pattern in MARKERS}


def centered_average(values, radius):
    padded = np.pad(values, radius, mode="edge")
    return np.convolve(padded, np.full(2 * radius + 1, 1 / (2 * radius + 1)), mode="valid")


def previous_background(run, other, base):
    if other is None:
        return base, base
    past = centered_average(other.minima(), 5)
    latest = centered_average(run.minima(), 5)
    # Correct slow spatial deviations within the section without importing its old thickness level.
    correction = (past - past.mean()) - (latest - latest.mean())
    local = run.global_m - run.start
    inside = (local >= 0) & (local < run.length)
    bins = np.floor(local[inside] / 10).astype(int)
    count = np.bincount(bins, minlength=run.n)
    sparse = np.minimum(1., 5. / np.maximum(count, 1))
    return base + 0.25 * correction, base + 0.25 * sparse * correction


GEOMETRY = ["Distance_raw", "Relative Distance", "Angle", "Section Number", WALL]


def shift_between_old_runs(route, old, last):
    fit, welds = weld_drift(route.loc[route.date.isin([old, last])], last)
    if fit is None:
        return np.empty((0, 2))
    anchors = np.column_stack((welds[:, 0] - welds[:, 1], welds[:, 1]))
    old_markers = measured_markers(route.loc[route.date.eq(old)])
    new_markers = measured_markers(route.loc[route.date.eq(last)])
    for name in ("launch", "receipt"):
        before, after = old_markers[name], new_markers[name]
        if before is not None and after is not None and abs(after - before) <= 20:
            anchors = np.vstack([anchors, [before, after - before]])
    anchors = anchors[np.argsort(anchors[:, 0])]
    if np.any(np.diff(anchors[:, 0]) <= 0) or np.any(np.diff(anchors.sum(axis=1)) <= 0):
        return np.empty((0, 2))
    return anchors


def old_to_last(x, anchors):
    return x + np.interp(x, anchors[:, 0], anchors[:, 1])


def occupied_joints(rows):
    start = rows.Distance_raw - rows["Relative Distance"]
    return start.groupby(rows["Section Number"]).median().dropna().sort_values()


def nearest_index(sorted_values, query):
    right = np.clip(np.searchsorted(sorted_values, query), 0, len(sorted_values) - 1)
    left = np.clip(right - 1, 0, len(sorted_values) - 1)
    return np.where(abs(sorted_values[right] - query) < abs(sorted_values[left] - query), right, left)


def pair_joints(older, latest, anchors, tolerance_m):
    old_starts, new_starts = occupied_joints(older), occupied_joints(latest)
    if old_starts.empty or new_starts.empty:
        return []
    predicted = old_to_last(old_starts.to_numpy(), anchors)
    actual = new_starts.to_numpy()
    next_idx = nearest_index(actual, predicted)
    prev_idx = nearest_index(predicted, actual)
    paired = (prev_idx[next_idx] == np.arange(len(predicted))) & (abs(actual[next_idx] - predicted) <= tolerance_m)
    return [(old_starts.index[i], new_starts.index[j], actual[j]) for i, j in enumerate(next_idx) if paired[i]]


def pair_defects(older, latest, axial_m, angle_deg, *, with_weights=False):
    before = older.loc[older.side_normalized.eq("internal"), GEOMETRY].dropna()
    after = latest.loc[latest.side_normalized.eq("internal"), GEOMETRY].dropna()
    if before.empty or after.empty:
        return (np.empty(0), np.empty(0)) if with_weights else np.empty(0)
    rel = abs(before["Relative Distance"].to_numpy()[:, None] - after["Relative Distance"].to_numpy()[None, :])
    angle = abs(before.Angle.to_numpy()[:, None] - after.Angle.to_numpy()[None, :]) % 360
    angle = np.minimum(angle, 360 - angle)
    distance = np.where((rel <= axial_m) & (angle <= angle_deg),
                        (rel / axial_m) ** 2 + (angle / angle_deg) ** 2, np.inf)
    new_idx = distance.argmin(axis=1)
    old_idx = distance.argmin(axis=0)
    i = np.arange(len(before))
    # Correspondence is decided by position and angle, independently of thickness.
    paired = np.isfinite(distance[i, new_idx]) & (old_idx[new_idx] == i)
    losses = before[WALL].to_numpy()[paired] - after[WALL].to_numpy()[new_idx[paired]]
    if with_weights:
        # Reliability of a pair falls with its distance in the tolerance ellipse.
        return losses, np.exp(-distance[i[paired], new_idx[paired]] / 2)
    return losses


def weighted_median(values, weights):
    order = np.argsort(values)
    values, weights = np.asarray(values, float)[order], np.asarray(weights, float)[order]
    cumulative = weights.cumsum()
    half = cumulative[-1] / 2
    index = int(np.searchsorted(cumulative, half))
    if index + 1 < len(values) and np.isclose(cumulative[index], half):
        return (values[index] + values[index + 1]) / 2
    return values[index]


def matched_growth(older, latest, joints, years, axial_m, angle_deg, *, weighted=False):
    """Loss rate per paired joint: median of matched pairs, or with ``weighted`` the median and
    support weighted by each pair's position reliability."""
    old_groups = dict(tuple(older.groupby("Section Number")))
    new_groups = dict(tuple(latest.groupby("Section Number")))
    observations = []
    for first, second, position in joints:
        if weighted:
            loss, weights = pair_defects(old_groups[first], new_groups[second], axial_m, angle_deg,
                                         with_weights=True)
            if len(loss):
                observations.append((position, weighted_median(loss, weights) / years, weights.sum()))
            continue
        loss = pair_defects(old_groups[first], new_groups[second], axial_m, angle_deg)
        if len(loss):
            observations.append((position, np.median(loss) / years, len(loss)))
    return np.asarray(observations, dtype=float).reshape(-1, 3)


def growth_correction(run, observations, years, *, reference_rate=None):
    if not len(observations):
        return np.zeros(run.n)
    positions, rates, counts = observations.T
    # Use deviations from the pooled rate. The PR's general wall-loss term is still applied.
    relative_rates = rates - (np.median(rates) if reference_rate is None else reference_rate)
    close = abs(run.km[:, None] * 1000 - positions) <= 30.
    support = close * np.minimum(counts, 3)
    rate = (support * relative_rates).sum(axis=1) / (support.sum(axis=1) + 6.)
    return np.clip(rate * years, -.2, .2)
