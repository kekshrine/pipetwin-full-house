"""Raw CSV -> cleaned inspections, 10-m grid and labels (organisers' clarifications)."""

from hashlib import sha256
import json
import math

import numpy as np
import pandas as pd


def regression_metrics(truth, prediction):
    """Organizer metric is Pearson r squared, NOT sklearn r2_score."""
    y, p = list(map(float, truth)), list(map(float, prediction))
    if not y or len(y) != len(p) or not all(math.isfinite(v) for v in y + p):
        raise ValueError("Metrics require equal nonempty finite vectors")
    n = len(y)
    my, mp = math.fsum(y) / n, math.fsum(p) / n
    yc = [v - my for v in y]
    pc = [v - mp for v in p]
    sy, sp = math.fsum(v*v for v in yc), math.fsum(v*v for v in pc)
    constant = min(y) == max(y) or min(p) == max(p)
    r = None if constant or sy == 0 or sp == 0 else max(-1.0, min(1.0,
        math.fsum(a*b for a, b in zip(yc, pc)) / math.sqrt(sy * sp)))
    return {"n": n, "pearson_r": r, "pearson_r_squared": None if r is None else r*r,
            "mae_mm": math.fsum(abs(a-b) for a, b in zip(y, p)) / n,
            "rmse_mm": math.sqrt(math.fsum((a-b)**2 for a, b in zip(y, p)) / n)}


MODE_FILE = "tech_mode_for_embedings_with_length_with_temperatures_ids16.csv"


def read_passport(folder):
    raw = pd.read_csv(folder / "pot_south.csv", low_memory=False)
    columns = {
        "ID основного участка трубопровода": "main_id",
        "ID простого участка трубопровода": "id",
        "L простого участка, км": "length_m",
        "Общие данные_Дата ввода": "commissioned_at",
        "Общие данные_D, мм": "diameter_mm",
        "Общие данные_Толщина  стенки по исполнительной доку-ии, мм": "nominal_thickness_mm",
    }
    mapping = {next(c for c in raw.columns if c.endswith(suffix)): dest for suffix, dest in columns.items()}
    passport = raw[list(mapping)].rename(columns=mapping)
    passport["id"] = passport["id"].astype("int64")
    passport["length_m"] = passport["length_m"].mul(1000).round(6)
    passport["commissioned_at"] = pd.to_datetime(passport["commissioned_at"], format="%d.%m.%Y")
    if not passport["id"].is_unique or not passport["length_m"].gt(0).all():
        raise ValueError("Expected one passport with positive length per simple section")
    return passport.sort_values("id")


def make_segments(passport, altitude):
    profile = altitude[["id", "total_distance", "z"]].drop_duplicates()
    if profile.duplicated(["id", "total_distance"]).any():
        raise ValueError("Conflicting elevation at the same distance")
    if set(profile["id"]) != set(passport["id"]):
        raise ValueError("Profile IDs do not match passport IDs")
    frames = []
    for row in passport.itertuples(index=False):
        segment_id = np.arange(int(np.ceil(row.length_m / 10)))
        start = 10 * segment_id
        end = np.minimum(start + 10, row.length_m)
        midpoint = (start + end) / 2
        heights = profile.loc[profile["id"].eq(row.id)].sort_values("total_distance")
        distance = heights["total_distance"]
        z = heights["z"]
        z_start = np.interp(start, distance, z, left=np.nan, right=np.nan)
        z_end = np.interp(end, distance, z, left=np.nan, right=np.nan)
        frames.append(pd.DataFrame({
            "id": row.id, "main_id": row.main_id, "segment_id": segment_id,
            "start_m": start, "end_m": end, "midpoint_m": midpoint,
            "section_length_m": row.length_m, "commissioned_at": row.commissioned_at,
            "diameter_mm": row.diameter_mm, "nominal_thickness_mm": row.nominal_thickness_mm,
            "elevation_mid_m": np.interp(midpoint, distance, z, left=np.nan, right=np.nan),
            "elevation_change_m": z_end - z_start,
        }))
    return pd.concat(frames, ignore_index=True)


DAYS_PER_YEAR = 365.25
THICKNESS = "min_residual_thickness_mm"
GEO_FEATURES = ["relative_elevation", "local_grade"]
DEFAULT_CASE1_MODEL = "hybrid_10m"
CASE2_MODELS = ("nominal", "pooled_loss_rate", "terrain_ridge")
SEGMENT_KEYS = ["id", "segment_id"]


def between_vtd(section, history, target, model):
    id_ = section["id"].iloc[0]
    earlier = history.loc[history["id"].eq(id_) & history["inspection_date"].lt(target)]
    if earlier.empty:
        raise ValueError(f"Case 1 requires a past VTD for section {id_}")
    start = earlier["inspection_date"].max()
    latest = earlier.loc[earlier["inspection_date"].eq(start)].set_index("segment_id")[THICKNESS]
    latest = latest.sort_index()
    section_median = latest.median()
    previous = earlier.sort_values("inspection_date").drop_duplicates("segment_id", keep="last")
    previous = previous.set_index("segment_id")[THICKNESS]
    rate, train_rows = 0.0, 0
    rows = []
    for row in section.itertuples(index=False):
        sid = row.segment_id
        nearby = latest.loc[sid - 2:sid + 2]
        if model == DEFAULT_CASE1_MODEL:
            if sid in latest:
                estimate, source = latest.loc[sid - 1:sid + 1].median(), "latest_vtd_radius_10m"
            else:
                estimate, source = section_median, "latest_vtd_section_median"
        elif model != "last_min" and not nearby.empty:
            estimate, source = nearby.median(), "latest_vtd_radius_20m"
        elif sid in latest:
            estimate, source = latest[sid], "latest_vtd_segment"
        elif sid in previous:
            estimate, source = previous[sid], "older_vtd_segment"
        elif not nearby.empty:
            estimate, source = nearby.median(), "latest_vtd_radius_20m"
        else:
            estimate, source = section_median, "latest_vtd_section_median"
        # Rate is calibrated on visit-to-visit changes and applied over that horizon.
        # An older fallback is left unadjusted rather than using a mismatched horizon.
        if source != "older_vtd_segment":
            estimate -= rate * (target - start).days / DAYS_PER_YEAR
        rows.append({
            "id": id_, "segment_id": sid,
            "thickness_pred": float(np.clip(estimate, 0, row.nominal_thickness_mm)),
            "source": source, "training_rows": train_rows,
            "rate": rate,
        })
    return pd.DataFrame(rows)


def ridge_rate(train, section):
    features = train[GEO_FEATURES]
    center = features.median().fillna(0)
    x = features.fillna(center).to_numpy()
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1
    x = (x - mean) / scale
    loss = train["loss_rate"].to_numpy()
    # Fixed L2 penalty. Centered features allow an unpenalized intercept.
    coef = np.linalg.solve(x.T @ x + 10 * np.eye(len(GEO_FEATURES)), x.T @ (loss - loss.mean()))
    target = (section[GEO_FEATURES].fillna(center).to_numpy() - mean) / scale
    return np.maximum(0, loss.mean() + target @ coef)


NOMINAL = "nominal_thickness_mm"
PASSPORT_FEATURES = [NOMINAL, "diameter_mm", "age_at_start", "horizon_years"]
PAST = ["input_min", "previous_observed", "hybrid_10m"]
TERRAIN = ["grade_50m", "relief_110m", "valley_depth_110m", "abs_grade"]
PAST_EXTENDED = [
    "input_p10", "input_std", "input_inner_min", "log_input_count",
    "prior_median_50m", "prior_median_100m", "prior_coverage_50m",
    "days_since_last_observation", "previous_annual_loss",
]
MODE = ["watercut", "q", "v_liquid_true", "t_start", "gas_content_rate"]
PRESSURE = ["p_start", "p_end"]
REGIMES = ["Ламинарный", "Переходный", "Турбулентный"]
REGIME_FEATURES = ["laminar", "transition", "turbulent"]
MODE_FEATURES = [
    f"{key}_{stat}" for key in MODE for stat in ("median", "mean", "p90", "change")
] + [*REGIME_FEATURES, "mode_coverage", "temperature_coverage", "zero_flow_share"]
PRESSURE_FEATURES = [f"{key}_{stat}" for key in PRESSURE for stat in ("median", "p90")]


def feature_sets(case):
    prior = PAST if case == 1 else []
    extended = PAST_EXTENDED if case == 1 else []
    basic = PASSPORT_FEATURES + GEO_FEATURES + prior
    spatial = basic + TERRAIN + extended
    return {
        "basic": basic,
        "spatial": spatial,
        "spatial_mode": spatial + MODE_FEATURES,
        "spatial_pressure": spatial + MODE_FEATURES + PRESSURE_FEATURES,
        "mode_only": PASSPORT_FEATURES + MODE_FEATURES,
    }


def terrain_features(grid):
    grid = grid.sort_values(SEGMENT_KEYS).copy()
    z = grid.groupby("id")["elevation_mid_m"]
    midpoint = grid.groupby("id")["midpoint_m"]
    grid["grade_50m"] = (z.shift(-2) - z.shift(2)) / (midpoint.shift(-2) - midpoint.shift(2))
    rolling = z.rolling(11, center=True, min_periods=3)
    low = rolling.min().droplevel(0)
    grid["relief_110m"] = rolling.max().droplevel(0) - low
    grid["valley_depth_110m"] = (z.shift(5) + z.shift(-5)) / 2 - grid["elevation_mid_m"]
    grid["abs_grade"] = grid["local_grade"].abs()
    return grid


def daily_regime(data_dir, grid):
    fields = ["simple_section_id", "calc_date", "length", "d_mm", "flow_regime_real", *MODE, *PRESSURE]
    raw = pd.read_csv(data_dir / MODE_FILE, usecols=fields).rename(columns={"simple_section_id": "id"})
    dimensions = grid[["id", "section_length_m", "diameter_mm"]].drop_duplicates()
    raw = raw.merge(dimensions, on="id", how="left", validate="many_to_one")
    if not (np.isclose(raw["length"], raw["section_length_m"], atol=0.1, rtol=0).all()
            and np.isclose(raw["d_mm"] * 1000, raw["diameter_mm"], atol=0.5, rtol=0).all()):
        raise ValueError("Regime rows disagree with passport IDs, lengths or diameters")
    raw["day"] = pd.to_datetime(raw["calc_date"], format="mixed").dt.normalize()
    if raw["day"].isna().any():
        raise ValueError("Regime contains undated rows")
    for source, key in zip(REGIMES, REGIME_FEATURES):
        raw[key] = raw["flow_regime_real"].eq(source).astype(float).where(raw["flow_regime_real"].isin(REGIMES))
    return raw.groupby(["id", "day"])[MODE + PRESSURE + REGIME_FEATURES].median().reset_index()


def regime_window(daily, id_, start, target):
    # Contest snapshot: regime observed before the target day is available.
    part = daily.loc[daily["id"].eq(id_) & daily["day"].ge(start) & daily["day"].lt(target)]
    midpoint = start + (target - start) / 2
    early = part.loc[part["day"].lt(midpoint), MODE]
    late = part.loc[part["day"].ge(midpoint), MODE]
    stats = {}
    for key in MODE:
        values = part[key].dropna()
        if values.empty:
            stats.update({f"{key}_{stat}": np.nan for stat in ("median", "mean", "p90", "change")})
            continue
        stats[f"{key}_median"] = values.median()
        stats[f"{key}_mean"] = values.mean()
        stats[f"{key}_p90"] = values.quantile(0.9)
        stats[f"{key}_change"] = (
            late[key].median() - early[key].median()
            if min(late[key].count(), early[key].count()) >= 7 else np.nan
        )
    for key in PRESSURE:
        values = part[key].dropna()
        stats[f"{key}_median"] = values.median() if len(values) else np.nan
        stats[f"{key}_p90"] = values.quantile(0.9)
    stats.update(part[REGIME_FEATURES].mean().to_dict())
    stats["mode_days"] = len(part)
    stats["mode_coverage"] = len(part) / (target - start).days
    stats["temperature_coverage"] = part["t_start"].count() / (target - start).days
    stats["zero_flow_share"] = part["q"].dropna().eq(0).mean()
    stats["last_mode_day"] = part["day"].max()
    return stats


def feature_frame(case, section, history, observations, daily, start, target):
    if not section["commissioned_at"].le(start).all() or start >= target:
        raise ValueError("Invalid forecast window")
    if case != 2:
        raise ValueError("Model features are built for case 2 only")
    frame = section.copy()
    frame["initial_thickness"] = frame[NOMINAL]
    frame["age_at_start"] = (start - frame["commissioned_at"]).dt.days / DAYS_PER_YEAR
    frame["horizon_days"] = (target - start).days
    frame["horizon_years"] = frame["horizon_days"] / DAYS_PER_YEAR
    return frame.assign(case=case, start_date=start, target_date=target,
                        **regime_window(daily, section["id"].iat[0], start, target))


VTD = "vtd_south_train.csv"
PROFILE = "altitudes_south.csv"
PASSPORT = "pot_south.csv"
CHEMISTRY = "repcorr_south.csv"
UZT = "uzt_south.csv"
TARGETS = {1: ("22871", "2023-12-01"), 2: ("9583", "2022-07-01")}
MODE_FIELDS = ["t_start", "t_end", "p_start", "p_end", "p_start_fact", "p_end_fact", "q", "q_oil",
               "v_liquid_true", "watercut"]
DAY = "observation_day"
DATE = "inspection_date"
ROWS = "source_row"
WALL = "Residual Wall Thickness"
DEPTH = "Defect Depth in %"
FILES = (PASSPORT, PROFILE, VTD, MODE_FILE, CHEMISTRY, UZT)
SIGNED_DELTA = "target_minus_previous_mm"


def read_source(folder, name):
    frame = pd.read_csv(folder / name, low_memory=False)
    return frame.assign(source_row=np.arange(2, len(frame) + 2))


def record_flags(flags, masks):
    for reason, mask in masks.items():
        flags[reason] = mask.fillna(False).astype(bool)
    flags["primary_reason"] = ""
    flags["reasons"] = ""
    for reason in masks:
        mask = flags[reason]
        flags.loc[mask & flags["primary_reason"].eq(""), "primary_reason"] = reason
        flags.loc[mask, "reasons"] = flags.loc[mask, "reasons"] + reason + "|"
    flags["reasons"] = flags["reasons"].str.rstrip("|")
    return flags


def exact_duplicates(frame, ignore=()):
    return frame.drop(columns=[ROWS, *ignore], errors="ignore").duplicated()


def numeric_audit(frame, name):
    result = []
    for col in frame.columns.drop(ROWS):
        s = frame[col]
        numeric = pd.to_numeric(s, errors="coerce").astype(float) if pd.api.types.is_numeric_dtype(s) else None
        finite = (numeric[np.isfinite(numeric)] if numeric is not None else pd.Series(dtype=float))
        p01, p25, p50, p75, p99 = finite.quantile([.01, .25, .5, .75, .99])
        iqr = p75 - p25
        result.append({
            "file": name, "column": col, "rows": len(frame), "dtype": str(s.dtype),
            "missing": int(s.isna().sum()), "missing_pct": 100 * s.isna().mean(),
            "distinct": s.nunique(), "constant_nonnull": bool(s.nunique() <= 1),
            "numeric_nonfinite_nonmissing": int(numeric.notna().sum() - len(finite)) if numeric is not None else None,
            "min": finite.min(), "p01": p01, "median": p50, "p99": p99, "max": finite.max(),
            "outside_3iqr": int((finite.lt(p25 - 3 * iqr) | finite.gt(p75 + 3 * iqr)).sum()) if iqr > 0 else None,
        })
    return result


def scope_info(raw, passport, id_field, date_field=None, fmt="mixed", dayfirst=False):
    scoped = raw.assign(id=pd.to_numeric(raw[id_field], errors="coerce").astype("Int64"))
    if date_field:
        scoped[DAY] = pd.to_datetime(raw[date_field], errors="coerce", format=fmt, dayfirst=dayfirst).dt.normalize()
    else:
        scoped[DAY] = pd.NaT
    return scoped.merge(
        passport[["id", "length_m", "commissioned_at", "diameter_mm", NOMINAL]], on="id",
        how="left", validate="many_to_one", suffixes=("", "_passport"),
    )


def closed_survey_mask(id_, date, targets):
    hidden = pd.Series(False, index=id_.index)
    for target in targets.itertuples(index=False):
        hidden |= id_.eq(target.id) & date.ge(target.target_date)
    return hidden.fillna(False)


def clean_vtd(raw, passport, targets):
    source = scope_info(raw, passport, "id", "date", "%Y-%m-%d")
    x = source["Distance_adjusted"]
    wall = source[WALL]
    hidden = closed_survey_mask(source.id, source[DAY], targets)
    initial = source["Wall Thickness"]
    alternate_date = pd.to_datetime(source["vtd_date_raw"], errors="coerce", format="mixed", dayfirst=True)
    source["date_text_conflict"] = alternate_date.notna() & source[DAY].ne(alternate_date)
    source["residual_gt_vtd_wall"] = wall.gt(initial + 1e-8)
    source["vtd_wall_ne_passport"] = (initial - source[NOMINAL]).abs().gt(.05)
    source["distance_fields_disagree"] = (source["Distance"] - x).abs().gt(.05)
    source["section_length_diff_gt_0_1m"] = (source["section_length"] - source["length_m"]).abs().gt(.1)
    source["depth_pct"] = source[DEPTH] * np.where(source[DAY].dt.year.eq(2018), 100, 1)
    source["depth_pct_outside_0_100"] = ~source["depth_pct"].between(0, 100) & source["depth_pct"].notna()
    scale = (1 - wall / initial.where(initial.gt(0))) * 100
    source["depth_discrepancy_gt_2pp"] = (scale - source["depth_pct"]).abs().gt(2)
    source["angle_outside_0_360"] = ~source["Angle"].between(0, 360) & source["Angle"].notna()
    source["negative_defect_size"] = source["Feature Length"].lt(0) | source["Feature Width"].lt(0)
    source["adjusted_distance"] = x
    source["side_normalized"] = source["Defect Type"].map({
        "внутр.": "internal", "Внутренний": "internal", "внешн.": "external", "Внешний": "external",
    })
    flags = record_flags(source[[ROWS, "id", DAY]].copy(), {
        "vtd_before_commission": source[DAY].lt(source["commissioned_at"]),
        "unknown_id": source["commissioned_at"].isna(),
        "invalid_date": source[DAY].isna(),
        "target_vtd_at_or_after_t": hidden,
        "invalid_distance": ~np.isfinite(x) | x.lt(0) | x.ge(source["length_m"]),
        "invalid_residual_thickness": ~np.isfinite(wall) | wall.le(0),
        "duplicate_payload": exact_duplicates(raw, ["row_id"]),
    })
    keep = flags["primary_reason"].eq("")
    clean = source.loc[keep].copy()
    clean["segment_id"] = np.floor(clean["adjusted_distance"] / 10).astype(int)
    clean[DATE] = clean[DAY]
    return clean, flags, source


def clean_profile(raw, passport):
    source = scope_info(raw, passport, "id")
    invalid = ~np.isfinite(source["total_distance"]) | source["total_distance"].lt(0) | ~np.isfinite(source["z"])
    valid = source.loc[~invalid]
    conflict = valid.groupby(["id", "total_distance"])["z"].transform("nunique").gt(1).reindex(source.index, fill_value=False)
    flags = record_flags(source[[ROWS, "id", DAY]].copy(), {
        "unknown_id": source["commissioned_at"].isna(), "invalid_profile_value": invalid,
        "conflicting_height": conflict, "duplicate_elevation": source.duplicated(["id", "total_distance", "z"]),
    })
    return source.loc[flags.primary_reason.eq("")].copy(), flags


def clean_mode(raw, passport):
    source = scope_info(raw, passport, "simple_section_id", "calc_date")
    source["length_mismatch"] = (source["length"] - source["length_m"]).abs().gt(.1)
    source["diameter_mismatch_m_converted"] = (source["d_mm"] * 1000 - source["diameter_mm"]).abs().gt(.5)
    source["calculated_pressure_gt_100"] = source[["p_start", "p_end"]].gt(100).any(axis=1)
    source["watercut_between_zero_and_one"] = source["watercut"].between(0, 1) & source["watercut"].gt(0)
    flags = record_flags(source[[ROWS, "id", DAY]].copy(), {
        "unknown_id": source["commissioned_at"].isna(), "invalid_date": source[DAY].isna(),
        "mode_before_commission": source[DAY].lt(source["commissioned_at"]),
        "exact_duplicate": exact_duplicates(raw),
    })
    clean = source.loc[flags.primary_reason.eq("")].copy()
    # Negative magnitudes and fractions outside both plausible scales are masked per cell.
    for col in MODE_FIELDS:
        if col in ("t_start", "t_end"):
            continue
        clean[col] = clean[col].mask(clean[col].lt(0))
    clean["watercut"] = clean["watercut"].mask(clean["watercut"].gt(100))
    # Reconcile values only within the same calendar day, retaining both the daily view and source rows.
    daily = clean.groupby(["id", DAY], as_index=False).agg(
        mode_rows=(ROWS, "size"), **{col: (col, "median") for col in MODE_FIELDS},
    )
    return clean, flags, daily


def summarize_visits(clean):
    keys = ["id", DATE, "segment_id"]
    groups = clean.groupby(keys)
    visits = groups.agg(
        min_residual_thickness_mm=(WALL, "min"), records=(WALL, "size"),
        median_residual_thickness_mm=(WALL, "median"), depth_p90_pct=("depth_pct", lambda s: s.quantile(.9)),
        max_defect_length_mm=("Feature Length", "max"), max_defect_width_mm=("Feature Width", "max"),
    ).reset_index()
    selected = clean.loc[groups[WALL].idxmin(), [*keys, ROWS, "row_id"]].rename(columns={
        ROWS: "minimum_source_csv_row", "row_id": "minimum_source_vtd_id",
    })
    return visits.merge(selected, on=keys, validate="one_to_one")


def visit_pairs(history, grid):
    result = []
    for id_, observations in history.groupby("id"):
        visits = list(observations.groupby(DATE))
        for (start, past), (end, current) in zip(visits, visits[1:]):
            last_col = "previous_min_mm"
            origin = "minimum_source_csv_row"
            prev = past[["segment_id", THICKNESS, origin]].rename(columns={
                THICKNESS: last_col, origin: "previous_source_csv_row",
            })
            current = current[["segment_id", THICKNESS, origin]].rename(columns={origin: "target_source_csv_row"})
            current = current.merge(prev, on="segment_id", how="left", validate="one_to_one")
            current[SIGNED_DELTA] = current[THICKNESS] - current[last_col]
            current["pair_status"] = np.select(
                [current[last_col].isna(), current[SIGNED_DELTA].gt(1e-9)],
                ["no_previous_minimum", "thickness_increased"], default="accepted",
            )
            current["delta_rate_mm_year"] = (-current[SIGNED_DELTA] * DAYS_PER_YEAR / (end - start).days).where(
                current["pair_status"].eq("accepted")
            )
            section = grid.loc[grid.id.eq(id_)]
            compatible = past.assign(inspection_date=start)
            # Fixed Hybrid uses only the preceding inspection; no labels influence which rows are predicted.
            pred = between_vtd(section, compatible, end, "hybrid_10m")
            result.append(current.assign(id=id_, start_date=start, target_date=end).merge(
                pred[["segment_id", "thickness_pred"]], on="segment_id", validate="one_to_one",
            ))
    return pd.concat(result, ignore_index=True)


def first_visits(history, grid):
    first = history.loc[history[DATE].eq(history.groupby("id")[DATE].transform("min"))]
    info = grid[["id", "commissioned_at", NOMINAL]].drop_duplicates()
    first = first.merge(info, validate="many_to_one").rename(columns={DATE: "target_date", "commissioned_at": "start_date"})
    first["age_years"] = (first["target_date"] - first["start_date"]).dt.days / DAYS_PER_YEAR
    first["nominal_minus_observed_mm"] = first[NOMINAL] - first[THICKNESS]
    return first


def resolve_targets(passport):
    rows = []
    for case, (suffix, date) in TARGETS.items():
        found = passport.loc[passport.id.astype(str).str.endswith(suffix)]
        if len(found) != 1:
            raise ValueError(f"Need one passport ID ending with {suffix}, got {len(found)}")
        rows.append(found.assign(case=case, target_date=pd.Timestamp(date)))
    return pd.concat(rows, ignore_index=True)


def coverage_stats(history, grid):
    row_counts = grid.groupby("id").size()
    visit = history.groupby(["id", DATE], as_index=False).agg(
        observed_segments=("segment_id", "size"), valid_rows=("records", "sum"), median_mm=(THICKNESS, "median"),
    )
    visit["grid_segments"] = visit["id"].map(row_counts)
    visit["unobserved_segments"] = visit["grid_segments"] - visit["observed_segments"]
    return visit


def historical_scores(pairs):
    results = []
    for _, group in [("all_pairs", pairs), *pairs.groupby("target_date")]:
        for subset, frame in [
            ("full_observed_keys", group),
            ("diagnostic_without_increase", group.loc[~group.pair_status.eq("thickness_increased")]),
            ("diagnostic_confirmed_pairs_only", group.loc[group.pair_status.eq("accepted")]),
        ]:
            if len(frame):
                results.append({
                    "date": group.target_date.iat[0] if group.target_date.nunique() == 1 else "pooled",
                    "scope": subset, **regression_metrics(frame[THICKNESS], frame["thickness_pred"]),
                })
    return pd.DataFrame(results)


PRIOR_COLUMNS = {
    THICKNESS: "prior_min_mm", "records": "prior_records", "depth_p90_pct": "prior_depth_p90_pct",
    "median_residual_thickness_mm": "prior_median_mm", "max_defect_length_mm": "prior_max_length_mm",
    "max_defect_width_mm": "prior_max_width_mm",
}


def prior_inputs(section, history, date):
    previous = history.loc[history[DATE].eq(date), SEGMENT_KEYS + list(PRIOR_COLUMNS)].rename(columns=PRIOR_COLUMNS)
    return section.merge(previous, on=SEGMENT_KEYS, how="left", validate="one_to_one")


PROFILE_COLUMNS = ["elevation_mid_m", "elevation_change_m", "relative_elevation", "local_grade",
                   "grade_50m", "relief_110m", "valley_depth_110m", "abs_grade"]


def window_mode(daily, id_, start, end):
    period = daily.loc[daily.id.eq(id_) & daily[DAY].ge(start) & daily[DAY].lt(end)]
    result = {"horizon_days": (end - start).days, "mode_days": len(period),
              "mode_first_day": period[DAY].min(), "mode_last_day": period[DAY].max()}
    for name in MODE_FIELDS:
        known = period[name].dropna()
        result[f"{name}_median"] = known.median() if len(known) else np.nan
        result[f"{name}_p90"] = known.quantile(.9)
        result[f"{name}_known_days"] = len(known)
    return result


def target_sets(targets, grid, history, daily):
    tables, metadata = {}, []
    for target in targets.itertuples(index=False):
        at = target.target_date
        past = history.loc[history.id.eq(target.id) & history[DATE].lt(at)]
        last = past[DATE].max()
        start = last if target.case == 1 else target.commissioned_at
        if pd.isna(start) or start >= at:
            raise ValueError("Missing usable forecast history")
        regime = window_mode(daily, target.id, start, at)
        section = grid.loc[grid.id.eq(target.id)].assign(case=target.case, start_date=start, target_date=at, **regime)
        if target.case == 1:
            section = prior_inputs(section, past, start)
            section["prior_observed"] = section["prior_min_mm"].notna()
        else:
            section["prior_observed"] = False
        if not section["mode_last_day"].dropna().lt(at).all() or THICKNESS in section:
            raise ValueError("Invalid target feature snapshot")
        tables[f"target_{target.id}_with_profile"] = section
        if target.case == 2:
            tables[f"target_{target.id}_mode_only"] = section.drop(columns=PROFILE_COLUMNS)
        metadata.append({
            "id": target.id, "case": target.case, "target_date": at, "forecast_start": start,
            "segments": len(section), "profile_midpoint_known": int(section.elevation_mid_m.count()),
            "prior_observed_segments": int(section.prior_observed.sum()), "historical_visits": past[DATE].nunique(),
            "mode_days": regime["mode_days"], "mode_first": regime["mode_first_day"], "mode_last": regime["mode_last_day"],
            "mode_coverage": regime["mode_days"] / (at - start).days,
        })
    return tables, pd.DataFrame(metadata)


def audit_keys(raw, passport_raw, passport):
    key_fields = {
        VTD: ["id", "date", "Distance_adjusted"], PROFILE: ["id", "total_distance"],
        UZT: ["id", "ID замера"], MODE_FILE: ["simple_section_id", "calc_date"],
        CHEMISTRY: ["Simple Section ID", "UKK Number", "Sample Removal Date"],
        PASSPORT: [next(c for c in passport_raw if c.endswith("ID простого участка трубопровода"))],
    }
    rows = []
    for name, data in raw.items():
        fields = key_fields[name]
        id_ = pd.to_numeric(data[fields[0]], errors="coerce")
        rows.append({
            "file": name, "rows": len(data), "columns": len(data.columns) - 1,
            "exact_duplicate_extra_rows": int(exact_duplicates(data).sum()),
            "key": "|".join(fields), "repeated_key_extra_rows": int(data.duplicated(fields).sum()),
            "unlinked_id_rows": int((id_.notna() & ~id_.isin(passport.id)).sum()), "missing_id_rows": int(id_.isna().sum()),
        })
    return pd.DataFrame(rows)


def file_audits(raw, flags):
    rows, reasons, groups = [], [], []
    for name, frame in raw.items():
        audit = flags[name]
        excluded = audit.primary_reason.ne("")
        rows.append({"file": name, "raw_rows": len(frame), "excluded": int(excluded.sum()),
                     "retained": int((~excluded).sum())})
        remaining = len(frame)
        for reason in audit.columns.drop([ROWS, "id", DAY, "primary_reason", "reasons"]):
            count = int(audit.primary_reason.eq(reason).sum())
            reasons.append({"file": name, "reason": reason, "starting_rows": remaining, "excluded": count,
                            "retained": remaining - count, "flagged_including_overlaps": int(audit[reason].sum())})
            remaining -= count
        counts = audit.assign(excluded=excluded).groupby(["id", DAY], dropna=False, as_index=False).agg(
            raw_rows=(ROWS, "size"), excluded=("excluded", "sum"),
        )
        counts["retained"] = counts.raw_rows - counts.excluded
        groups.append(counts.assign(file=name))
        if remaining != int((~excluded).sum()):
            raise ValueError("Audit rows did not reconcile")
    return pd.DataFrame(rows), pd.DataFrame(reasons), pd.concat(groups, ignore_index=True)


def train_features(labels, which, grid, history, daily):
    parts = []
    for (id_, start, end), group in labels.groupby(["id", "start_date", "target_date"]):
        section = grid.loc[grid.id.eq(id_)].copy()
        days = (end - start).days
        if days <= 0:
            continue
        section = section.assign(case=which, **window_mode(daily, id_, start, end))
        if which == 1:
            section = prior_inputs(section, history, start)
            section["prior_observed"] = section["prior_min_mm"].notna()
        else:
            section["prior_observed"] = False
        mask = section["mode_last_day"].dropna().lt(end)
        if not mask.all():
            raise ValueError("Regime date leaks into historical features")
        label_cols = SEGMENT_KEYS + ["start_date", "target_date", THICKNESS]
        if which == 1:
            label_cols.append("delta_rate_mm_year")
        parts.append(group[label_cols].merge(section, on=SEGMENT_KEYS, validate="one_to_one"))
    return pd.concat(parts, ignore_index=True)


def run_cleaning(data_dir, out):
    raw = {name: read_source(data_dir, name) for name in FILES}
    passport = read_passport(data_dir)
    passport_id = next(c for c in raw[PASSPORT] if c.endswith("ID простого участка трубопровода"))
    passport = passport.merge(raw[PASSPORT][[ROWS, passport_id]].rename(columns={passport_id: "id"}),
                              on="id", validate="one_to_one")
    targets = resolve_targets(passport)
    vtd, vtd_flags, vtd_audit = clean_vtd(raw[VTD], passport, targets)
    open_audit = ~vtd_flags["target_vtd_at_or_after_t"]
    elevation, profile_flags = clean_profile(raw[PROFILE], passport)
    mode, mode_flags, daily = clean_mode(raw[MODE_FILE], passport)
    chem = scope_info(raw[CHEMISTRY], passport, "Simple Section ID", "Sample Removal Date")
    chem["installed_at"] = pd.to_datetime(chem["Sample Installation Date"], format="mixed", errors="coerce")
    chem_flags = record_flags(chem[[ROWS, "id", DAY]].copy(), {
        "no_linked_passport": chem["commissioned_at"].isna(),
        "missing_result_date": chem[DAY].isna(),
        "sampling_dates_reversed": chem["installed_at"].dt.normalize().gt(chem[DAY]),
        "sample_before_commission": chem[DAY].lt(chem["commissioned_at"]),
    })
    uzt = scope_info(raw[UZT], passport, "id", "date")
    uzt_flags = record_flags(uzt[[ROWS, "id", DAY]].copy(), {
        "uzt_excluded_by_organizers": pd.Series(True, index=uzt.index),
    })
    passport_flags = record_flags(passport[[ROWS, "id"]].assign(observation_day=passport["commissioned_at"]), {})
    flags = {PASSPORT: passport_flags, PROFILE: profile_flags, VTD: vtd_flags, MODE_FILE: mode_flags,
             CHEMISTRY: chem_flags, UZT: uzt_flags}

    grid = make_segments(passport, elevation)
    height = grid.groupby("id")["elevation_mid_m"]
    limit = height.transform("max") - height.transform("min")
    grid["relative_elevation"] = (grid["elevation_mid_m"] - height.transform("min")) / limit.replace(0, np.nan)
    grid["local_grade"] = grid["elevation_change_m"] / (grid["end_m"] - grid["start_m"])
    grid = terrain_features(grid)
    history = summarize_visits(vtd)
    if history["records"].sum() != len(vtd) or history.duplicated(["id", DATE, "segment_id"]).any():
        raise ValueError("Observation totals or historical segment keys do not reconcile")
    for target in targets.itertuples(index=False):
        if (vtd.id.eq(target.id) & vtd[DATE].ge(target.target_date)).any():
            raise ValueError("A closed target inspection entered the main table")
    pairs = visit_pairs(history, grid)
    first = first_visits(history, grid)
    clean_pairs = pairs.loc[pairs.pair_status.eq("accepted")].copy()
    targets_frames, metadata = target_sets(targets, grid, history, daily)
    audit, reasons, by_id_date = file_audits(raw, flags)
    visits = coverage_stats(history, grid)
    pair_summary = pairs.groupby(["id", "start_date", "target_date", "pair_status"], as_index=False).agg(
        segments=("segment_id", "size"), median_change_mm=(SIGNED_DELTA, "median"),
    )
    # Every forecast input is assembled against its own target date. Full historical feature tables
    # serve research; per-target label subsets enforce another cutoff at the actual forecast date.
    ml1 = train_features(clean_pairs, 1, grid, history, daily)
    ml2 = train_features(first.loc[first.age_years.gt(0)], 2, grid, history, daily)
    for target in targets.itertuples(index=False):
        data = ml1 if target.case == 1 else ml2
        train = data.loc[data.target_date.lt(target.target_date)].copy()
        targets_frames[f"target_{target.id}_training"] = train
        if target.case == 2:
            targets_frames[f"target_{target.id}_training_mode_only"] = train.drop(columns=PROFILE_COLUMNS)
        metadata.loc[metadata.id.eq(target.id), "train_labels_before_t"] = len(train)
    scoped_mode = daily.merge(targets[["id", "target_date", "commissioned_at"]], on="id", validate="many_to_one")
    scoped_mode = scoped_mode.loc[scoped_mode[DAY].ge(scoped_mode.commissioned_at)
                                  & scoped_mode[DAY].lt(scoped_mode.target_date)]
    comparisons = historical_scores(pairs)
    future_mode = mode.merge(targets[["id", "target_date"]], on="id", validate="many_to_one")
    future_mode = future_mode.loc[future_mode[DAY].ge(future_mode["target_date"]), [ROWS, "id", DAY, "target_date"]]
    future_mode["reason"] = "mode_on_or_after_target_date"
    accounting = audit_keys(raw, raw[PASSPORT], passport)

    out.mkdir(parents=True, exist_ok=True)
    cleaned = out / "tables"
    excluded = out / "excluded"
    for path in (cleaned, excluded):
        path.mkdir(exist_ok=True)
    for name, frame in {
        "passport": passport, "profile": elevation, "vtd": vtd, "regime": mode, "regime_daily": daily,
        "repcorr_linked": chem.loc[chem_flags.primary_reason.eq("")], "grid": grid, "vtd_segment_minima": history,
        "case1_pairs_all": pairs, "case1_training": ml1, "case2_first_visits": first, "case2_training": ml2,
        **targets_frames,
    }.items():
        frame.to_csv(cleaned / f"{name}.csv", index=False)
    for name, frame in raw.items():
        where = flags[name].loc[flags[name].primary_reason.ne(""), [ROWS, "primary_reason", "reasons"]]
        where.merge(frame, on=ROWS, validate="one_to_one").to_csv(excluded / name, index=False)
    pairs.loc[~pairs.pair_status.eq("accepted")].to_csv(excluded / "case1_pairs.csv", index=False)
    future_mode.to_csv(excluded / "target_mode_after_cutoff.csv", index=False)
    per_section = grid.groupby("id", as_index=False).agg(grid_segments=("segment_id", "size"))
    per_section["unique_segments_with_any_vtd"] = per_section.id.map(history.groupby("id")["segment_id"].nunique()).fillna(0)
    per_section["unique_segments_without_vtd"] = per_section.grid_segments - per_section.unique_segments_with_any_vtd
    train_compare = []
    for target in targets.itertuples(index=False):
        train = targets_frames[f"target_{target.id}_training"]
        ready = targets_frames[f"target_{target.id}_with_profile"]
        fields = [col for col in ready if col in train and pd.api.types.is_numeric_dtype(ready[col])
                  and col not in ("id", "segment_id", "case")]
        # Constant values per section/window are recorded in table, with robust spread and coverage.
        for col in fields:
            a, b = train[col].dropna().astype(float), ready[col].dropna().astype(float)
            train_compare.append({"target_id": target.id, "feature": col,
                                  "train_known": len(a), "train_n": len(train), "target_known": len(b),
                                  "target_n": len(ready), "train_p05": a.quantile(.05),
                                  "train_median": a.median() if len(a) else np.nan, "train_p95": a.quantile(.95),
                                  "target_median": b.median() if len(b) else np.nan,
                                  "target_outside_train_support": int((b.lt(a.min() - 1e-9) | b.gt(a.max() + 1e-9)).sum())
                                  if len(a) else None})
    for name, frame in {
        "audit_files": audit, "audit_reasons": reasons, "audit_by_id_date": by_id_date, "audit_keys": accounting,
        "audit_columns": pd.DataFrame([row for name, frame in raw.items() for row in numeric_audit(frame, name)]),
        "target_readiness": metadata, "target_passports": targets, "visit_coverage": visits,
        "section_coverage": per_section, "target_feature_support": pd.DataFrame(train_compare),
        "pair_summary": pair_summary, "historical_hybrid": comparisons,
        "target_regime_coverage_by_year": scoped_mode.assign(year=scoped_mode[DAY].dt.year)
        .groupby(["id", "year"], as_index=False).agg(days=(DAY, "size"), first=(DAY, "min"), last=(DAY, "max"),
                                                      **{col: (col, "count") for col in MODE_FIELDS}),
        "vtd_suspicions": vtd_audit.loc[open_audit].groupby(["id", DAY], dropna=False)[[
            "date_text_conflict", "residual_gt_vtd_wall", "vtd_wall_ne_passport", "distance_fields_disagree",
            "section_length_diff_gt_0_1m", "depth_pct_outside_0_100", "depth_discrepancy_gt_2pp",
            "angle_outside_0_360", "negative_defect_size"]].sum().reset_index(),
        "mode_suspicions": mode.groupby("id", as_index=False)[[
            "length_mismatch", "diameter_mismatch_m_converted", "calculated_pressure_gt_100", "watercut_between_zero_and_one"
        ]].sum(),
    }.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    (out / "manifest.json").write_text(json.dumps({
        "input_sha256": {name: sha256((data_dir / name).read_bytes()).hexdigest() for name in FILES},
        "comparison_tolerance_mm": 1e-9, "vtd_depth_scale_2018": 100, "segment_m": 10,
    }, indent=2) + "\n")
    print(audit.to_string(index=False))
    print(metadata.to_string(index=False))
    print(pair_summary.to_string(index=False))


ORIGIN = "thickness_origin"
MEASURED = "measured"
NOMINAL_LABEL = "nominal_no_reported_defect"
UNKNOWN = "recorded_anomaly_without_thickness"
ANOMALY = r"аномали|вмятин|дефект|подрез|корроз|потеря металла|риска|царап|питтинг|язва|канавка"
LABEL_KEYS = [*SEGMENT_KEYS, DATE]


def full_survey_labels(grid, measured, audit, flags):
    # Build a grid only for valid survey IDs and dates. A pipe without a survey has no labels.
    visits = measured[["id", DATE]].drop_duplicates()
    census = visits.merge(grid[SEGMENT_KEYS + [NOMINAL]], on="id", validate="many_to_many")
    census = census.merge(measured, on=LABEL_KEYS, how="left", validate="one_to_one")
    text = audit[["Feature Description", "trait"]].fillna("").agg(" ".join, axis=1)
    relevant = text.str.contains(ANOMALY, case=False, regex=True)
    unresolved = audit.loc[flags.primary_reason.eq("invalid_residual_thickness") & relevant,
                           ["id", DAY, "adjusted_distance"]].copy()
    unresolved["segment_id"] = (unresolved["adjusted_distance"] // 10).astype(int)
    unresolved = unresolved.rename(columns={DAY: DATE})[LABEL_KEYS].drop_duplicates().assign(has_anomaly=True)
    census = census.merge(unresolved, on=LABEL_KEYS, how="left", validate="one_to_one")
    census["has_anomaly"] = census["has_anomaly"].eq(True)
    census[ORIGIN] = np.select(
        [census[THICKNESS].notna(), census["has_anomaly"]],
        [MEASURED, UNKNOWN], default=NOMINAL_LABEL,
    )
    census.loc[census[ORIGIN].eq(NOMINAL_LABEL), THICKNESS] = census[NOMINAL]
    census["records"] = census["records"].fillna(0).astype(int)
    return census.drop(columns=NOMINAL)


def completed_pairs(history):
    parts = []
    for id_, section in history.groupby("id"):
        visits = list(section.groupby(DATE))
        for (start, previous), (end, current) in zip(visits, visits[1:]):
            current = current[["segment_id", THICKNESS, ORIGIN]].merge(
                previous[["segment_id", THICKNESS, ORIGIN]].rename(columns={
                    THICKNESS: "previous_min_mm", ORIGIN: "previous_thickness_origin",
                }), on="segment_id", validate="one_to_one",
            )
            difference = current[THICKNESS] - current["previous_min_mm"]
            current["pair_status"] = np.select(
                [difference.isna(), difference.gt(1e-9)], ["unknown_thickness", "thickness_increased"],
                default="accepted",
            )
            current["delta_rate_mm_year"] = (-difference * DAYS_PER_YEAR / (end - start).days).where(
                current.pair_status.eq("accepted"),
            )
            current["target_minus_previous_mm"] = difference
            parts.append(current.assign(id=id_, start_date=start, target_date=end))
    return pd.concat(parts, ignore_index=True)


def run_labels(data_dir, cleaning, out):
    tables = cleaning / "tables"
    passport = pd.read_csv(tables / "passport.csv", parse_dates=["commissioned_at"])
    grid = pd.read_csv(tables / "grid.csv", parse_dates=["commissioned_at"])
    daily = pd.read_csv(tables / "regime_daily.csv", parse_dates=[DAY])
    targets = resolve_targets(passport)
    clean, flags, audit = clean_vtd(read_source(data_dir, VTD), passport, targets)
    measured = summarize_visits(clean)
    census = full_survey_labels(grid, measured, audit, flags)
    pairs = completed_pairs(census)
    first = first_visits(census, grid)
    strict = {}
    strict[1] = pairs.loc[pairs.pair_status.eq("accepted")].copy()
    strict[2] = first.loc[first[THICKNESS].notna() & first.age_years.gt(0)].copy()
    ready, _ = target_sets(targets, grid, census, daily)
    outputs = {"vtd_labels_full_grid": census, "case1_pairs_full_grid": pairs, "case2_first_visits_full_grid": first}
    visits = census.groupby(["id", DATE, ORIGIN], as_index=False).size()
    outputs["vtd_nominal_coverage"] = visits
    outputs["pair_status_by_origin"] = pairs.groupby(
        ["id", "start_date", "target_date", "previous_thickness_origin", ORIGIN, "pair_status"], as_index=False,
    ).size()
    summary = []
    for target in targets.itertuples(index=False):
        labels = strict[target.case].loc[strict[target.case].target_date.lt(target.target_date)]
        features = train_features(labels, target.case, grid, census, daily)
        cols = SEGMENT_KEYS + ["start_date", "target_date"]
        features = features.merge(labels[cols + [ORIGIN]], on=cols, validate="one_to_one")
        if target.case == 1:
            features = features.merge(labels[cols + ["previous_thickness_origin"]].rename(
                columns={"previous_thickness_origin": "prior_thickness_origin"},
            ), on=cols, validate="one_to_one")
            features["prior_observed"] = features["prior_thickness_origin"].eq(MEASURED)
            current = census.loc[census.id.eq(target.id) & census[DATE].lt(target.target_date)]
            latest = current.loc[current[DATE].eq(current[DATE].max()), SEGMENT_KEYS + [ORIGIN]]
            key = f"target_{target.id}_with_profile"
            ready[key] = ready[key].merge(latest.rename(columns={ORIGIN: "prior_thickness_origin"}),
                                          on=SEGMENT_KEYS, validate="one_to_one")
            ready[key]["prior_observed"] = ready[key]["prior_thickness_origin"].eq(MEASURED)
        else:
            outputs[f"target_{target.id}_training_mode_only"] = features.drop(columns=PROFILE_COLUMNS)
        outputs[f"target_{target.id}_training"] = features
        summary.append({
            "id": target.id, "target_date": target.target_date, "accepted_training": len(labels),
            "training_measured_target": int(labels[ORIGIN].eq(MEASURED).sum()),
            "training_nominal_target": int(labels[ORIGIN].eq(NOMINAL_LABEL).sum()),
            "training_passes": labels[["id", "start_date", "target_date"]].drop_duplicates().shape[0],
        })
    outputs.update(ready)
    outputs["training_summary"] = pd.DataFrame(summary)
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    print(outputs["training_summary"].to_string(index=False))
