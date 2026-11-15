"""Case-2 reference models: features, spatial folds before T, level and spread of the case-2 forecast."""

from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from pipetwin.data import (
    closed_survey_mask,
    daily_regime,
    DATE,
    feature_frame,
    feature_sets,
    GEO_FEATURES,
    MEASURED,
    MODE_FILE,
    NOMINAL,
    ORIGIN,
    resolve_targets,
    ridge_rate,
    SEGMENT_KEYS,
    THICKNESS,
    VTD,
)

GEOMETRY = [
    "defect_length_max_mm", "defect_width_max_mm", "defect_depth_p90_pct",
    "pitting_share_known", "internal_share_known",
]
NEIGHBORHOOD = [
    "defect_records_per_m_50m", "nearest_observed_center_m",
    "local_min_minus_section_median_mm", "neighbor_min_iqr_50m_mm",
]
MICROTERRAIN = [
    "grade_before_50m", "grade_after_50m", "grade_change_100m",
    "depression_below_shoulders_100m_m", "detrended_roughness_100m_m",
]
REGIME_VARIATION = [
    "watercut_iqr", "velocity_p10", "velocity_iqr", "temperature_iqr",
    "recent_365d_mode_coverage",
]


def microterrain_features(data_dir, grid):
    profile = pd.read_csv(data_dir / "altitudes_south.csv", usecols=["id", "total_distance", "z"])
    parts = []
    offsets = np.arange(-50, 51, 10)
    for id_, section in grid.groupby("id"):
        points = profile.loc[profile["id"].eq(id_)].sort_values("total_distance")
        distances = section["midpoint_m"].to_numpy()[:, None] + offsets
        heights = np.interp(distances, points["total_distance"], points["z"], left=np.nan, right=np.nan)
        middle = heights[:, 5]
        before = (middle - heights[:, 0]) / 50
        after = (heights[:, -1] - middle) / 50
        shoulder = np.minimum(heights[:, :5].max(axis=1), heights[:, 6:].max(axis=1))
        slope = heights @ offsets / (offsets @ offsets)
        residual = heights - slope[:, None] * offsets
        parts.append(section[SEGMENT_KEYS].assign(
            grade_before_50m=before, grade_after_50m=after, grade_change_100m=after - before,
            depression_below_shoulders_100m_m=np.maximum(shoulder - middle, 0),
            detrended_roughness_100m_m=residual.std(axis=1),
        ))
    return pd.concat(parts, ignore_index=True)


SIDE_MIN = ['paot_inner_min', 'paot_outer_min', 'paot_side_gap']
SIDE_ZONE = ['paot_inner_deep_occupied', 'paot_inner_deep_run', 'paot_outer_deep_occupied']
BASIN = ['paot_basin_elevation', 'paot_basin_extent', 'paot_escape_steepness']
ASYMMETRY = ['paot_escape_rise_diff', 'paot_escape_distance_diff']
BENDS = ['horizontal_turn_30m', 'horizontal_turn_100m', 'vertical_turn_30m', 'vertical_turn_100m']


def bend_features(data_dir, grid):
    raw = pd.read_csv(data_dir / 'altitudes_south.csv').sort_values('total_distance')
    profiles = dict(iter(raw.groupby('id')))
    parts = []
    for id_, cells in grid.groupby('id'):
        g = profiles[id_]
        x = cells.midpoint_m.to_numpy()[:, None]
        result = cells[SEGMENT_KEYS].copy()
        for radius in (30, 100):
            sample = x + [-radius, 0, radius]
            lon, lat, z = (np.interp(sample, g.total_distance, g[k], left=np.nan, right=np.nan)
                           for k in ['lon', 'lat', 'z'])
            xy = np.stack([np.diff(lon, axis=1)*np.cos(np.deg2rad(lat[:, 1:2])), np.diff(lat, axis=1)], axis=2)
            a, b = xy[:, 0], xy[:, 1]
            cross = a[:, 0]*b[:, 1] - a[:, 1]*b[:, 0]
            turn = np.arctan2(np.abs(cross), np.sum(a*b, axis=1))
            result[f'horizontal_turn_{radius}m'] = np.where((np.linalg.norm(a, axis=1) > 0)
                                                            & (np.linalg.norm(b, axis=1) > 0), turn, np.nan)
            result[f'vertical_turn_{radius}m'] = np.diff(np.arctan(np.diff(z, axis=1) / radius), axis=1).ravel()
        parts.append(result)
    return pd.concat(parts, ignore_index=True)


def basin_features(data_dir, grid):
    profile = pd.read_csv(data_dir / 'altitudes_south.csv').sort_values('total_distance')
    profiles = dict(iter(profile.groupby('id')))
    parts = []
    steps = np.arange(10., 151., 10.)
    for id_, cells in grid.groupby('id'):
        coords = profiles[id_]
        center = cells.midpoint_m.to_numpy()
        distances = np.r_[-steps[::-1], 0., steps]
        z = np.interp(center[:, None] + distances, coords.total_distance, coords.z, left=np.nan, right=np.nan)
        out = cells[SEGMENT_KEYS].copy()
        valid = np.isfinite(z).all(axis=1)
        z = z[valid]
        split = len(steps)
        arms = np.stack([z[:, :split][:, ::-1], z[:, split+1:]], axis=1)
        rel = arms - z[:, split, None, None]
        rim = np.maximum(rel.max(axis=2), 0)
        rise_index = rel.argmax(axis=2)
        crest_distance = steps[rise_index]
        lower_rim = rim.min(axis=1)
        is_basin = (lower_rim > 0) & (rel[:, :, 0] >= -.05).all(axis=1)
        # Distance to the first crossing of half the lower shoulder height.
        at_half = rel >= lower_rim[:, None, None] / 2
        half_width = steps[at_half.argmax(axis=2)].sum(axis=1)
        out.loc[valid, BASIN[0]] = np.where(is_basin, lower_rim, 0)
        out.loc[valid, BASIN[1]] = np.where(is_basin, half_width, 0)
        out.loc[valid, BASIN[2]] = np.where(is_basin, (rim / crest_distance).min(axis=1), 0)
        out.loc[valid, ASYMMETRY[0]] = np.where(is_basin, np.ptp(rim, axis=1), 0)
        out.loc[valid, ASYMMETRY[1]] = np.where(is_basin, np.ptp(crest_distance, axis=1), 0)
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


WINDOW = ["id", "start_date", "target_date"]
ANGLE = ["angle_mean_sin", "angle_mean_cos", "angle_coverage"]
EPISODES = ["flow_observed_fraction", "stopped_share_observed", "longest_observed_stop_days",
            "velocity_observed_fraction", "recent_velocity_median", "recent_temperature_median"]
COUNTS = ["log_input_count", "defect_records_per_m_50m"]


def feature_groups():
    base1 = feature_sets(1)
    core = base1["basic"] + ["prior_median_50m", "prior_median_100m", "input_p10", "input_inner_min"]
    defects = core + GEOMETRY + [name for name in NEIGHBORHOOD if name not in COUNTS]
    return {
        1: {"core": core, "defects": defects, "defects_count": defects + COUNTS,
            "defects_angle": defects + ANGLE,
            "rich": defects + ANGLE + MICROTERRAIN + REGIME_VARIATION + EPISODES},
        2: {"mode_only": feature_sets(2)["mode_only"] + REGIME_VARIATION + EPISODES,
            "terrain": feature_sets(2)["spatial"] + MICROTERRAIN,
            "terrain_mode": feature_sets(2)["spatial_mode"] + MICROTERRAIN + REGIME_VARIATION + EPISODES},
    }


SEED = 42
BLOCK_SEGMENTS = 25
BUFFER_SEGMENTS = 10
FOLDS = 4
PRED = "thickness_pred"
KEYS = ["case", *SEGMENT_KEYS, "start_date", "target_date"]
INCUMBENT = {1: "hybrid_10m", 2: "terrain_ridge"}
FORWARD = {
    1: ("2022-07-06", "2023-12-01"),
    2: ("2018-09-23", "2023-08-12"),
}
ESTIMATORS = {
    "catboost": dict(iterations=220, depth=3, learning_rate=0.035, l2_leaf_reg=12, boosting_type="Plain",
                     loss_function="RMSE", random_seed=SEED, thread_count=2, allow_writing_files=False,
                     verbose=False),
    "lightgbm": dict(n_estimators=220, max_depth=3, num_leaves=7, min_child_samples=24, learning_rate=0.035,
                     reg_lambda=8, deterministic=True, force_col_wise=True, verbosity=-1,
                     random_state=SEED, n_jobs=2),
    "xgboost": dict(n_estimators=220, max_depth=3, min_child_weight=16, learning_rate=0.035,
                    reg_lambda=12, tree_method="hist", objective="reg:squarederror", random_state=SEED,
                    n_jobs=2),
}
REFERENCE_COLUMNS = {
    "hybrid_10m": ["input_min", "previous_observed"],
    "last_min": ["input_min", "days_since_last_observation"],
    "median_20m": ["input_min", "days_since_last_observation"],
    "median_20m_with_rate": ["input_min", "days_since_last_observation", "horizon_years"],
    "local_extrapolation": ["initial_thickness", "previous_annual_loss", "horizon_years"],
    "ridge_residual": ["hybrid_10m", "previous_observed", "horizon_years", *GEO_FEATURES],
    "nominal": [NOMINAL],
    "pooled_loss_rate": [NOMINAL, "horizon_years"],
    "terrain_ridge": [NOMINAL, "horizon_years", *GEO_FEATURES],
}
BASELINES = {
    1: ("hybrid_10m", "last_min", "median_20m", "median_20m_with_rate", "local_extrapolation", "ridge_residual"),
    2: ("nominal", "pooled_loss_rate", "terrain_ridge"),
}


@dataclass(frozen=True)
class Candidate:
    name: str
    algorithm: str
    features: str = "basic"
    formulation: str = "direct"
    overrides: dict = field(default_factory=dict)

    def params(self):
        if self.algorithm in ESTIMATORS:
            return {**ESTIMATORS[self.algorithm], **self.overrides}
        if self.algorithm in ("ridge", "extra_trees"):
            return self.overrides.copy()
        if self.algorithm == "ridge_residual":
            return {"l2": 10}
        if self.algorithm == "terrain_ridge":
            return {"l2": 10, "loss_rate_floor": 0}
        return {}


def blocked_splits(data, *, block_segments=BLOCK_SEGMENTS, buffer_segments=BUFFER_SEGMENTS):
    """Each segment is evaluated once. Remove training neighbors across block edges."""
    if block_segments <= 0 or buffer_segments < 0:
        raise ValueError("Invalid block width or buffer")
    # Pair each (section, spatial block) to one fold, balancing with a fixed seed.
    blocks = data[["id"]].assign(block=data["segment_id"] // block_segments)
    unique = blocks.drop_duplicates().sort_values(["id", "block"])
    if len(unique) < FOLDS:
        raise ValueError(f"At least {FOLDS} occupied blocks required")
    allocation = np.random.default_rng(SEED).permutation(len(unique)) % FOLDS
    folds = blocks.merge(unique.assign(fold=allocation), on=["id", "block"], sort=False)["fold"]
    folds.index = data.index
    for fold in range(FOLDS):
        test = data.loc[folds.eq(fold)]
        train = data.loc[folds.ne(fold)]
        keep = pd.Series(True, index=train.index)
        for id_, part in test.assign(block=test["segment_id"] // block_segments).groupby("id"):
            x = train.loc[train["id"].eq(id_), "segment_id"]
            for block in part["block"].unique():
                keep.loc[x.index] &= ~x.between(block * block_segments - buffer_segments,
                                                (block + 1) * block_segments + buffer_segments - 1)
        yield train.loc[keep], test


def rate_reference(train, test, model):
    if model == "nominal":
        return test[NOMINAL].to_numpy()
    rate = ((1 - train[THICKNESS] / train[NOMINAL]).clip(lower=0) / train["horizon_years"])
    train = train.assign(loss_rate=rate)
    estimate = float(rate.median()) if model == "pooled_loss_rate" else ridge_rate(train, test)
    return (test[NOMINAL] * (1 - estimate * test["horizon_years"])).to_numpy()


def clipped(test, prediction):
    return np.clip(np.asarray(prediction, dtype=float), 0, test[NOMINAL].to_numpy())


def base_prediction(case, train, test, model):
    if case != 2:
        raise ValueError("Reference baselines are defined for case 2")
    return clipped(test, rate_reference(train, test, model))


def make_estimator(candidate):
    if candidate.algorithm != "catboost":
        raise ValueError(f"Unknown algorithm: {candidate.algorithm}")
    from catboost import CatBoostRegressor
    return CatBoostRegressor(**candidate.params())


def fit_predict(case, candidate, train, test, columns=None):
    if candidate.algorithm in BASELINES[case]:
        return base_prediction(case, train, test, candidate.algorithm), None, []
    y = train[THICKNESS].to_numpy()
    restore = None
    scale = None
    if candidate.formulation == "loss":
        # Case 2 transfers annual fractional loss to pipes with a different age/nominal.
        train_scale = train[NOMINAL] * train["horizon_years"] if case == 2 else 1.0
        scale = test[NOMINAL] * test["horizon_years"] if case == 2 else 1.0
        y = (train["initial_thickness"] - y) / train_scale
        restore = test["initial_thickness"]
    if candidate.formulation == "residual":
        reference = pd.Series(index=train.index, dtype=float)
        for inner_train, inner_test in blocked_splits(train):
            reference.loc[inner_test.index] = base_prediction(case, inner_train, inner_test, INCUMBENT[case])
        if reference.isna().any():
            raise ValueError("Missing cross-fitted terrain reference")
        y = y - reference.to_numpy()
        restore = base_prediction(case, train, test, INCUMBENT[case])
    if columns is None:
        columns = feature_sets(case)[candidate.features]
    # Drop constants and entirely missing columns using training rows only.
    active = [name for name in columns if train[name].nunique(dropna=True) > 1]
    if not active:
        prediction = np.full(len(test), np.mean(y))
        model = None
    else:
        model = make_estimator(candidate)
        model.fit(train[active].astype(float), y)
        prediction = model.predict(test[active].astype(float))
    if candidate.formulation == "loss":
        prediction = restore - prediction * scale
    if candidate.formulation == "residual":
        prediction = restore + prediction
    return clipped(test, prediction), model, active


INPUT_NAMES = {
    "pot_south.csv", "vtd_south_train.csv", "altitudes_south.csv", "uzt_south.csv", "repcorr_south.csv",
    "tech_mode_for_embedings_with_length_with_temperatures_ids16.csv",
}


def predict_fold(case, train, test, *, columns=None):
    control = INCUMBENT[case]
    baseline = base_prediction(case, train, test, control)
    group = "defects_count" if case == 1 else "terrain"
    candidate = Candidate(group, "catboost", formulation="residual" if case == 1 else "loss")
    if columns is None:
        columns = feature_groups()[case][group]
    prediction, _, _ = fit_predict(case, candidate, train, test, columns)
    return {control: baseline, "catboost": prediction,
            "baseline_75_catboost_25": 0.75 * baseline + 0.25 * prediction}


RAW = Path('../../raw_data')
FOOTPRINT = ['thin_length_norm', 'thin_loss_fraction', 'thin_volume_proxy', 'thin_cross_boundary']
SITE = ['id', 'segment_id']
WATER = ['water_flow_weighted_pct', 'wet_speed', 'wet_low_speed_share']
INTERACTIONS = ['wet_basin', 'wet_asymmetry']


def daily_water(raw):
    data = raw.copy()
    q = data.q
    volume_oil = 1000 * data.q_oil / data.density_oil.where(data.density_oil.gt(0))
    fraction = (1 - volume_oil / q.where(q.gt(0)))
    valid = q.gt(0) & data.q_oil.ge(0) & fraction.between(0, 1)
    data['water_fraction'] = fraction.where(valid)
    data['water_flow'] = (q * fraction).where(valid)
    data['flow_valid'] = q.where(valid)
    data['wet_speed'] = (fraction * data.v_liquid_true).where(valid & data.v_liquid_true.ge(0))
    data['wet_low_speed_share'] = (fraction * data.v_liquid_true.lt(.2)).where(valid & data.v_liquid_true.ge(0))
    data['day'] = pd.to_datetime(data.calc_date, format='mixed').dt.normalize()
    fields = ['water_fraction', 'water_flow', 'flow_valid', 'wet_speed', 'wet_low_speed_share']
    return data.groupby(['id', 'day'], as_index=False)[fields].median()


def windows_water(windows, daily):
    rows = []
    for id_, start, target in windows[WINDOW].drop_duplicates().itertuples(index=False, name=None):
        a, b = pd.Timestamp(start), pd.Timestamp(target)
        part = daily.loc[daily.id.eq(id_) & daily.day.ge(a) & daily.day.lt(b)]
        q = part.flow_valid.sum(min_count=1)
        rows.append(dict(id=id_, start_date=start, target_date=target,
                         water_flow_weighted_pct=100 * part.water_flow.sum(min_count=1) / q,
                         water_days=part.flow_valid.count(), water_median_pct=100 * part.water_fraction.median(),
                         wet_speed=part.wet_speed.mean(), wet_low_speed_share=part.wet_low_speed_share.mean()))
    return pd.DataFrame(rows)


def add_inputs(frame, geometry, daily):
    frame = frame.merge(geometry, on=SITE, how='left', validate='many_to_one')
    frame = frame.merge(windows_water(frame, daily), on=WINDOW, validate='many_to_one')
    frame['wet_basin'] = frame.wet_low_speed_share * frame[BASIN[0]]
    frame['wet_asymmetry'] = frame.wet_low_speed_share * frame[ASYMMETRY[0]]
    return frame


def hurdle(train, test, cols):
    from catboost import CatBoostClassifier

    x = train[cols].astype(float)
    test_x = test[cols].astype(float)
    observed = train.is_measured
    classifier = CatBoostClassifier(iterations=150, depth=3, learning_rate=.035, l2_leaf_reg=12,
                                    random_seed=42, thread_count=2, allow_writing_files=False, verbose=False)
    classifier.fit(x, observed)
    p = classifier.predict_proba(test_x)[:, 1]
    depth, _, _ = fit_predict(2, Candidate('severity', 'catboost', formulation='loss'),
                              train.loc[observed], test, cols)
    nominal = test.nominal_thickness_mm.to_numpy()
    return nominal - p * (nominal - depth)


NOMINAL = "nominal_thickness_mm"
C1 = feature_groups()[1]["defects_count"]
C2 = feature_groups()[2]["terrain"] + BENDS + ASYMMETRY
WATER_PROFILE = WATER + INTERACTIONS


def case_features(case, requests, data_dir, grid, history, census):
    forecasts = []
    daily = daily_regime(data_dir, grid)
    obs = None
    for id_, start, target in requests[WINDOW].drop_duplicates().itertuples(index=False, name=None):
        section = grid.loc[grid.id.eq(id_)]
        frame = feature_frame(case, section, history, obs, daily, start, target)
        forecasts.append(frame)
    features = pd.concat(forecasts, ignore_index=True)
    return features


def dataset(data_dir, cleaning, nominal, cases=(1, 2)):
    grid = pd.read_csv(cleaning / "tables/grid.csv", parse_dates=["commissioned_at"])
    passport = pd.read_csv(cleaning / "tables/passport.csv", parse_dates=["commissioned_at"])
    history = pd.read_csv(cleaning / "tables/vtd_segment_minima.csv", parse_dates=[DATE])
    census = pd.read_csv(nominal / "vtd_labels_full_grid.csv", parse_dates=[DATE])
    targets = resolve_targets(passport)
    if closed_survey_mask(history.id, history[DATE], targets).any():
        raise ValueError("Clean history contains official forecast labels")
    clean = pd.read_csv(cleaning / "tables/vtd.csv", low_memory=False)
    if closed_survey_mask(clean.id, pd.to_datetime(clean.date), targets).any() or (pd.to_datetime(clean.date) < pd.to_datetime(clean.commissioned_at)).any():
        raise ValueError("Unexpected inspection in the cleaned inputs")
    raw_columns = pd.read_csv(data_dir / VTD, nrows=0).columns
    geo = microterrain_features(data_dir, grid).merge(bend_features(data_dir, grid), on=SEGMENT_KEYS)
    geo = geo.merge(basin_features(data_dir, grid), on=SEGMENT_KEYS, validate="one_to_one")
    water = daily_water(pd.read_csv(data_dir / MODE_FILE).rename(columns={"simple_section_id": "id"}))
    data, blind = {}, {}
    with TemporaryDirectory(prefix="pipetwin-clean-models-") as folder:
        safe = Path(folder)
        for name in ("pot_south.csv", "altitudes_south.csv", MODE_FILE):
            (safe / name).symlink_to((data_dir / name).resolve())
        clean[raw_columns].to_csv(safe / VTD, index=False)
        for target in targets.itertuples(index=False):
            if target.case not in cases:
                continue
            case = target.case
            name = "case1_pairs_full_grid" if case == 1 else "case2_first_visits_full_grid"
            labels = pd.read_csv(nominal / f"{name}.csv", parse_dates=["start_date", "target_date"])
            if case == 1:
                keep = [THICKNESS, ORIGIN, "pair_status", "delta_rate_mm_year", "previous_thickness_origin"]
            else:
                keep = [THICKNESS, ORIGIN]
            target_table = pd.read_csv(nominal / f"target_{target.id}_with_profile.csv",
                                       parse_dates=["start_date", "target_date"])
            requests = pd.concat([labels[WINDOW], target_table[WINDOW]]).drop_duplicates()
            features = case_features(case, requests, safe, grid, history, census)
            features = add_inputs(features, geo, water)
            features = features.replace([np.inf, -np.inf], np.nan)
            data[case] = features.merge(labels[[*SEGMENT_KEYS, "start_date", "target_date", *keep]],
                                        on=KEYS[1:], validate="one_to_one")
            blind[case] = features.merge(target_table[KEYS], on=KEYS, validate="one_to_one")
            if len(blind[case]) != len(target_table) or blind[case].duplicated(KEYS).any():
                raise ValueError("Blind prediction keys differ from the cleaned target grid")
    return data, blind


def predictors(case, train, test):
    yield "nominal", base_prediction(2, train, test, "nominal")
    yield "common_rate_no_profile", base_prediction(2, train, test, "pooled_loss_rate")
    yield "terrain_ridge", base_prediction(2, train, test, "terrain_ridge")
    cols = feature_groups()[2]["terrain"]
    yield "terrain_catboost", predict_fold(2, train, test, columns=cols)["baseline_75_catboost_25"]
    yield "paot_blend", predict_fold(2, train, test, columns=C2)["baseline_75_catboost_25"]
    yield "paot_water_blend", predict_fold(2, train, test, columns=C2 + WATER_PROFILE)["baseline_75_catboost_25"]
    labeled = train.assign(is_measured=train[ORIGIN].eq(MEASURED))
    if labeled.is_measured.nunique() == 2:
        yield "hurdle_paot", hurdle(labeled, test, C2)
        yield "hurdle_paot_water", hurdle(labeled, test, C2 + WATER_PROFILE)


def training(frame, case, policy):
    keep = frame[THICKNESS].notna()
    if policy == "measured":
        keep &= frame[ORIGIN].eq(MEASURED)
    return frame.loc[keep]


def evaluate(examples, blind):
    predictions, splits = [], []
    for case, frame in examples.items():
        early, late = map(pd.Timestamp, FORWARD[case])
        initial = frame.loc[frame.target_date.eq(early)]
        history = frame.loc[frame.target_date.eq(late)]
        test_ids = set(history.id)
        transfer_train = initial if case == 1 else initial.loc[~initial.id.isin(test_ids)]
        windows = [("blocked_oof", f, a, b) for f, (a, b) in enumerate(blocked_splits(initial))]
        windows.append(("chronological", 0, transfer_train, history))
        official = blind[case]
        available = frame.loc[frame.target_date.lt(official.target_date.min())]
        windows.append(("official_unscored", 0, available, official))
        for phase, fold, train, test in windows:
            fields = [c for c in [THICKNESS, ORIGIN, "pair_status", "delta_rate_mm_year",
                                  "previous_thickness_origin"] if c in test]
            output = test[KEYS + fields].copy()
            x = test.drop(columns=fields)
            for policy in ("measured", "known"):
                fit = training(train, case, policy)
                splits.append(fit[KEYS].assign(phase=phase, fold=fold, policy=policy, role="train"))
                splits.append(test[KEYS].assign(phase=phase, fold=fold, policy=policy, role="test"))
                for model, pred in predictors(case, fit, x):
                    if not np.isfinite(pred).all() or len(pred) != len(x):
                        raise ValueError(f"Invalid forecasts: {case}/{phase}/{model}")
                    predictions.append(output.assign(
                        phase=phase, fold=fold, training_policy=policy, training_rows=len(fit),
                        model=model, thickness_pred=np.asarray(pred),
                    ))
    return pd.concat(predictions, ignore_index=True), pd.concat(splits, ignore_index=True)


def verify_forecasts(forecasts):
    for (case, phase, policy), group in forecasts.groupby(["case", "phase", "training_policy"]):
        expected = group[KEYS].drop_duplicates().set_index(KEYS).sort_index().index
        if phase == "official_unscored" and group[THICKNESS].notna().any():
            raise ValueError("Official forecast unexpectedly has evaluation labels")
        for model, rows in group.groupby("model"):
            keys = rows.set_index(KEYS).sort_index().index
            if not keys.is_unique or not keys.equals(expected):
                raise ValueError(f"Non-matching evaluation keys: {case}/{phase}/{policy}/{model}")


def run(data_dir, cleaning, nominal, out):
    examples, blind = dataset(Path(data_dir), Path(cleaning), Path(nominal), cases=(2,))
    forecasts, _ = evaluate({2: examples[2]}, blind)
    verify_forecasts(forecasts)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    official = forecasts.phase.eq("official_unscored")
    forecasts.loc[~official].to_csv(out / "predictions_historical.csv", index=False, float_format="%.12g")
    forecasts.loc[official].drop(columns=[THICKNESS, ORIGIN]).to_csv(
        out / "official_forecasts.csv", index=False, float_format="%.12g")
