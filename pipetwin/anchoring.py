"""Case-1 odometer anchoring by welds of the neighbouring sections inspected on the target date."""

from copy import copy

import numpy as np
import pandas as pd

from pipetwin import forecast as rf

RESIDUAL_WINDOW_M = 2.0
UNANCHORED_SHARE = 0.15  # insurance against a non-linear odometer path between the anchors
ANCHOR_JOINTS = 3
LENGTH_TOLERANCE_M = 0.3


def joint_starts(defects):
    """Odometer position of each joint start in one run: distance minus position in the joint."""
    start = defects["Distance_adjusted"] - defects["Relative Distance"]
    return defects.assign(start=start).groupby("Section Number").agg(
        start=("start", "median"), pipe_length=("Pipe Length", "median"))


def joint_offsets(previous, current, tolerance_m=LENGTH_TOLERANCE_M):
    """Shift (current minus previous) of joints reported by both runs with a consistent length."""
    both = joint_starts(previous).join(joint_starts(current), lsuffix="_prev", rsuffix="_cur", how="inner")
    known = both["pipe_length_prev"].notna() & both["pipe_length_cur"].notna()
    consistent = (both["pipe_length_prev"] - both["pipe_length_cur"]).abs().le(tolerance_m)
    both = both.loc[~known | consistent]
    return pd.DataFrame({"start_prev": both["start_prev"], "shift_m": both["start_cur"] - both["start_prev"]})


def neighbour_offsets(vtd, main_id, target_id, prev_date, cur_date):
    """Joint shifts between two runs on every section of the main except the target, in global metres."""
    rows = []
    for id_, section in vtd.loc[vtd.main_id.eq(main_id) & vtd.id.ne(target_id)].groupby("id"):
        prev, cur = section.loc[section.date.eq(prev_date)], section.loc[section.date.eq(cur_date)]
        if prev.empty or cur.empty:
            continue
        offsets = joint_offsets(prev, cur)
        base = float(prev["accumulated_value"].iloc[0])
        rows.append(offsets.assign(id=id_, section_start=base, global_prev=offsets["start_prev"] + base))
    if not rows:
        return pd.DataFrame(columns=["id", "start_prev", "shift_m", "section_start", "global_prev"])
    return pd.concat(rows, ignore_index=True).sort_values("global_prev")


def anchors_around(offsets, start_m, count=ANCHOR_JOINTS):
    """Median shift of the last ``count`` joints of the sections before the target and the first after it."""
    before, after = offsets.loc[offsets.section_start.lt(start_m)], offsets.loc[offsets.section_start.gt(start_m)]
    low = (float(before.global_prev.tail(count).median()), float(before.shift_m.tail(count).median())) if len(before) else None
    high = (float(after.global_prev.head(count).median()), float(after.shift_m.head(count).median())) if len(after) else None
    if low is None and high is None:
        return None
    low, high = low or high, high or low
    return {"x0": low[0], "lo": low[1], "x1": high[0], "hi": high[1]}


def path_offset(global_m, anchors):
    """Shift along the main: linear between the anchors, constant outside them."""
    if anchors is None:
        return np.zeros(len(np.atleast_1d(global_m)))
    span = anchors["x1"] - anchors["x0"]
    u = np.clip((np.asarray(global_m, float) - anchors["x0"]) / span, 0, 1) if span > 0 else 0.0
    return anchors["lo"] + (anchors["hi"] - anchors["lo"]) * u


def anchored_run(run, anchors):
    moved = copy(run)
    moved.global_m = run.global_m + path_offset(run.global_m, anchors)
    return moved


def forecast(vtd, grid, labels, target_id, target_date, window_m=RESIDUAL_WINDOW_M):
    """Anchored case-1 forecast; reads no row of ``target_id`` dated ``target_date`` or later."""
    target_date = pd.Timestamp(target_date)
    own_late = vtd.id.eq(target_id) & vtd.date.ge(target_date)
    usable = vtd.loc[~own_late]
    history = rf.History.before(usable, labels, target_date)
    prev_date = history.vtd.loc[history.vtd.id.eq(target_id), "date"].max()
    main_id = history.vtd.loc[history.vtd.id.eq(target_id), "main_id"].iloc[0]
    run = rf.SectionRun(history.vtd, grid, target_id, prev_date)
    offsets = neighbour_offsets(usable, main_id, target_id, prev_date, target_date)
    anchors = anchors_around(offsets, run.start)
    prior = rf.odometer_prior(history)
    rate = rf.loss_rate(rf.past_transition_tables(history, grid, prior))
    years = (target_date - prev_date).days / rf.DAYS_PER_YEAR
    anchored = rf.expected_minimum(anchored_run(run, anchors), window_m)
    base = rf.final_forecast(run, prior)
    raw = (1 - UNANCHORED_SHARE) * anchored + UNANCHORED_SHARE * base
    centers = run.start + 10 * np.arange(run.n) + 5
    frame = run.keys.assign(thickness_pred=np.clip(raw - rate * years, 0.0, run.nominal), raw_mm=raw,
                            anchored_raw_mm=anchored, unanchored_base_mm=base,
                            shift_m=path_offset(centers, anchors), loss_mm=rate * years,
                            start_date=prev_date, target_date=target_date, source="anchored_expected_minimum")
    note = {"anchors": anchors, "neighbour_joints": int(len(offsets)), "window_m": window_m,
            "unanchored_share": UNANCHORED_SHARE,
            "loss_rate_mm_per_year": rate, "last_run": str(pd.Timestamp(prev_date).date())}
    return frame, note
