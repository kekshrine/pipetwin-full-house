"""Case 2 forecast from the large-scale route elevation.

The metric is Pearson r² within one pipe, so only the strength of the along-pipe association matters,
not its sign. On every inspected section of the pool (MAIN-1 2014, MAIN-2 2018/2022/2023,
MAIN-3 2023) the route elevation smoothed over about 2 km (201 segments) had the largest mean r² among
the candidates: 0.031 on the six long sections (300+ segments, min 0.006), with the same sign on all
six (higher route -> thinner wall), against 0.0095 for the two-stage PAOT model on its sections.
On 1250009583 the route is high (72-78 m) in the first 6 km, at the warm inlet, and low (52-56 m) after.

The forecast keeps the level and spread of the selected comparison model (for MAE) and takes the
ranking from the smoothed elevation: higher -> thinner.
"""

import numpy as np
import pandas as pd

WINDOW_SEGMENTS = 201


def smoothed_elevation(section, window=WINDOW_SEGMENTS):
    z = section["elevation_mid_m"].interpolate(limit_direction="both").to_numpy(float)
    pad = window // 2
    return np.convolve(np.pad(z, pad, mode="edge"), np.ones(window) / window, mode="valid")[:len(z)]


def forecast(grid, reference, id_, target_date):
    """Elevation-ranked forecast with the mean and spread of ``reference`` (same keys, column thickness_pred)."""
    section = grid.loc[grid.id.eq(id_)].sort_values("segment_id").reset_index(drop=True)
    nominal = float(section["nominal_thickness_mm"].iloc[0])
    level = reference.set_index("segment_id")["thickness_pred"].reindex(section.segment_id).to_numpy(float)
    z = smoothed_elevation(section)
    score = (z - z.mean()) / (z.std() if z.std() > 0 else 1.0)
    spread = np.nanstd(level) if np.nanstd(level) > 0 else 0.05 * nominal
    pred = np.clip(np.nanmean(level) - spread * score, 0.1 * nominal, nominal)
    return section[["id", "segment_id"]].assign(thickness_pred=pred, elevation_smoothed_m=z,
                                                start_date=section["commissioned_at"].iloc[0],
                                                target_date=pd.Timestamp(target_date), source="elevation_2km")
