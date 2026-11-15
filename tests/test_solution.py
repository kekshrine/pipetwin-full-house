import hashlib
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pipetwin import case2_elevation, solution
from pipetwin.forecast import History

ROOT = Path(__file__).resolve().parents[1]


def read_result(path):
    return pd.read_csv(path, dtype={"id": str})


class ResultFile(unittest.TestCase):
    def test_committed_result_matches_the_format(self):
        frame = read_result(ROOT / "result.csv")
        solution.check_result(frame)
        self.assertEqual(frame.groupby("id").size().to_dict(), {"1250009583": 1475, "1250022871": 525})
        self.assertTrue(frame.thickness_pred.between(0, 8.0).all())

    def test_check_rejects_broken_files(self):
        good = read_result(ROOT / "result.csv")
        broken = {
            "missing key": good.iloc[1:],
            "duplicate key": pd.concat([good, good.iloc[:1]]),
            "nan": good.assign(thickness_pred=good.thickness_pred.where(good.index != 5)),
            "header": good.rename(columns={"thickness_pred": "thickness"}),
        }
        for name, frame in broken.items():
            with self.subTest(name), self.assertRaises(ValueError):
                solution.check_result(frame)


class Components(unittest.TestCase):
    def test_blend_keeps_anchored_level_and_spread(self):
        rng = np.random.default_rng(0)
        base = rng.normal(6, 0.5, 300)
        parts = {"strict": base + rng.normal(0, .2, 300), "narrow": base + rng.normal(0, .2, 300),
                 "anchored": base + rng.normal(0, .2, 300)}
        out = solution.blend_case1(parts, 8.0)
        self.assertAlmostEqual(out.mean(), parts["anchored"].mean(), places=3)
        self.assertAlmostEqual(out.std(), parts["anchored"].std(), places=3)

    def test_case2_higher_route_is_thinner(self):
        grid = pd.DataFrame({"id": "x", "segment_id": range(600), "nominal_thickness_mm": 4.5,
                             "elevation_mid_m": np.linspace(50, 80, 600), "commissioned_at": pd.Timestamp("1993-01-01")})
        reference = pd.DataFrame({"segment_id": range(600), "thickness_pred": np.linspace(3, 4, 600)})
        pred = case2_elevation.forecast(grid, reference, "x", "2022-07-01").thickness_pred.to_numpy()
        self.assertTrue((np.diff(pred) <= 1e-12).all())
        self.assertLess(pred[-1], pred[0])

    def test_history_drops_the_target_date(self):
        vtd = pd.DataFrame({"date": pd.to_datetime(["2022-07-06", "2023-12-01"]), "main_id": "M"})
        labels = pd.DataFrame({"inspection_date": pd.to_datetime(["2022-07-06", "2023-12-01"])})
        history = History.before(vtd, labels, "2023-12-01")
        self.assertEqual(len(history.vtd), 1)
        self.assertEqual(len(history.labels), 1)


@unittest.skipUnless(os.environ.get("PIPETWIN_DATA"), "set PIPETWIN_DATA to the folder with the six raw CSV")
class EndToEnd(unittest.TestCase):
    def test_pipeline_rebuilds_the_committed_result(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "result.csv"
            solution.run(Path(os.environ["PIPETWIN_DATA"]), Path(folder) / "work", out)
            digest = hashlib.sha256(out.read_bytes()).hexdigest()
        self.assertEqual(digest, hashlib.sha256((ROOT / "result.csv").read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
