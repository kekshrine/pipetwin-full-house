"""Local web interface for the final forecast.

    python -m pipetwin.web --data-dir /path/to/raw-csv --open

On start the pipeline builds ``result.csv`` from the raw CSV (the same run as ``python -m pipetwin``).
The two test pipes at their forecast dates show the submitted forecast. Any other pipe or date uses the
same building blocks without the parts tied to the test pipes: case 1 is the forecast from past
inspections only, case 2 ranks segments by the smoothed route elevation with a terrain-based level.
"""

import argparse
from datetime import date
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import webbrowser

import pandas as pd

from pipetwin import case2_elevation, solution
from pipetwin import forecast as rf

WEB = Path(__file__).parent / "web"
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "raw_data"
FINAL = {1: "итоговая модель сдачи (смесь трёх прогнозов)", 2: "итоговая модель сдачи (высота трассы 2 км)"}
GENERIC = {1: "только прошлые ВТД: учёт сдвига прибора + соседи + рост", 2: "высота трассы 2 км, уровень по рельефу"}
CASE_NAMES = {1: "«После ВТД»", 2: "«От ввода»"}


class Forecasts:
    def __init__(self, data_dir, work_dir):
        work_dir = Path(work_dir)
        result = solution.run(Path(data_dir), work_dir, work_dir / "result.csv")
        self.vtd, self.grid, _, self.labels = rf.load_tables(work_dir / "cleaning", work_dir / "labels")
        self.result = result.assign(id=result["id"].astype(str))
        self.targets = {t["id"]: t for t in solution.TARGETS}
        # Sections whose every raw inspection predates commissioning are left out, as in the first prototype.
        raw = pd.read_csv(Path(data_dir) / "vtd_south_train.csv", usecols=["id", "date"], dtype={"id": str})
        raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
        commissioned = self.grid.groupby("id")["commissioned_at"].first()
        early = raw.groupby("id")["date"].max().lt(commissioned.reindex(raw["id"].unique()))
        ids = [pid for pid in commissioned.index if pid not in set(early[early].index)]
        self.sections = sorted(ids, key=lambda pid: (pid not in self.targets, pid))

    def catalog(self):
        with_history = set(self.labels["id"].astype(str))
        out = []
        for pid in self.sections:
            item = {"id": pid, "has_history": pid in with_history, "target": pid in self.targets}
            if pid in self.targets:
                item.update(case=self.targets[pid]["case"], date=self.targets[pid]["target_date"])
            out.append(item)
        return out

    def calculate(self, payload):
        pid = str(payload.get("id", "")).strip()
        if pid not in self.sections:
            raise ValueError("Выберите трубу из списка.")
        try:
            case = int(payload.get("case", ""))
        except (ValueError, TypeError):
            raise ValueError("Выберите режим прогноза.") from None
        if case not in (1, 2):
            raise ValueError("Выберите режим прогноза.")
        try:
            chosen = date.fromisoformat(payload.get("date", ""))
        except (ValueError, TypeError):
            raise ValueError("Укажите дату прогноза.") from None
        at = pd.Timestamp(chosen)
        section = self.grid.loc[self.grid["id"].astype(str).eq(pid)].set_index("segment_id").sort_index()
        if not section["commissioned_at"].lt(at).all():
            raise ValueError("Дата прогноза должна быть позже даты ввода участка.")

        target = self.targets.get(pid)
        if target and case == target["case"] and chosen == date.fromisoformat(target["target_date"]):
            model = FINAL[case]
            prediction = self.result.loc[self.result["id"].eq(pid), ["segment_id", "thickness_pred"]]
        elif case == 1:
            history = rf.History.before(self.vtd, self.labels, at)
            made = rf.forecast_case1(history, self.grid, pid, at)
            if made is None:
                raise ValueError("Для трубы нет ВТД до выбранной даты. Проверьте дату или выберите «От ввода».")
            model, prediction = GENERIC[1], made[0][["segment_id", "thickness_pred"]]
        else:
            history = rf.History.before(self.vtd, self.labels, at)
            level = rf.terrain_fallback(history, self.grid, pid, at)
            model = GENERIC[2]
            prediction = case2_elevation.forecast(self.grid, level, pid, at)[["segment_id", "thickness_pred"]]

        if case == 1:
            labels = self.labels.loc[self.labels["id"].astype(str).eq(pid) & self.labels["inspection_date"].lt(at)]
            last = labels.loc[labels["inspection_date"].eq(labels["inspection_date"].max())]
            reference = last.set_index("segment_id")["min_residual_thickness_mm"].dropna()
        else:
            reference = section["nominal_thickness_mm"]
        rows, segments = [], []
        for sid, value in zip(prediction["segment_id"], prediction["thickness_pred"]):
            sid = int(sid)
            row = {"id": pid, "segment_id": sid, "thickness_pred": round(float(value), 4)}
            rows.append(row)
            segments.append(dict(row, baseline=float(reference[sid]) if sid in reference.index else None,
                                 nominal=float(section.at[sid, "nominal_thickness_mm"]),
                                 end_m=float(section.at[sid, "end_m"])))
        rows.sort(key=lambda r: r["segment_id"])
        segments.sort(key=lambda r: r["segment_id"])
        count = math.ceil(len(rows) * 0.1)
        priority = sorted(rows, key=lambda r: (r["thickness_pred"], r["segment_id"]))[:count]
        return {"id": pid, "case": str(case), "date": chosen.isoformat(),
                "length": float(section["end_m"].max()), "model": model,
                "segments": segments, "rows": rows, "priority": priority}


class Handler(BaseHTTPRequestHandler):
    def respond(self, code, body, content_type="application/json; charset=utf-8"):
        encoded = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        if self.path == "/api/pipes":
            self.respond(200, self.server.forecasts.catalog())
        elif self.path == "/":
            self.respond(200, (WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
        else:
            self.respond(404, {"error": "Не найдено"})

    def do_POST(self):
        if self.path != "/api/predict":
            return self.respond(404, {"error": "Не найдено"})
        try:
            size = int(self.headers.get("Content-Length", 0))
            if not 0 < size <= 4096:
                return self.respond(413, {"error": "Недопустимый размер запроса."})
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict):
                raise ValueError("Некорректный запрос.")
            self.respond(200, self.server.forecasts.calculate(payload))
        except (ValueError, TypeError, OverflowError) as error:
            self.respond(400, {"error": str(error)})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="folder with the six organiser CSV files")
    parser.add_argument("--work-dir", type=Path, default=Path("outputs"), help="intermediate tables and result.csv")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    print("PipeTwin: считаю прогноз из исходных CSV…", flush=True)
    try:
        forecasts = Forecasts(args.data_dir, args.work_dir)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Не удалось загрузить данные из {args.data_dir}: {error}\n")
    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        server.forecasts = forecasts
        url = f"http://127.0.0.1:{server.server_port}/"
        print(f"PipeTwin {url}", flush=True)
        if args.open:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
