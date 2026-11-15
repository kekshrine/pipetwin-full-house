"""Local web interface for the final forecast.

    python -m pipetwin.web --data-dir /path/to/raw-csv --open

On start the pipeline builds ``result.csv`` from the raw CSV (the same run as ``python -m pipetwin``),
then the page shows the forecast along each target pipe and the 10 % thinnest segments.
"""

import argparse
from datetime import date
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import webbrowser

import pandas as pd

from pipetwin import forecast as rf
from pipetwin import solution

WEB = Path(__file__).parent / "web"
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "raw_data"
MODELS = {1: "итоговая смесь трёх прогнозов", 2: "сглаженная высота трассы"}
CASE_NAMES = {1: "«После ВТД»", 2: "«От ввода»"}


class Forecasts:
    def __init__(self, data_dir, work_dir):
        work_dir = Path(work_dir)
        result = solution.run(Path(data_dir), work_dir, work_dir / "result.csv")
        _, self.grid, _, self.labels = rf.load_tables(work_dir / "cleaning", work_dir / "labels")
        self.result = result.assign(id=result["id"].astype(str))
        self.targets = {t["id"]: t for t in solution.TARGETS}

    def catalog(self):
        return [{"id": pid, "has_history": t["case"] == 1} for pid, t in self.targets.items()]

    def calculate(self, payload):
        pid = str(payload.get("id", "")).strip()
        if pid not in self.targets:
            raise ValueError("Выберите одну из двух труб кейса.")
        target = self.targets[pid]
        try:
            case = int(payload.get("case", ""))
        except (ValueError, TypeError):
            raise ValueError("Выберите режим прогноза.") from None
        if case != target["case"]:
            raise ValueError(f"Для трубы {pid} итоговая модель считает прогноз в режиме {CASE_NAMES[target['case']]}.")
        try:
            chosen = date.fromisoformat(payload.get("date", ""))
        except (ValueError, TypeError):
            raise ValueError("Укажите дату прогноза.") from None
        target_date = date.fromisoformat(target["target_date"])
        if chosen != target_date:
            raise ValueError(f"Итоговая модель считает прогноз для трубы {pid} на {target_date:%d.%m.%Y}.")

        section = self.grid.loc[self.grid["id"].astype(str).eq(pid)].set_index("segment_id").sort_index()
        rows = (self.result.loc[self.result["id"].eq(pid), solution.COLUMNS]
                .sort_values("segment_id").to_dict("records"))
        if case == 1:
            labels = self.labels.loc[self.labels["id"].astype(str).eq(pid)
                                     & self.labels["inspection_date"].lt(pd.Timestamp(target_date))]
            last = labels.loc[labels["inspection_date"].eq(labels["inspection_date"].max())]
            reference = last.set_index("segment_id")["min_residual_thickness_mm"].dropna()
        else:
            reference = section["nominal_thickness_mm"]
        segments = []
        for row in rows:
            sid = int(row["segment_id"])
            row.update(id=pid, segment_id=sid, thickness_pred=float(row["thickness_pred"]))
            segments.append(dict(row, baseline=float(reference[sid]) if sid in reference.index else None,
                                 nominal=float(section.at[sid, "nominal_thickness_mm"]),
                                 end_m=float(section.at[sid, "end_m"])))
        count = math.ceil(len(rows) * 0.1)
        priority = sorted(rows, key=lambda r: (r["thickness_pred"], r["segment_id"]))[:count]
        return {"id": pid, "case": str(case), "date": target_date.isoformat(),
                "length": float(section["end_m"].max()), "model": MODELS[case],
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
