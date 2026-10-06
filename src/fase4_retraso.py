"""
Fase 4 - ¿Va Polymarket por detrás del precio de BTC en la última hora?

Hipótesis de la fase 3: en la última hora Polymarket pierde contra referencias simples
no por estimar mal la volatilidad, sino porque sus precios tardan en reaccionar
cuando BTC se mueve.

Método:
  1. Descarga, para cada día, los últimos 75 minutos antes del cierre:
       - velas de 1 minuto de Binance BTCUSDT
       - precio minuto a minuto del "Sí" de cada tramo en Polymarket
  2. En cada minuto calcula la probabilidad "justa" de cada tramo a partir del spot
     de Binance, el tiempo que falta y la volatilidad de referencia del día
     (volatilidad realizada 7 días ajustada por hora, de la fase 3).
  3. Compara Polymarket en el minuto t con la probabilidad justa en el minuto t-k,
     para k = 0..10 minutos. Si el error mínimo aparece en k > 0, Polymarket
     va k minutos por detrás del precio.

Entrada: data/buckets.csv, data/fase3_scores.csv
Salida:  data/btc_1m.csv, data/pm_prices_1m.csv, data/fase4_lag.csv
Uso:
  python src/fase4_retraso.py              # descarga + análisis
  python src/fase4_retraso.py --solo-analisis   # si ya descargaste
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from fase2_vol_implicita import YEAR_SECONDS, bucket_probs

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CLOB = "https://clob.polymarket.com"
BINANCE_BASES = ["https://api.binance.com", "https://data-api.binance.vision"]
WINDOW_MIN = 75          # minutos descargados antes del cierre
ANALYSIS_MIN = 60        # minutos analizados (los 15 primeros dan margen al lag máximo)
MAX_LAG = 10
GRID_OFFSET_S = 30

session = requests.Session()
session.headers["User-Agent"] = "pm-vs-deribit-research/1.0"


def get_json(url, params, retries=4, pause=0.2):
    for i in range(retries):
        try:
            r = session.get(url, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(2 * (i + 1))
                continue
            r.raise_for_status()
            time.sleep(pause)
            return r.json()
        except requests.RequestException as e:
            if i == retries - 1:
                print(f"  ! fallo {url}: {e}")
                return None
            time.sleep(2 * (i + 1))
    return None


def download(buckets: pd.DataFrame):
    btc_rows, pm_rows = [], []
    days = buckets.groupby("date")
    for n, (d, g) in enumerate(days, 1):
        close_ts = int(g.close_ts.iloc[0])
        start = close_ts - WINDOW_MIN * 60

        kl = None
        for base in BINANCE_BASES:
            kl = get_json(f"{base}/api/v3/klines", {
                "symbol": "BTCUSDT", "interval": "1m",
                "startTime": start * 1000, "endTime": close_ts * 1000, "limit": WINDOW_MIN + 1})
            if kl:
                break
        for k in kl or []:
            # precio disponible al final del minuto
            btc_rows.append({"date": d, "t": int(k[0] // 1000) + 60, "close": float(k[4])})

        for _, b in g.iterrows():
            data = get_json(f"{CLOB}/prices-history", {
                "market": b.yes_token, "startTs": start, "endTs": close_ts, "fidelity": 1})
            for h in (data or {}).get("history", []):
                pm_rows.append({"date": d, "bucket": b.bucket, "t": int(h["t"]), "p": float(h["p"])})
        print(f"  {n}/{days.ngroups} {d}")

    btc = pd.DataFrame(btc_rows)
    pm = pd.DataFrame(pm_rows)
    btc.to_csv(DATA / "btc_1m.csv", index=False)
    pm.to_csv(DATA / "pm_prices_1m.csv", index=False)
    return btc, pm


def minute_grid(series_t, series_v, grid):
    """Último valor conocido en cada minuto de la rejilla (sin mirar al futuro)."""
    s = pd.Series(series_v.values, index=series_t.values).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s.reindex(s.index.union(grid)).ffill().reindex(grid).to_numpy(float)


def analyse(buckets, btc, pm, scores):
    sigma_day = scores[scores.horizon_h == 1].set_index("date").vol_rv_seas
    lag_err = {k: [] for k in range(MAX_LAG + 1)}
    best_lags = []

    for d, g in buckets.groupby("date"):
        if d not in sigma_day.index:
            continue
        g = g.sort_values("low")
        close_ts = int(g.close_ts.iloc[0])
        # Rejilla en el segundo 30 de cada minuto. Polymarket guarda su precio hacia el
        # segundo 13 y Binance cierra la vela en el segundo 0: así el precio de BTC usado
        # es siempre MÁS antiguo que el de Polymarket, y cualquier sesgo juega a favor de
        # Polymarket (subestima el retraso, nunca lo inventa).
        grid = np.arange(close_ts - WINDOW_MIN * 60 + 120 + GRID_OFFSET_S, close_ts, 60)
        bd, pd_ = btc[btc.date == d], pm[pm.date == d]
        if len(bd) < WINDOW_MIN * 0.8 or pd_.bucket.nunique() < len(g):
            continue

        spot = minute_grid(bd.t, bd.close, grid)
        pm_mat = np.column_stack([
            minute_grid(pd_[pd_.bucket == b].t, pd_[pd_.bucket == b].p, grid) for b in g.bucket])
        if np.isnan(pm_mat).any() or np.isnan(spot).any():
            continue
        pm_mat = pm_mat / pm_mat.sum(axis=1, keepdims=True)

        lows, highs = g.low.to_numpy(float), g.high.to_numpy(float)
        sig = float(sigma_day[d])
        fair = np.vstack([
            bucket_probs(spot[i], sig, max(close_ts - grid[i], 30) / YEAR_SECONDS, lows, highs)
            for i in range(len(grid))])
        fair = fair / fair.sum(axis=1, keepdims=True)

        first = len(grid) - ANALYSIS_MIN  # índice desde el que se analiza (>= MAX_LAG)
        day_err = []
        for k in range(MAX_LAG + 1):
            # Polymarket en t frente a precio justo en t-k
            e = np.mean(np.sum((pm_mat[first:] - fair[first - k:len(grid) - k]) ** 2, axis=1))
            lag_err[k].append(e)
            day_err.append(e)
        best_lags.append({"date": d, "best_lag_min": int(np.argmin(day_err)),
                          "err_lag0": day_err[0], "err_best": min(day_err)})

    table = pd.DataFrame({
        "lag_min": list(lag_err),
        "error_medio": [np.mean(v) for v in lag_err.values()],
        "dias": [len(v) for v in lag_err.values()],
    })
    per_day = pd.DataFrame(best_lags)
    per_day.to_csv(DATA / "fase4_lag.csv", index=False)
    return table, per_day


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo-analisis", action="store_true")
    args = ap.parse_args()

    buckets = pd.read_csv(DATA / "buckets.csv")
    scores = pd.read_csv(DATA / "fase3_scores.csv")

    if args.solo_analisis:
        btc = pd.read_csv(DATA / "btc_1m.csv")
        pm = pd.read_csv(DATA / "pm_prices_1m.csv")
    else:
        print("Descargando la última hora de cada día (minuto a minuto)...")
        btc, pm = download(buckets)

    table, per_day = analyse(buckets, btc, pm, scores)
    if per_day.empty:
        print("Sin días válidos: revisa que btc_1m.csv y pm_prices_1m.csv tengan datos.")
        return

    best = int(table.loc[table.error_medio.idxmin(), "lag_min"])
    print("\nError medio entre Polymarket(t) y precio justo(t - k)")
    print(table.round(5).to_string(index=False))
    print(f"\nRetraso que mejor explica a Polymarket: {best} min "
          f"({len(per_day)} días analizados)")
    print(f"Días con retraso > 0: {(per_day.best_lag_min > 0).mean() * 100:.0f}% · "
          f"retraso mediano por día: {per_day.best_lag_min.median():.0f} min")
    print(f"Guardado: {DATA / 'fase4_lag.csv'}")


if __name__ == "__main__":
    main()
