"""
Fase 2 - Volatilidad implícita de Polymarket y comparación con el DVOL de Deribit.

Para cada día y cada horizonte (24h, 12h, 6h, 3h, 1h antes del cierre):
  1. Toma el último precio del "Sí" de cada tramo.
  2. Normaliza los precios para que sumen 1 (quita el margen de la suma de tramos).
  3. Ajusta una lognormal centrada en el spot de Binance y despeja la volatilidad
     anualizada que mejor reproduce esos precios -> pm_vol.
  4. Lee el DVOL de Deribit en ese mismo momento -> dvol.
  5. Calcula, con ambos modelos, la probabilidad asignada al tramo ganador
     y el Brier score de cada uno.

Entrada: data/*.csv (fase 1). Salida: data/fase2_snapshots.csv y un resumen en pantalla.
Uso: python src/fase2_vol_implicita.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
HORIZONS_H = [24, 12, 6, 3, 1]
YEAR_SECONDS = 365 * 24 * 3600
MAX_PRICE_AGE_S = 180  # una foto solo vale si todos sus precios tienen < 3 min


def load_prices():
    """Usa los precios minuto a minuto si existen; si no, los de cada 10 min (avisa)."""
    fresh = DATA / "pm_prices_fresh.csv"
    if fresh.exists():
        return pd.read_csv(fresh), "pm_prices_fresh.csv (1 min)"
    print("AVISO: no existe pm_prices_fresh.csv. Ejecuta antes:\n"
          "  python src/fase1_descarga.py --solo-precios\n"
          "Se usan los precios cada 10 min: sesgan los resultados en contra de Polymarket.\n")
    return pd.read_csv(DATA / "pm_prices.csv"), "pm_prices.csv (10 min)"


def snapshot_prices(prices_day, buckets_sorted, t):
    """Último precio de cada tramo en o antes de t y su antigüedad máxima (s)."""
    sub = prices_day[prices_day.t <= t].sort_values("t").groupby("bucket").agg(
        p=("p", "last"), t=("t", "last")).reindex(buckets_sorted)
    if sub.p.isna().any():
        return None, None
    return sub.p.to_numpy(float), float(t - sub.t.min())


def bucket_probs(spot, sigma, t_years, lows, highs):
    """Probabilidad de cada tramo bajo una lognormal sin deriva."""
    s = sigma * np.sqrt(t_years)

    def cdf(x):
        with np.errstate(divide="ignore"):
            z = (np.log(np.where(x > 0, x, np.nan) / spot) + 0.5 * s ** 2) / s
        out = norm.cdf(z)
        out = np.where(x <= 0, 0.0, out)
        out = np.where(np.isinf(x), 1.0, out)
        return out

    return cdf(highs) - cdf(lows)


def fit_sigma(spot, t_years, lows, highs, target):
    def loss(sig):
        return np.sum((bucket_probs(spot, sig, t_years, lows, highs) - target) ** 2)

    res = minimize_scalar(loss, bounds=(0.03, 3.0), method="bounded")
    return res.x, res.fun



def main():
    buckets = pd.read_csv(DATA / "buckets.csv")
    prices, source = load_prices()
    print(f"Precios de Polymarket: {source}")
    dvol = pd.read_csv(DATA / "dvol_hourly.csv").sort_values("t")
    btc = pd.read_csv(DATA / "btc_hourly.csv").sort_values("t")
    settle = pd.read_csv(DATA / "settlement.csv")

    # Spot disponible en el instante t = cierre de la vela 1h que termina en t
    btc["t_end"] = btc.t + 3600
    rows = []
    for d, g in buckets.groupby("date"):
        g = g.sort_values("low")
        close_ts = int(g.close_ts.iloc[0])
        lows, highs = g.low.to_numpy(float), g.high.to_numpy(float)
        winner = g.bucket[g.resolved_yes == True]  # noqa: E712
        if winner.empty:
            continue
        winner = winner.iloc[0]
        settle_px = settle.loc[settle.date == d, "settle_close"].iloc[0]
        pday = prices[prices.date == d]

        for h in HORIZONS_H:
            t = close_ts - h * 3600
            raw, age = snapshot_prices(pday, g.bucket, t)
            if raw is None or age > MAX_PRICE_AGE_S:
                continue
            total = raw.sum()
            if total <= 0:
                continue
            target = raw / total

            spot_row = btc[btc.t_end <= t].tail(1)
            dv_row = dvol[dvol.t <= t].tail(1)
            if spot_row.empty or dv_row.empty:
                continue
            spot = float(spot_row.close.iloc[0])
            dv = float(dv_row.dvol_close.iloc[0]) / 100
            t_years = h * 3600 / YEAR_SECONDS

            pm_vol, sse = fit_sigma(spot, t_years, lows, highs, target)
            p_dvol = bucket_probs(spot, dv, t_years, lows, highs)
            p_dvol = p_dvol / p_dvol.sum()
            outcome = (g.bucket == winner).to_numpy(float)
            iw = int(np.argmax(outcome))

            rows.append({
                "date": d, "horizon_h": h, "spot": spot, "settle": settle_px,
                "realized_logret": np.log(settle_px / spot),
                "price_age_s": age, "sum_yes": total, "pm_vol": pm_vol, "dvol": dv,
                "vol_gap": pm_vol - dv, "fit_sse": sse,
                "p_win_pm": target[iw], "p_win_dvol": p_dvol[iw],
                "brier_pm": np.sum((target - outcome) ** 2),
                "brier_dvol": np.sum((p_dvol - outcome) ** 2),
                "logloss_pm": -np.log(max(target[iw], 1e-4)),
                "logloss_dvol": -np.log(max(p_dvol[iw], 1e-4)),
            })

    snap = pd.DataFrame(rows)
    snap.to_csv(DATA / "fase2_snapshots.csv", index=False)

    # Volatilidad realizada coherente con cada horizonte (anualizada)
    snap["realized_vol"] = snap.realized_logret.abs() * np.sqrt(
        YEAR_SECONDS / (snap.horizon_h * 3600)) * np.sqrt(np.pi / 2)

    pd.set_option("display.width", 140)
    summary = snap.groupby("horizon_h").agg(
        dias=("date", "nunique"),
        antig_precio_s=("price_age_s", "median"),
        suma_si=("sum_yes", "median"),
        pm_vol=("pm_vol", "median"),
        dvol=("dvol", "median"),
        vol_realizada=("realized_vol", "mean"),
        brier_pm=("brier_pm", "mean"),
        brier_dvol=("brier_dvol", "mean"),
        logloss_pm=("logloss_pm", "mean"),
        logloss_dvol=("logloss_dvol", "mean"),
    ).sort_index(ascending=False)
    print(summary.round(3).to_string())
    print(f"\nGuardado: {DATA / 'fase2_snapshots.csv'} ({len(snap)} fotos)")


if __name__ == "__main__":
    main()
