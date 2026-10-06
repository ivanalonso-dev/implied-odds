"""
Fase 3 - ¿La ventaja de Polymarket es real o es un artefacto del DVOL?

El DVOL es una volatilidad plana a 30 días. Los mercados de Polymarket vencen a las
12:00 ET, en plena sesión de EE. UU., cuando BTC suele moverse más. Para que la
comparación sea justa se construyen tres referencias más exigentes, sin mirar al futuro
(solo con datos anteriores a cada foto):

  dvol        DVOL tal cual (referencia de la fase 2)
  dvol_seas   DVOL ajustado por estacionalidad intradía: se reparte la varianza diaria
              según el peso histórico de cada hora del día en las horas que faltan
  rv_seas     Volatilidad realizada de los últimos 7 días, con el mismo ajuste horario

Para cada referencia se calcula el Brier y el log-loss sobre los mismos tramos y se
compara con Polymarket (diferencia media por día y estadístico t pareado).

Entrada: data/*.csv y data/fase2_snapshots.csv. Salida: data/fase3_scores.csv
Uso: python src/fase3_benchmarks.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from fase2_vol_implicita import MAX_PRICE_AGE_S, YEAR_SECONDS, bucket_probs, load_prices, snapshot_prices

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MIN_HISTORY_DAYS = 7
RV_WINDOW_DAYS = 7


def seasonal_weights(rets: pd.DataFrame, t_cut: int):
    """Peso de varianza de cada hora UTC usando solo datos anteriores a t_cut."""
    past = rets[rets.t_end <= t_cut]
    if past.t_end.nunique() < MIN_HISTORY_DAYS * 24:
        return None
    by_hour = past.groupby("hour").r2.mean()
    return by_hour / by_hour.mean()


def remaining_multiplier(weights, t_from: int, t_to: int):
    """Media de los pesos horarios de las horas entre t_from y t_to."""
    hours = [((t_from + 3600 * k) // 3600) % 24 for k in range((t_to - t_from) // 3600)]
    return float(np.mean([weights.get(h, 1.0) for h in hours]))


def score(p, outcome):
    p = p / p.sum()
    iw = int(np.argmax(outcome))
    return np.sum((p - outcome) ** 2), -np.log(max(p[iw], 1e-4))


def main():
    buckets = pd.read_csv(DATA / "buckets.csv")
    prices, source = load_prices()
    print(f"Precios de Polymarket: {source}")
    snaps = pd.read_csv(DATA / "fase2_snapshots.csv")
    btc = pd.read_csv(DATA / "btc_hourly.csv").sort_values("t").reset_index(drop=True)

    btc["t_end"] = btc.t + 3600
    btc["r"] = np.log(btc.close / btc.close.shift(1))
    btc["r2"] = btc.r ** 2
    btc["hour"] = (btc.t // 3600) % 24  # hora UTC de inicio de la vela
    rets = btc.dropna(subset=["r"])

    rows = []
    for _, s in snaps.iterrows():
        g = buckets[buckets.date == s.date].sort_values("low")
        close_ts = int(g.close_ts.iloc[0])
        t = close_ts - int(s.horizon_h) * 3600
        lows, highs = g.low.to_numpy(float), g.high.to_numpy(float)
        outcome = g.resolved_yes.astype(bool).to_numpy(float)
        t_years = s.horizon_h * 3600 / YEAR_SECONDS

        w = seasonal_weights(rets, t)
        if w is None:
            continue
        mult = remaining_multiplier(w, t, close_ts)

        window = rets[(rets.t_end <= t) & (rets.t_end > t - RV_WINDOW_DAYS * 86400)]
        rv7 = np.sqrt(window.r2.mean() * 24 * 365)

        # Precios de Polymarket en la misma foto (mismo criterio que la fase 2)
        last, age = snapshot_prices(prices[prices.date == s.date], g.bucket, t)
        if last is None or age > MAX_PRICE_AGE_S:
            continue
        p_pm = last / last.sum()

        vols = {"dvol": s.dvol, "dvol_seas": s.dvol * np.sqrt(mult), "rv_seas": rv7 * np.sqrt(mult)}
        row = {"date": s.date, "horizon_h": int(s.horizon_h), "season_mult": mult,
               "pm_vol": s.pm_vol, **{f"vol_{k}": v for k, v in vols.items()}}
        row["brier_pm"], row["logloss_pm"] = score(p_pm, outcome)
        for k, v in vols.items():
            row[f"brier_{k}"], row[f"logloss_{k}"] = score(
                bucket_probs(s.spot, v, t_years, lows, highs), outcome)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(DATA / "fase3_scores.csv", index=False)

    print(f"Fotos evaluadas: {len(df)} ({df.date.nunique()} días con al menos "
          f"{MIN_HISTORY_DAYS} días de historia previa)\n")
    print("Volatilidad mediana usada por cada modelo")
    print(df.groupby("horizon_h")[["pm_vol", "vol_dvol", "vol_dvol_seas", "vol_rv_seas",
                                   "season_mult"]].median()
          .sort_index(ascending=False).round(3).to_string())

    print("\nBrier medio (más bajo = mejor) y t pareado frente a Polymarket")
    print("(t > 2: Polymarket mejor de forma significativa; t < -2: la referencia es mejor)")
    out = []
    for h, g in df.groupby("horizon_h"):
        r = {"horizon_h": h, "pm": g.brier_pm.mean()}
        for k in ["dvol", "dvol_seas", "rv_seas"]:
            d = g[f"brier_{k}"] - g.brier_pm
            r[k] = g[f"brier_{k}"].mean()
            r[f"t_{k}"] = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
        out.append(r)
    print(pd.DataFrame(out).set_index("horizon_h").sort_index(ascending=False).round(3).to_string())


if __name__ == "__main__":
    main()
