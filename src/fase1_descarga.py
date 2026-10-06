"""
Fase 1 - Descarga de datos
Estudio: volatilidad implícita de los mercados de rango diarios de BTC en Polymarket
frente al DVOL de Deribit.

Descarga a ./data/:
  buckets.csv      -> un registro por tramo y día (límites, tokens, volumen, ganador oficial)
  pm_prices.csv    -> historial de precio del "Sí" de cada tramo (últimas 26 h, cada 10 min)
  pm_prices_fresh.csv -> precio minuto a minuto en los 20 min previos a cada foto (24h..1h)
  dvol_hourly.csv  -> DVOL de Deribit (BTC), horario
  btc_hourly.csv   -> velas 1h de Binance BTCUSDT (spot para las fotos)
  settlement.csv   -> cierre de la vela 1m de Binance a las 12:00 ET (regla de resolución)
  check.csv        -> control: ¿el tramo que marca Binance coincide con el ganador oficial?

Uso:
  pip install requests pandas
  python src/fase1_descarga.py --start 2026-07-01 --end 2026-10-05
  python src/fase1_descarga.py --solo-precios   # solo rehace los precios frescos
"""

import argparse
import json
import math
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
DERIBIT = "https://www.deribit.com/api/v2/public"
BINANCE_BASES = ["https://api.binance.com", "https://data-api.binance.vision"]
ET = ZoneInfo("America/New_York")
OUT = Path(__file__).resolve().parent.parent / "data"  # siempre en la carpeta del proyecto
MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]

session = requests.Session()
session.headers["User-Agent"] = "pm-vs-deribit-research/1.0"


def get_json(url, params=None, retries=4, pause=0.25):
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
                print(f"  ! fallo {url} {params}: {e}")
                return None
            time.sleep(1.5 * (i + 1))
    return None


# ---------------------------------------------------------------- Polymarket
def close_time_et(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, 12, 0, tzinfo=ET)


def find_event(d: date):
    """Busca el evento '¿Precio de Bitcoin el <fecha>?' probando patrones de slug."""
    base = f"bitcoin-price-on-{MONTHS[d.month - 1]}-{d.day}"
    for slug in (f"{base}-{d.year}", base, f"{base}-{d.year}-2"):
        data = get_json(f"{GAMMA}/events", {"slug": slug})
        if not data:
            continue
        for ev in data:
            end = ev.get("endDate", "")
            if end[:10] == d.isoformat():
                return ev
    return None


def parse_bucket(label: str):
    s = label.replace(",", "").replace("$", "").strip()
    if s.startswith("<"):
        return 0.0, float(s[1:])
    if s.startswith(">"):
        return float(s[1:]), math.inf
    lo, hi = s.split("-")
    return float(lo), float(hi)


def collect_buckets(dates):
    rows = []
    for d in dates:
        ev = find_event(d)
        if ev is None:
            print(f"  - {d}: evento no encontrado")
            continue
        n = 0
        for m in ev.get("markets", []):
            label = m.get("groupItemTitle")
            if not label:
                continue
            try:
                lo, hi = parse_bucket(label)
                tokens = json.loads(m["clobTokenIds"])
                prices = json.loads(m.get("outcomePrices", "[]"))
            except Exception as e:
                print(f"  ! {d} tramo {label}: {e}")
                continue
            rows.append({
                "date": d.isoformat(),
                "event_slug": ev["slug"],
                "close_ts": int(close_time_et(d).timestamp()),
                "bucket": label,
                "low": lo,
                "high": hi,
                "yes_token": tokens[0],
                "no_token": tokens[1],
                "volume": float(m.get("volumeNum") or m.get("volume") or 0),
                "closed": m.get("closed"),
                "resolved_yes": (float(prices[0]) > 0.5) if prices else None,
            })
            n += 1
        print(f"  + {d}: {ev['slug']} ({n} tramos)")
    return pd.DataFrame(rows)


def collect_prices(buckets: pd.DataFrame, hours_before=26, fidelity_min=10):
    rows = []
    for i, b in buckets.iterrows():
        end = int(b.close_ts)
        start = end - hours_before * 3600
        data = get_json(f"{CLOB}/prices-history", {
            "market": b.yes_token, "startTs": start, "endTs": end,
            "fidelity": fidelity_min,
        })
        hist = (data or {}).get("history", [])
        for h in hist:
            rows.append({"date": b.date, "bucket": b.bucket, "t": int(h["t"]), "p": float(h["p"])})
        if i % 25 == 0:
            print(f"  precios {i + 1}/{len(buckets)}")
    return pd.DataFrame(rows)


SNAPSHOT_HORIZONS_H = [24, 12, 6, 3, 1]


def collect_fresh_prices(buckets: pd.DataFrame, window_min=20):
    """Precio minuto a minuto en los 20 min previos a cada foto (24h, 12h, 6h, 3h, 1h).

    Corrige el problema de la primera versión: con fidelity=10 el precio de cada foto
    tenía de media ~10 minutos de antigüedad.
    """
    rows = []
    for i, b in enumerate(buckets.itertuples(index=False)):
        for h in SNAPSHOT_HORIZONS_H:
            t = int(b.close_ts) - h * 3600
            data = get_json(f"{CLOB}/prices-history", {
                "market": b.yes_token, "startTs": t - window_min * 60, "endTs": t,
                "fidelity": 1,
            })
            for x in (data or {}).get("history", []):
                rows.append({"date": b.date, "bucket": b.bucket, "t": int(x["t"]), "p": float(x["p"])})
        if i % 25 == 0:
            print(f"  precios frescos {i + 1}/{len(buckets)}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Deribit
def collect_dvol(start: datetime, end: datetime):
    rows = []
    chunk = timedelta(days=30)
    t = start
    while t < end:
        t2 = min(t + chunk, end)
        data = get_json(f"{DERIBIT}/get_volatility_index_data", {
            "currency": "BTC",
            "start_timestamp": int(t.timestamp() * 1000),
            "end_timestamp": int(t2.timestamp() * 1000),
            "resolution": "3600",
        })
        for r in ((data or {}).get("result") or {}).get("data", []):
            rows.append({"t": int(r[0] // 1000), "dvol_open": r[1], "dvol_high": r[2],
                         "dvol_low": r[3], "dvol_close": r[4]})
        t = t2
    return pd.DataFrame(rows).drop_duplicates("t").sort_values("t")


# ---------------------------------------------------------------- Binance
def binance_klines(params):
    for base in BINANCE_BASES:
        data = get_json(f"{base}/api/v3/klines", params)
        if data is not None:
            return data
    return []


def collect_btc_hourly(start: datetime, end: datetime):
    rows = []
    t = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    while t < end_ms:
        data = binance_klines({"symbol": "BTCUSDT", "interval": "1h",
                               "startTime": t, "endTime": end_ms, "limit": 1000})
        if not data:
            break
        for k in data:
            rows.append({"t": int(k[0] // 1000), "open": float(k[1]), "high": float(k[2]),
                         "low": float(k[3]), "close": float(k[4])})
        t = int(data[-1][0]) + 3600_000
    return pd.DataFrame(rows).drop_duplicates("t").sort_values("t")


def collect_settlement(dates):
    """Cierre de la vela 1m que abre a las 12:00 ET (supuesto; se valida en check.csv)."""
    rows = []
    for d in dates:
        t = int(close_time_et(d).timestamp() * 1000)
        data = binance_klines({"symbol": "BTCUSDT", "interval": "1m",
                               "startTime": t, "limit": 1})
        if data:
            rows.append({"date": d.isoformat(), "settle_close": float(data[0][4])})
    return pd.DataFrame(rows)


def build_check(buckets, settle):
    rows = []
    for d, g in buckets.groupby("date"):
        s = settle.loc[settle.date == d, "settle_close"]
        if s.empty:
            continue
        px = s.iloc[0]
        # Regla: si cae justo en un límite, gana el tramo superior -> low <= px < high
        mine = g[(g.low <= px) & (px < g.high)]
        official = g[g.resolved_yes == True]  # noqa: E712
        rows.append({
            "date": d, "settle_close": px,
            "bucket_binance": mine.bucket.iloc[0] if len(mine) else None,
            "bucket_oficial": official.bucket.iloc[0] if len(official) else None,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["coincide"] = df.bucket_binance == df.bucket_oficial
    return df


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=(date.today() - timedelta(days=90)).isoformat())
    ap.add_argument("--end", default=(date.today() - timedelta(days=1)).isoformat())
    ap.add_argument("--solo-precios", action="store_true",
                    help="solo descarga los precios frescos (minuto a minuto) de cada foto")
    args = ap.parse_args()

    if args.solo_precios:
        OUT.mkdir(exist_ok=True)
        buckets = pd.read_csv(OUT / "buckets.csv")
        print("Precios frescos (1 min) alrededor de cada foto")
        fresh = collect_fresh_prices(buckets)
        fresh.to_csv(OUT / "pm_prices_fresh.csv", index=False)
        print(f"Guardado: {OUT / 'pm_prices_fresh.csv'} ({len(fresh)} filas)")
        return

    d0, d1 = date.fromisoformat(args.start), date.fromisoformat(args.end)
    dates = [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]
    OUT.mkdir(exist_ok=True)
    t_start = datetime.combine(d0 - timedelta(days=2), datetime.min.time(), tzinfo=timezone.utc)
    t_end = datetime.combine(d1 + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)

    print("1/5 Eventos y tramos de Polymarket")
    buckets = collect_buckets(dates)
    buckets.to_csv(OUT / "buckets.csv", index=False)
    if buckets.empty:
        print("Sin eventos: revisa el patrón de slug en find_event().")
        return

    print("2/5 Historial de precios de cada tramo")
    collect_prices(buckets).to_csv(OUT / "pm_prices.csv", index=False)

    print("2b/5 Precios frescos (1 min) alrededor de cada foto")
    collect_fresh_prices(buckets).to_csv(OUT / "pm_prices_fresh.csv", index=False)

    print("3/5 DVOL de Deribit")
    collect_dvol(t_start, t_end).to_csv(OUT / "dvol_hourly.csv", index=False)

    print("4/5 Velas 1h de Binance")
    collect_btc_hourly(t_start, t_end).to_csv(OUT / "btc_hourly.csv", index=False)

    print("5/5 Precio de liquidación y control")
    settle = collect_settlement(sorted(buckets.date.map(date.fromisoformat).unique()))
    settle.to_csv(OUT / "settlement.csv", index=False)
    check = build_check(buckets, settle)
    check.to_csv(OUT / "check.csv", index=False)

    n_days = buckets.date.nunique()
    match = check.coincide.mean() * 100 if not check.empty else float("nan")
    print(f"\nListo: {n_days} días, {len(buckets)} tramos. "
          f"Coincidencia Binance vs ganador oficial: {match:.1f}%")


if __name__ == "__main__":
    main()
