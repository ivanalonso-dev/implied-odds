"""
Fase 5 - Gráficos para la publicación.

Genera en docs/:
  fig1_volatilidad.png  volatilidad esperada por Polymarket, DVOL de Deribit y
                        volatilidad que realmente hubo, según las horas al cierre
  fig2_brier.png        precisión (Brier score, más bajo = mejor) de Polymarket
                        frente al DVOL y al DVOL ajustado por hora del día

Se generan en inglés (fig1_volatilidad.png, fig2_brier.png: README, LinkedIn, informe EN)
y en español (fig1_volatilidad_es.png, fig2_brier_es.png: informe ES).
Entrada: data/fase2_snapshots.csv, data/fase3_scores.csv
Uso: python src/fase5_graficos.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA, DOCS = ROOT / "data", ROOT / "docs"
YEAR_SECONDS = 365 * 24 * 3600

# Paleta categórica validada (3 primeros huecos: pasan todas las parejas en daltonismo)
C_PM, C_DVOL, C_THIRD = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

HORIZONS = [24, 12, 6, 3, 1]

TXT = {
    "en": {
        "xlabel": "Hours before market close (12:00 ET)",
        "footer": "Data: Polymarket daily BTC price-range markets, Deribit DVOL, Binance BTCUSDT · "
                  "{n} days, Jul–Oct 2026 · Ivan Alonso H.",
        "realized": "Realized", "dvol_adj": "DVOL, time-of-day adj.",
        "vol_y": "Annualized volatility (%)",
        "vol_t": "Polymarket reprices volatility as the close approaches; DVOL stays flat",
        "vol_s": "Median implied volatility from daily BTC range markets vs Deribit's 30-day DVOL "
                 "and the volatility that actually happened",
        "brier_y": "Brier score (lower = more accurate)",
        "brier_t": "The crowd out-forecasts the options desk a day ahead; by the last hour it's a tie",
        "brier_s": "Accuracy of each model's probabilities for the winning price bracket, "
                   "averaged across days",
        "suffix": "",
    },
    "es": {
        "xlabel": "Horas antes del cierre del mercado (12:00 ET)",
        "footer": "Datos: mercados diarios de rango de BTC en Polymarket, DVOL de Deribit, Binance "
                  "BTCUSDT · {n} días, jul–oct 2026 · Ivan Alonso H.",
        "realized": "Realizada", "dvol_adj": "DVOL ajustado por hora",
        "vol_y": "Volatilidad anualizada (%)",
        "vol_t": "Polymarket ajusta la volatilidad al acercarse el cierre; el DVOL no se mueve",
        "vol_s": "Volatilidad implícita mediana de los mercados diarios de BTC frente al DVOL a 30 días "
                 "de Deribit y la volatilidad real",
        "brier_y": "Brier score (más bajo = más preciso)",
        "brier_t": "Con un día de antelación, la multitud acierta más que el mercado de opciones",
        "brier_s": "Precisión de las probabilidades de cada modelo sobre el tramo ganador, "
                   "media de todos los días · en la última hora, empate",
        "suffix": "_es",
    },
}


def style_axes(ax, t):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=10, length=0, pad=6)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xticks(range(len(HORIZONS)))
    ax.set_xticklabels([f"{h}h" for h in HORIZONS])
    ax.set_xlabel(t["xlabel"], color=INK_2, fontsize=10, labelpad=8)


def line(ax, y, color, label, marker="o", dashed=False, label_dy=0):
    x = np.arange(len(y))
    ax.plot(x, y, color=color, linewidth=2, linestyle=(0, (4, 3)) if dashed else "-",
            marker=marker, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
            label=label, zorder=3)
    ax.annotate(label, (x[-1], y[-1]), xytext=(10, label_dy), textcoords="offset points",
                va="center", fontsize=10, color=INK)


def header(fig, title, subtitle):
    fig.text(0.06, 0.95, title, fontsize=14, fontweight="bold", color=INK, va="top")
    fig.text(0.06, 0.885, subtitle, fontsize=10, color=INK_2, va="top")


def footer(fig, n_days, t):
    fig.text(0.06, 0.025, t["footer"].format(n=n_days), fontsize=8, color=INK_2)


def fig_volatility(snap, t):
    snap = snap.copy()
    snap["realized"] = snap.realized_logret.abs() * np.sqrt(
        YEAR_SECONDS / (snap.horizon_h * 3600)) * np.sqrt(np.pi / 2)
    g = snap.groupby("horizon_h").agg(pm=("pm_vol", "median"), dvol=("dvol", "median"),
                                      real=("realized", "mean")).reindex(HORIZONS) * 100

    fig, ax = plt.subplots(figsize=(9, 5.4), facecolor=SURFACE)
    fig.subplots_adjust(left=0.08, right=0.76, top=0.80, bottom=0.17)
    style_axes(ax, t)
    line(ax, g.dvol.values, C_DVOL, "Deribit DVOL")
    line(ax, g.real.values, C_THIRD, t["realized"], marker="s", dashed=True)
    line(ax, g.pm.values, C_PM, "Polymarket")
    ax.set_ylabel(t["vol_y"], color=INK_2, fontsize=10)
    ax.set_ylim(0, max(g.max()) * 1.15)
    ax.legend(loc="upper left", frameon=False, fontsize=10, labelcolor=INK, ncols=3)
    header(fig, t["vol_t"], t["vol_s"])
    footer(fig, snap.date.nunique(), t)
    out = DOCS / f"fig1_volatilidad{t['suffix']}.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return out


def fig_brier(scores, t):
    g = scores.groupby("horizon_h")[["brier_pm", "brier_dvol", "brier_dvol_seas"]].mean() \
        .reindex(HORIZONS)

    fig, ax = plt.subplots(figsize=(9, 5.4), facecolor=SURFACE)
    fig.subplots_adjust(left=0.08, right=0.76, top=0.80, bottom=0.17)
    style_axes(ax, t)
    line(ax, g.brier_dvol.values, C_DVOL, "Deribit DVOL", label_dy=-16)
    line(ax, g.brier_dvol_seas.values, C_THIRD, t["dvol_adj"], marker="s", dashed=True,
         label_dy=16)
    line(ax, g.brier_pm.values, C_PM, "Polymarket")
    ax.set_ylabel(t["brier_y"], color=INK_2, fontsize=10)
    # Línea, no barras: el eje no necesita empezar en 0 y así se ven las diferencias
    ax.set_ylim(0.15, g.max().max() * 1.08)
    ax.legend(loc="upper right", frameon=False, fontsize=10, labelcolor=INK, ncols=3)
    header(fig, t["brier_t"], t["brier_s"])
    footer(fig, scores.date.nunique(), t)
    out = DOCS / f"fig2_brier{t['suffix']}.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return out


def main():
    DOCS.mkdir(exist_ok=True)
    snap = pd.read_csv(DATA / "fase2_snapshots.csv")
    scores = pd.read_csv(DATA / "fase3_scores.csv")
    for t in TXT.values():
        for f in (fig_volatility(snap, t), fig_brier(scores, t)):
            print(f"Guardado: {f}")


if __name__ == "__main__":
    main()
