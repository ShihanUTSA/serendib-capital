"""
make_figures.py
===============
Regenerates the figures in docs/figures/ from a scan report produced by
src/statarb_ver3_intraday.py.

Usage:
    python scripts/make_figures.py path/to/intraday_signal_report.csv

Every number plotted comes from the CSV. Nothing is simulated.
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Z_THRESHOLD = 1.7          # z_threshold used in the scan being plotted
OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"

PASS_C, FAIL_C, LONG_C, SHORT_C = "#2a6f9e", "#c9c9c9", "#2a6f9e", "#b5484d"


def load(path):
    df = pd.read_csv(path, parse_dates=["Signal_Time"])
    df["is_signal"] = df["Signal"].str.startswith(("BUY", "SELL"))
    return df


def fig_screen(df):
    """Cointegration screen outcome + signal count per pair."""
    order = df.drop_duplicates("Pair")["Pair"].tolist()
    counts = df[df.is_signal].groupby("Pair").size()
    coint = df.groupby("Pair")["Cointegrated"].any()

    vals = [counts.get(p, 0) for p in order]
    cols = [PASS_C if coint.get(p, False) else FAIL_C for p in order]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(range(len(order)), vals, color=cols, height=0.7)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Signals generated")
    ax.set_title(f"Cointegration screen — {len(order)} candidate pairs\n"
                 f"{int(coint.sum())} passed, {int((~coint).sum())} rejected",
                 fontsize=11)
    for i, (v, p) in enumerate(zip(vals, order)):
        if not coint.get(p, False):
            ax.text(0.06, i, "not cointegrated", va="center",
                    fontsize=7.5, color="#808080")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    fig.tight_layout()
    fig.savefig(OUT / "cointegration_screen.png", dpi=150)
    plt.close(fig)


def fig_signals(df):
    """Intraday z-score of every signal, against the entry threshold."""
    from matplotlib.lines import Line2D

    s = df[df.is_signal].copy()
    s["clock"] = s["Signal_Time"].dt.hour + s["Signal_Time"].dt.minute / 60
    pairs = sorted(s.Pair.unique())
    marks = dict(zip(pairs, "os^Dv"))

    fig, ax = plt.subplots(figsize=(10, 5))
    for pair in pairs:
        g = s[s.Pair == pair]
        for mask, col in ((g.Signal.str.startswith("BUY"), LONG_C),
                          (g.Signal.str.startswith("SELL"), SHORT_C)):
            sub = g[mask]
            ax.scatter(sub.clock, sub.Z_Score, marker=marks[pair], s=75,
                       facecolor=col, edgecolor="white", lw=0.8, zorder=3)

    for y in (Z_THRESHOLD, -Z_THRESHOLD):
        ax.axhline(y, ls="--", lw=1, color="#666")
    ax.axhline(0, lw=0.8, color="black")
    ax.text(9.5, Z_THRESHOLD + 0.05, f"entry threshold  +{Z_THRESHOLD}",
            fontsize=8, color="#666")
    ax.text(9.5, -Z_THRESHOLD - 0.13, f"entry threshold  -{Z_THRESHOLD}",
            fontsize=8, color="#666")

    pair_h = [Line2D([], [], marker=marks[p], ls="", color="#555",
                     markersize=7, label=p) for p in pairs]
    dir_h = [Line2D([], [], marker="o", ls="", color=SHORT_C, markersize=7,
                    label="short the spread  (sell A / buy B)"),
             Line2D([], [], marker="o", ls="", color=LONG_C, markersize=7,
                    label="long the spread  (buy A / sell B)")]
    leg1 = ax.legend(handles=pair_h, fontsize=8, frameon=False,
                     loc="upper left", bbox_to_anchor=(0.01, 0.80), title="Pair",
                     title_fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=dir_h, fontsize=8, frameon=False,
              loc="upper left", bbox_to_anchor=(0.30, 0.78), title="Direction",
              title_fontsize=8)

    ax.set_xlabel("Time of day (ET)")
    ax.set_ylabel("Rolling z-score of log-price spread")
    ax.set_title(f"Signals fired - {s.Signal_Time.dt.date.iloc[0]}  "
                 f"({len(s)} signals across {len(pairs)} pairs, 5-min bars)",
                 fontsize=11)
    ax.set_xlim(9.4, 16.1)
    ax.set_ylim(-2.35, 2.35)
    ax.set_xticks(range(10, 17))
    ax.set_xticklabels([f"{h}:00" for h in range(10, 17)])
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6)
    fig.tight_layout()
    fig.savefig(OUT / "intraday_signals.png", dpi=150)
    plt.close(fig)


def fig_magnitude(df):
    """Distribution of |z| at entry, relative to the threshold."""
    s = df[df.is_signal]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(s.Z_Score.abs(), bins=12, range=(1.6, 2.3),
            color=PASS_C, edgecolor="white")
    from matplotlib.ticker import MaxNLocator
    ax.axvline(Z_THRESHOLD, ls="--", lw=1.2, color="#b5484d")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(0, 5)
    ax.text(Z_THRESHOLD + 0.015, 4.55, f"entry threshold {Z_THRESHOLD}",
            fontsize=8, color="#b5484d")
    ax.set_xlabel("|z| at entry")
    ax.set_ylabel("Signals")
    ax.set_title(f"Entry magnitude — all {len(s)} signals fall in "
                 f"[{s.Z_Score.abs().min():.2f}, {s.Z_Score.abs().max():.2f}]",
                 fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    fig.tight_layout()
    fig.savefig(OUT / "entry_magnitude.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "src/intraday_signal_report.csv"
    OUT.mkdir(parents=True, exist_ok=True)
    df = load(path)
    fig_screen(df); fig_signals(df); fig_magnitude(df)
    n_sig = int(df.is_signal.sum())
    n_pass = int(df.groupby("Pair")["Cointegrated"].any().sum())
    print(f"pairs={df.Pair.nunique()}  cointegrated={n_pass}  signals={n_sig}")
    print(f"|z| range = [{df[df.is_signal].Z_Score.abs().min():.4f}, "
          f"{df[df.is_signal].Z_Score.abs().max():.4f}]")
    print(f"figures written to {OUT}")
