# -*- coding: utf-8 -*-
"""
plot_results.py — графики по результатам подгонки RMEP против NFW (SPARC).

Читает таблицу результатов и строит четыре графика:
  1. plot_regression.png — каноническая регрессия alpha ~ (Vmax/100)^k
     (разд. 6.8 документа: alpha ≈ 5 kpc*(Vmax/100)^1.89) — ГЛАВНЫЙ график
  2. plot_alpha_hist.png — распределение alpha по группам (<10 / 10-100 / degen)
  3. plot_chi2_hist.png  — распределение chi2: RMEP против NFW
  4. plot_scatter.png    — поточечное сопоставление chi2 по галактикам

Запуск:
  python plot_results.py                      # читает results.csv нового пайплайна
  python plot_results.py fit_results_with_Mb.csv   # старый прогон
Вырожденные решения (degenerate=True) исключаются из всех статистик
и помечаются отдельным цветом — цензурирование из протокола разд. 6.8.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # сохраняем в файлы, без окна показа
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})

# Пары имён колонок, которые встречаются в разных прогонах
ALIASES = {
    "chi2_rmep": "chi2_rmep", "chi2_rnep": "chi2_rmep",   # опечатка старого прогона
}


def load_table(path):
    """Загрузка таблицы результатов с нормализацией имён колонок."""
    df = pd.read_csv(path)
    df = df.rename(columns={k: v for k, v in ALIASES.items() if k in df.columns})
    # вырожденность: флаг или режим из xlsx-разметки
    if "degenerate" in df.columns:
        df["degenerate"] = (
            df["degenerate"].astype(str).str.lower()
            .isin(["true", "degen", "1", "yes"])
        )
    elif "mode" in df.columns:
        df["degenerate"] = df["mode"].astype(str).str.strip() == "degen"
    else:
        df["degenerate"] = False
    df["vmax"] = df["vmax_obs"] if "vmax_obs" in df.columns else df["Vmax"]
    return df


def plot_regression(df):
    """Каноническая регрессия alpha ~ A*(Vmax/100)^k — сверка с разд. 6.8."""
    ok = (~df["degenerate"]) & df["alpha"].notna() & (df["alpha"] > 0) & \
         df["vmax"].notna() & (df["vmax"] > 0)
    sel = df[ok]
    x = np.log10(sel["vmax"] / 100.0)
    y = np.log10(sel["alpha"])
    k, logA = np.polyfit(x, y, 1)
    A = 10.0 ** logA
    r_pearson = float(np.corrcoef(x, y)[0, 1])
    from scipy.stats import spearmanr
    rho_sp = spearmanr(sel["vmax"], sel["alpha"]).statistic

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.loglog(sel["vmax"], sel["alpha"], "o", ms=4, alpha=0.55,
              label=f"SPARC, n={len(sel)}")
    xs = np.logspace(np.log10(sel["vmax"].min()),
                     np.log10(sel["vmax"].max()), 100)
    ax.loglog(xs, A * (xs / 100.0) ** k, "r-", lw=1.8,
              label=f"fit: alpha = {A:.1f}*(V/100)^{k:.2f}")
    # документ 6.8: alpha ≈ 5 kpc*(Vmax/100)^1.89
    ax.loglog(xs, 5.0 * (xs / 100.0) ** 1.89, "g--", lw=1.4,
              label="документ 6.8: 5*(V/100)^1.89")
    # вырожденные отдельно
    de = df[df["degenerate"] & df["alpha"].notna() & (df["alpha"] > 0)]
    if len(de):
        ax.loglog(de["vmax"], de["alpha"], "x", ms=6, color="gray",
                  alpha=0.7, label=f"degenerate, n={len(de)}")
    ax.set_xlabel(r"$V_{max}$ (км/с)")
    ax.set_ylabel(r"$\alpha$ (кпк)")
    ax.set_title("Каноническая регрессия RMEP: alpha ~ Vmax")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("plot_regression.png")
    plt.close(fig)

    print("\n--- Сверка с документом 6.8 ---")
    print(f"Регрессия : alpha = {A:6.2f} kpc * (Vmax/100)^{k:5.2f}")
    print(f"Документ  : alpha = 5.00 kpc * (Vmax/100)^1.89")
    print(f"Пирсон r = {r_pearson:.2f}, Спирмен rho = {rho_sp:.2f}")
    print("Сохранено: plot_regression.png")
    return k, A


def plot_alpha_hist(df):
    """Распределение alpha по группам."""
    ok = (~df["degenerate"]) & df["alpha"].notna() & (df["alpha"] > 0)
    sel = df[ok]
    de = df[df["degenerate"] & df["alpha"].notna() & (df["alpha"] > 0)]
    g1 = sel[sel["alpha"] < 10]["alpha"]
    g2 = sel[sel["alpha"] >= 10]["alpha"]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bins = np.logspace(np.log10(max(df["alpha"].min(), 1e-3)),
                       np.log10(df["alpha"].max() * 1.1), 40)
    ax.hist(g1, bins=bins, alpha=0.7, color="tab:blue",
            label=f"alpha < 10 кпк (n={len(g1)})")
    ax.hist(g2, bins=bins, alpha=0.7, color="tab:orange",
            label=f"alpha >= 10 кпк (n={len(g2)})")
    if len(de):
        ax.hist(de["alpha"], bins=bins, alpha=0.7, color="gray",
                label=f"degenerate (n={len(de)})")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\alpha$ (кпк)")
    ax.set_ylabel("число галактик")
    ax.set_title("Распределение alpha")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("plot_alpha_hist.png")
    plt.close(fig)
    print("Сохранено: plot_alpha_hist.png")


def plot_chi2_hist(df):
    """Распределение chi2 для RMEP и NFW (вырожденные исключены)."""
    sel = df[~df["degenerate"]]
    lo = max(sel[["chi2_rmep", "chi2_nfw"]].min().min(), 0.05)
    hi = sel[["chi2_rmep", "chi2_nfw"]].max().max() * 1.1
    bins = np.logspace(np.log10(lo), np.log10(hi), 40)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(sel["chi2_nfw"], bins=bins, alpha=0.6, color="tab:red", label="NFW")
    ax.hist(sel["chi2_rmep"], bins=bins, alpha=0.6, color="tab:blue", label="RMEP")
    ax.axvline(sel["chi2_rmep"].mean(), color="tab:blue", ls="--",
               label=f"среднее RMEP = {sel['chi2_rmep'].mean():.1f}")
    ax.axvline(sel["chi2_nfw"].mean(), color="tab:red", ls="--",
               label=f"среднее NFW = {sel['chi2_nfw'].mean():.1f}")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\chi^2$")
    ax.set_ylabel("число галактик")
    ax.set_title("Качество подгонки: RMEP vs NFW (вырожденные исключены)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("plot_chi2_hist.png")
    plt.close(fig)

    frac = (sel["chi2_rmep"] < sel["chi2_nfw"]).mean() * 100
    print("Сохранено: plot_chi2_hist.png "
          f"(RMEP лучше в {frac:.1f}% из {len(sel)} невырожденных)")


def plot_scatter(df):
    """Сопоставление chi2 по каждой галактике."""
    sel = df[~df["degenerate"]]
    de = df[df["degenerate"]]
    lo = max(sel[["chi2_rmep", "chi2_nfw"]].min().min(), 0.05)
    hi = df[["chi2_rmep", "chi2_nfw"]].max().max() * 1.1
    lims = [lo * 0.5, hi]

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(lims, lims, "k--", lw=1, label="равное качество")
    ax.scatter(sel["chi2_nfw"], sel["chi2_rmep"], s=18, alpha=0.6,
               label=f"выборка (n={len(sel)})")
    if len(de):
        ax.scatter(de["chi2_nfw"], de["chi2_rmep"], s=24, alpha=0.7,
                   marker="x", color="gray", label=f"degenerate (n={len(de)})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(r"$\chi^2$ (NFW)")
    ax.set_ylabel(r"$\chi^2$ (RMEP)")
    ax.set_title("Каждая галактика: RMEP против NFW")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("plot_scatter.png")
    plt.close(fig)
    print("Сохранено: plot_scatter.png")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
    df = load_table(path)
    print(f"Загружено галактик: {len(df)} из {path}")
    print(f"Вырожденных (исключены из статистики): {int(df['degenerate'].sum())}")

    plot_regression(df)
    plot_alpha_hist(df)
    plot_chi2_hist(df)
    plot_scatter(df)
    print("\nГотово: 4 PNG-файла.")


if __name__ == "__main__":
    main()
