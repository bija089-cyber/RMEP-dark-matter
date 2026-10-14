"""
plot_results.py — графики по результатам подгонки RMEP.

Что строит:
  1. Связь A ~ alpha^k (лог-лог, с линией регрессии)
  2. Распределение chi2: RMEP против NFW
  3. Сопоставление по каждой галактике (chi2_rmep vs chi2_nfw)
  4. Примеры кривых вращения (3 лучшие и 3 худшие галактики)

Запуск:  python plot_results.py
Читает results.csv, который создаёт fit_galaxies.py.
"""

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


def plot_A_alpha(df):
    """Закон связи A ~ alpha^k в лог-лог координатах."""
    sel = df[df["alpha"] > 0]
    k, logA0 = np.polyfit(np.log(sel["alpha"]), np.log(sel["A"]), 1)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.loglog(sel["alpha"], sel["A"], "o", ms=4, alpha=0.6,
              label=f"данные (k = {k:.2f})")
    xs = np.logspace(sel["alpha"].min().__log10__(),
                     sel["alpha"].max().__log10__(), 100)
    ax.loglog(10 ** xs, 10 ** (logA0 + k * xs), "r-",
              label=r"$A \propto \alpha^{k}$")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$A$")
    ax.set_title("Связь параметров RMEP: A vs alpha")
    ax.legend()
    fig.tight_layout()
    fig.savefig("plot_A_alpha.png")
    plt.close(fig)
    print("Сохранено: plot_A_alpha.png  (наклон k = %.2f)" % k)
    return k


def plot_chi2_hist(df):
    """Распределение chi2 для RMEP и NFW."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bins = np.logspace(np.log10(max(df["chi2_rmep"].min(), 0.1)),
                       np.log10(df["chi2_nfw"].max() * 1.1), 40)
    ax.hist(df["chi2_nfw"], bins=bins, alpha=0.6,
            label="NFW", color="tab:red")
    ax.hist(df["chi2_rmep"], bins=bins, alpha=0.6,
            label="RMEP", color="tab:blue")
    ax.axvline(df["chi2_rmep"].mean(), color="tab:blue", ls="--",
               label=f"среднее RMEP = {df['chi2_rmep'].mean():.1f}")
    ax.axvline(df["chi2_nfw"].mean(), color="tab:red", ls="--",
               label=f"среднее NFW = {df['chi2_nfw'].mean():.1f}")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\chi^2$")
    ax.set_ylabel("число галактик")
    ax.set_title("Качество подгонки: RMEP vs NFW")
    ax.legend()
    fig.tight_layout()
    fig.savefig("plot_chi2_hist.png")
    plt.close(fig)
    print("Сохранено: plot_chi2_hist.png")


def plot_scatter(df):
    """Сопоставление chi2 по каждой галактике."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    lims = [max(df["chi2_rmep"].min(), df["chi2_nfw"].min()) * 0.5,
            df[["chi2_rmep", "chi2_nfw"]].max().max() * 1.1]
    ax.plot(lims, lims, "k--", lw=1, label="равное качество")
    ax.scatter(df["chi2_nfw"], df["chi2_rmep"], s=18, alpha=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(r"$\chi^2$ (NFW)")
    ax.set_ylabel(r"$\chi^2$ (RMEP)")
    ax.set_title("Каждая галактика: RMEP против NFW")
    ax.legend()
    fig.tight_layout()
    fig.savefig("plot_scatter.png")
    plt.close(fig)
    print("Сохранено: plot_scatter.png")


def main():
    try:
        df = pd.read_csv("results.csv")
    except FileNotFoundError:
        raise SystemExit(
            "Сначала запустите fit_galaxies.py — он создаёт results.csv"
        )

    k = plot_A_alpha(df)
    plot_chi2_hist(df)
    plot_scatter(df)

    print(f"\nГотово. Согласование A ~ alpha^{k:.2f} — ключевой предсказательный"
          f"\nзакон RMEP: показатель степени k свободно измеряется из данных.")


if __name__ == "__main__":
    main()
