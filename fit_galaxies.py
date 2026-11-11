# -*- coding: utf-8 -*-
"""
fit_galaxies.py — пайплайн подгонки RMEP и NFW по выборке галактик (v5.0).

Режимы:
  python fit_galaxies.py              # демо-режим: синтетическая выборка
  python fit_galaxies.py sparc/       # реальный запуск по каталогу SPARC

Подгонка использует канонический профиль RMEP из rmep_model.py (экспоненциальная
сфера, разд. 6.8.1): параметры (V0, alpha), rho0 = V0^2/(8*pi*G*alpha^2).

Выход: results.csv со столбцами
  name, n, chi2_rmep, chi2_nfw, V0, alpha, rho0, degenerate,
  vmax_obs, vmax_mod
+ сводка с канонической регрессией alpha ~ A*(Vmax/100)^k (сверка с документом:
  6.8 — A ~ 5 kpc, k ~ 1.89).
"""

import sys
import os
import glob
import numpy as np
import pandas as pd

from rmep_model import RMEPProfile, NFWProfile, fit_galaxy

# Регрессия из документа 6.8 (для сверки в сводке)
REG_A_DOC = 5.0     # kpc
REG_K_DOC = 1.89


# ----------------------------------------------------------------------
# Синтетическая выборка (демо-режим) — генерируется каноническим профилем
# ----------------------------------------------------------------------
def make_demo_sample(n=30, seed=42):
    rng = np.random.default_rng(seed)
    prof = RMEPProfile()
    rows = []
    for i in range(n):
        alpha = float(np.exp(rng.uniform(np.log(0.5), np.log(10.0))))
        V0 = float(rng.uniform(40.0, 300.0))
        r_d = rng.uniform(0.5, 6.0)
        radii = np.linspace(0.3 * r_d, 12 * r_d, 25)
        v_disk = 0.7 * prof.v(radii, V0, alpha) * np.tanh(radii / r_d)
        v_obs = np.sqrt(v_disk ** 2 + prof.v2(radii, V0, alpha)) \
            + rng.normal(0, 5.0, radii.shape)
        rows.append({
            "name": f"demo_{i:03d}",
            "r": radii,
            "v_obs": v_obs,
            "v_disk": v_disk,
            "v_err": np.full_like(radii, 5.0),
        })
    return rows


# ----------------------------------------------------------------------
# Загрузка реальных данных SPARC (Lelli et al. 2016)
# ----------------------------------------------------------------------
def load_sparc(path):
    files = sorted(glob.glob(os.path.join(path, "*_rotmod.dat")))
    if not files:
        raise FileNotFoundError(f"В каталоге {path} нет файлов *_rotmod.dat")
    sample = []
    for f in files:
        data = pd.read_csv(f, sep=r"\s+", comment="#",
                           names=["r", "v_obs", "v_err", "v_gas", "v_disk"])
        data = data.dropna()
        if len(data) < 3:
            continue
        sample.append({
            "name": os.path.basename(f).replace("_rotmod.dat", ""),
            "r": data["r"].to_numpy(),
            "v_obs": data["v_obs"].to_numpy(),
            "v_disk": np.hypot(data["v_gas"].to_numpy(), data["v_disk"].to_numpy()),
            "v_err": np.maximum(data["v_err"].to_numpy(), 1.0),
        })
    return sample


# ----------------------------------------------------------------------
# Каноническая регрессия alpha ~ A*(Vmax/100)^k
# ----------------------------------------------------------------------
def alpha_regression(df):
    """Регрессия по невырожденным точкам с ненулевыми alpha и vmax_mod."""
    sel = df[(~df["degenerate"]) & (df["alpha"] > 0) & (df["vmax_mod"] > 0)]
    if len(sel) < 5:
        return None
    x = np.log10(sel["vmax_mod"] / 100.0)
    y = np.log10(sel["alpha"])
    k, logA = np.polyfit(x, y, 1)
    pear = float(np.corrcoef(x, y)[0, 1])
    from scipy.stats import spearmanr
    spear = float(spearmanr(x, y).statistic) if hasattr(spearmanr(x, y), "statistic") \
        else float(spearmanr(x, y)[0])
    return {"A": 10.0 ** logA, "k": float(k), "pearson": pear,
            "spearman": spear, "n": len(sel)}


# ----------------------------------------------------------------------
# Основной цикл
# ----------------------------------------------------------------------
def run(sample, verbose=None):
    if verbose is None:
        verbose = len(sample) <= 50
    rmep_prof = RMEPProfile()
    nfw_prof = NFWProfile()
    records = []

    for gal in sample:
        fit_r = fit_galaxy(gal["r"], gal["v_obs"], gal["v_err"],
                           gal["v_disk"], rmep_prof)
        fit_n = fit_galaxy(gal["r"], gal["v_obs"], gal["v_err"],
                           gal["v_disk"], nfw_prof)
        V0, alpha = fit_r["params"]["V0"], fit_r["params"]["alpha"]
        records.append({
            "name": gal["name"],
            "n": len(gal["r"]),
            "chi2_rmep": fit_r["chi2"],
            "chi2_nfw": fit_n["chi2"],
            "V0": V0,
            "alpha": alpha,
            "rho0": rmep_prof.rho0(V0, alpha),
            "degenerate": fit_r["degenerate"],
            "vmax_obs": float(np.max(gal["v_obs"])),
            "vmax_mod": 0.4407 * V0,
        })
        if verbose:
            degen = " [DEGEN]" if fit_r["degenerate"] else ""
            print(f'{gal["name"]:>12s}:  chi2(RMEP)={fit_r["chi2"]:9.2f}   '
                  f'chi2(NFW)={fit_n["chi2"]:9.2f}   '
                  f'V0={V0:7.1f}  alpha={alpha:7.2f}{degen}')

    df = pd.DataFrame(records)
    good = df[~df["degenerate"]]

    print("\n" + "=" * 64)
    print("СВОДКА ПО ВЫБОРКЕ")
    print("=" * 64)
    print(f"Галактик обработано      : {len(df)}")
    print(f"Вырожденных (цензура)    : {int(df['degenerate'].sum())}")
    if len(good):
        print(f"Средний chi2 (RMEP)      : {good['chi2_rmep'].mean():10.2f}")
        print(f"Средний chi2 (NFW)       : {good['chi2_nfw'].mean():10.2f}")
        frac = (good['chi2_rmep'] < good['chi2_nfw']).mean() * 100
        print(f"RMEP лучше NFW в         : {frac:5.1f}% невырожденных")

    reg = alpha_regression(df)
    if reg:
        print("\nКаноническая регрессия alpha ~ A*(Vmax/100)^k (не вырожденные):")
        print(f"  A = {reg['A']:.2f} kpc   (документ 6.8: A = {REG_A_DOC} kpc)")
        print(f"  k = {reg['k']:.2f}       (документ 6.8: k = {REG_K_DOC})")
        print(f"  Пирсон r = {reg['pearson']:.2f}, Спирмен = {reg['spearman']:.2f}, "
              f"N = {reg['n']}")

    print("=" * 64)
    df.to_csv("results.csv", index=False)
    print("\nТаблица сохранена в results.csv")
    return df


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sample = load_sparc(sys.argv[1])
        print(f"Загружено галактик из SPARC: {len(sample)}")
    else:
        print("SPARC не указан — работает демо-режим (синтетика)")
        sample = make_demo_sample(30)
    if not sample:
        raise SystemExit("Выборка пуста — проверьте каталог с данными")
    run(sample)
