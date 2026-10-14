"""
fit_galaxies.py — пайплайн подгонки RMEP и NFW по выборке галактик.

Режимы:
  python fit_galaxies.py              # демо-режим: синтетическая выборка из 177 галактик
  python fit_galaxies.py sparc/       # реальный запуск по каталогу SPARC

Вывод: таблица результатов, средние chi2, сводка по модели RMEP.
"""

import sys
import numpy as np
import pandas as pd

from rmep_model import RMEPProfile, NFWProfile, fit_galaxy


# ----------------------------------------------------------------------
# Синтетическая выборка (демо-режим)
# ----------------------------------------------------------------------
def make_demo_sample(n=177, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        v_flat = rng.uniform(40, 300)          # км/с — плоская скорость
        r_d = rng.uniform(0.5, 6.0)            # кпк — масштаб диска
        radii = np.linspace(0.3 * r_d, 12 * r_d, 25)
        # Синтетическая кривая: диск + RMEP-подобная платформа
        v_disk = v_flat * np.tanh(radii / r_d)
        v_obs = v_flat + rng.normal(0, 5, radii.shape)
        rows.append({
            "name": f"demo_{i:03d}",
            "r": radii,
            "v_obs": v_obs,
            "v_disk": v_disk,
            "v_err": np.full_like(radii, 5.0),
        })
    return rows


# ----------------------------------------------------------------------
# Загрузка реальных данных SPARC (RotcurveTOO / lelli et al. 2016)
# ----------------------------------------------------------------------
def load_sparc(path):
    import glob, os
    files = sorted(glob.glob(os.path.join(path, "*_rotmod.dat")))
    if not files:
        raise FileNotFoundError(f"В каталоге {path} нет файлов *_rotmod.dat")
    sample = []
    for f in files:
        data = pd.read_csv(f, sep=r"\s+", comment="#",
                           names=["r", "v_obs", "v_err", "v_gas", "v_disk"])
        data = data.dropna()
        if len(data) < 10:
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
# Основной цикл
# ----------------------------------------------------------------------
def run(sample, verbose=False):
    rmep_prof = RMEPProfile()
    nfw_prof = NFWProfile()
    records = []

    for gal in sample:
        fit_r = fit_galaxy(gal["r"], gal["v_obs"], gal["v_err"],
                           gal["v_disk"], rmep_prof)
        fit_n = fit_galaxy(gal["r"], gal["v_obs"], gal["v_err"],
                           gal["v_disk"], nfw_prof)
        records.append({
            "name": gal["name"],
            "chi2_rmep": fit_r["chi2"],
            "chi2_nfw": fit_n["chi2"],
            "A": fit_r["params"]["A"],
            "alpha": fit_r["params"]["alpha"],
            "r_d": gal["r"].max() / 3.0,       # грубая оценка масштаба
        })
        if verbose:
            print(f'{gal["name"]:>12s}:  chi2(RMEP)={fit_r["chi2"]:8.2f}   '
                  f'chi2(NFW)={fit_n["chi2"]:8.2f}')

    df = pd.DataFrame(records)

    print("\n" + "=" * 62)
    print("СВОДКА ПО ВЫБОРКЕ")
    print("=" * 62)
    print(f"Галактик обработано : {len(df)}")
    print(f"Средний chi2 (RMEP) : {df['chi2_rmep'].mean():10.2f}")
    print(f"Средний chi2 (NFW)  : {df['chi2_nfw'].mean():10.2f}")
    print(f"RMEP лучше в        : {(df['chi2_rmep'] < df['chi2_nfw']).mean() * 100:5.1f}% случаев")
    print("=" * 62)
    print("\nЗакон связи A ~ alpha^k:")
    sel = df[df["alpha"] > 0]
    k, logA0 = np.polyfit(np.log(sel["alpha"]), np.log(sel["A"]), 1)
    print(f"  наклон k = {k:.2f}")

    df.to_csv("results.csv", index=False)
    print("\nТаблица сохранена в results.csv")
    return df


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sample = load_sparc(sys.argv[1])
        print(f"Загружено галактик из SPARC: {len(sample)}")
    else:
        print("SPARC не указан — работает демо-режим (синтетика, 177 галактик)")
        sample = make_demo_sample(177)
    run(sample)
