# -*- coding: utf-8 -*-
"""
nbody_skeleton.py — N-body скелет RMEP (версия 5.0).

Проверяет динамическую самосогласованность канонического регуляторного
профиля (экспоненциальная сфера, разд. 6.8.1):

  * ускорение от поля RMEP:  a(R) = -dPhi/dR * r_hat = -GM(R)/R^2 * r_hat
    — не подгонка под кривую вращения, а градиент того же потенциала,
    что и в rmep_model.RMEPProfile.potential();
  * полная энергия: кинетическая + взаимная гравитация (Пламмер) +
    потенциал поля RMEP per unit mass;
  * интегратор leapfrog (kick-drift-kick), контроль дрейфа энергии;
  * ограничение из Bullet Cluster: характерное время отклика поля
    tau_s = alpha / v_max должно быть tau_s <~ 1e8 лет — иначе поле
    успевало бы выравниваться внутри пролетающего субскопления и
    смещение массы (lensing vs gas) не наблюдалось бы.

Запуск:  python nbody_skeleton.py
Вывод:   nbody_energy_drift.png + сводка проверок в stdout.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rmep_model import RMEPProfile, G_KPC

# 1 кпк / (1 км/с) = 1.0227e-3 Гпк^-1 * 1e9 ... точнее: (1 кпк / 1 км/с) в Гиголетах
GYR_PER_KPC_KMS = 1.02271  # 1 кпк/(км/с) = 1.0227 Гиголет
TAU_S_BULLET_MAX = 1.0e8   # лет — верхний предел из Bullet Cluster


def field_acceleration(r, V0, alpha):
    """Радиальное ускорение поля RMEP на частицу единичной массы (км/с)^2/кпк.

    Самосогласованно: a(R) = -GM(R)/R^2 — градиент RMEPProfile.potential.
    Возвращает скалярную величину |a| (к центру).
    """
    r = np.maximum(np.asarray(r, float), 1e-9)
    return G_KPC * prof_mass(r, V0, alpha) / r ** 2


def prof_mass(r, V0, alpha):
    """Масса внутри R для экспоненциальной сферы (аналитически)."""
    x = np.maximum(np.asarray(r, float) / max(alpha, 1e-6), 0.0)
    return 8.0 * np.pi * (V0 ** 2 / (8.0 * np.pi * G_KPC * alpha ** 2)) \
        * alpha ** 3 * (1.0 - np.exp(-x) * (1.0 + x + x * x / 2.0))


def tau_s_years(alpha, V0):
    """Характерное время отклика поля: tau_s = alpha / v_max.

    alpha в кпк, v_max = 0.4407*V0 в км/с -> годы (1 кпк/(км/с) = 1.0227 Гиголет).
    """
    v_max = 0.4407 * V0
    return alpha / max(v_max, 1e-6) * GYR_PER_KPC_KMS * 1e9  # лет


def bullet_check(alpha, V0):
    """Проверка ограничения Bullet Cluster: tau_s <~ 1e8 лет."""
    tau = tau_s_years(alpha, V0)
    return tau, tau <= TAU_S_BULLET_MAX


def acceleration(pos, m, V0, alpha, eps2=0.25):
    """Ускорения всех частиц (N, 3).

    Взаимная гравитация (смягчение Пламмера, eps = sqrt(eps2) кпк)
    плюс самосогласованное радиальное поле RMEP (градиент потенциала).
    """
    d = pos[:, None, :] - pos[None, :, :]          # (N, N, 3)
    r2 = np.sum(d * d, axis=2) + eps2
    np.fill_diagonal(r2, np.inf)                   # без самодействия
    inv_r3 = 1.0 / r2 ** 1.5
    acc_g = -G_KPC * np.sum((m * inv_r3)[:, :, None] * d, axis=0)
    # поле RMEP: a = -GM(R)/R^2 * r_hat (самосогласованный градиент потенциала)
    r = np.linalg.norm(pos, axis=1)
    a_reg = field_acceleration(r, V0, alpha)
    return acc_g - pos * (a_reg / np.maximum(r, 1e-9))[:, None]


def total_energy(pos, vel, m, V0, alpha, eps2=0.25):
    """Полная энергия: кинетическая + гравитация + потенциал поля RMEP."""
    K = 0.5 * float(np.sum(m * np.sum(vel ** 2, axis=1)))
    d = pos[:, None, :] - pos[None, :, :]
    r2 = np.sum(d * d, axis=2) + eps2
    iu = np.triu_indices(len(m), 1)
    U_g = -G_KPC * float(np.sum(m[iu[0]] * m[iu[1]] / np.sqrt(r2[iu])))
    # потенциал поля RMEP per unit mass — из ядра модели (Phi(inf) = 0)
    r = np.linalg.norm(pos, axis=1)
    _prof = RMEPProfile()
    U_r = float(np.sum(m * _prof.potential(r, V0, alpha)))
    return K + U_g + U_r


def self_consistency_test(V0=120.0, alpha=0.5):
    """Тест самосогласованности: круговая орбита в чистом поле RMEP.

    Частица на радиусе R со скоростью v_c = sqrt(GM(R)/R) должна оставаться
    на круговой орбите (радиус не уплывает), если ускорение и потенциал
    согласованы. Интегрируем 20 оборотов leapfrog'ом.
    """
    prof = RMEPProfile()
    R0 = 3.38 * alpha                       # радиус пика кривой вращения
    v_c = np.sqrt(G_KPC * prof_mass(R0, V0, alpha) / R0)
    pos = np.array([[R0, 0.0, 0.0]])
    vel = np.array([[0.0, v_c, 0.0]])
    m = np.array([1.0])                     # масса частицы не влияет на поле
    T_orbit = 2.0 * np.pi * R0 / v_c        # кпк/(км/с)
    n_steps = 2000
    dt = T_orbit / 100.0                   # 20 оборотов за 2000 шагов

    radii = [R0]
    for _ in range(n_steps):
        acc = acceleration(pos, m, V0, alpha)
        vel += acc * (dt / 2)
        pos += vel * dt
        acc = acceleration(pos, m, V0, alpha)
        vel += acc * (dt / 2)
        radii.append(float(np.linalg.norm(pos[0])))

    radii = np.array(radii)
    drift_R = (radii.max() - radii.min()) / R0
    return v_c, drift_R, radii


def run_demo(n_steps=200, dt=0.005, eps=0.5, m_particle=1e3):
    """Демо: кольцо из 500 частиц на круговых орбитах в поле RMEP."""
    prof = RMEPProfile()
    V0, alpha = 120.0, 0.5                  # канонические параметры поля
    N = 500
    radii = np.linspace(2.0, 10.0, N)
    angles = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    pos = np.zeros((N, 3))
    pos[:, 0] = radii * np.cos(angles)
    pos[:, 1] = radii * np.sin(angles)
    vel = np.zeros((N, 3))
    vc = np.sqrt(prof.v2(radii, V0, alpha))
    vel[:, 0] = -vc * np.sin(angles)               # круговые орбиты
    vel[:, 1] = vc * np.cos(angles)
    m = np.full(N, m_particle)

    eps2 = eps ** 2
    acc = acceleration(pos, m, V0, alpha, eps2)
    E0 = total_energy(pos, vel, m, V0, alpha, eps2)
    drift = []
    for _ in range(n_steps):
        vel += acc * (dt / 2)                      # kick
        pos += vel * dt                            # drift
        acc = acceleration(pos, m, V0, alpha, eps2)
        vel += acc * (dt / 2)                      # kick
        drift.append((total_energy(pos, vel, m, V0, alpha, eps2)
                      - E0) / abs(E0))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(drift)
    ax.set_xlabel("step")
    ax.set_ylabel("total energy drift (rel.)")
    ax.set_title("N-body skeleton: energy conservation (leapfrog, canonical RMEP)")
    fig.tight_layout()
    fig.savefig("nbody_energy_drift.png", dpi=120)
    plt.close(fig)

    return drift


if __name__ == "__main__":
    print("=" * 64)
    print("N-BODY SKELETON RMEP — канонический профиль (экспоненциальная сфера)")
    print("=" * 64)

    # --- Тест 1: самосогласованность ускорения и потенциала ---
    v_c, drift_R, _ = self_consistency_test()
    print(f"\n[1] Круговая орбита: v_c = {v_c:.2f} км/с на R = 3.38*alpha")
    print(f"    относительный разброс радиуса за 20 оборотов: {drift_R:.2e}"
          f"  (должен быть ~ 0, если a и Phi согласованы)")

    # --- Тест 2: дрейф энергии в кольце из 500 частиц ---
    drift = run_demo()
    max_drift = max(abs(d) for d in drift)
    print(f"\n[2] Дрейф энергии за 200 шагов leapfrog: {max_drift:.2e}"
          f"  (< 0.1% ожидается)")
    print("    Сохранено: nbody_energy_drift.png")

    # --- Тест 3: ограничение Bullet Cluster tau_s <~ 1e8 лет ---
    print("\n[3] Ограничение Bullet Cluster: tau_s = alpha / v_max <~ 1e8 лет")
    cases = [
        ("карлик (SPARC-тип):  alpha=2,  V0=80 ", 2.0, 80.0),
        ("средняя галактика:    alpha=5,  V0=150", 5.0, 150.0),
        ("массивная:            alpha=12, V0=300", 12.0, 300.0),
        ("скопление (кластер):  alpha=50, V0=800", 50.0, 800.0),
    ]
    for label, a_, v_ in cases:
        tau, ok = bullet_check(a_, v_)
        status = "OK" if ok else "НАРУШЕНО"
        print(f"    {label}: tau_s = {tau:.2e} лет  -> {status}")

    print("\n" + "=" * 64)
