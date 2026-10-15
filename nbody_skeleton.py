"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rmep_model import RMEPProfile

G_KPC = 4.30091e-6   # кпк * (км/с)^2 / Msun


def acceleration(pos, m, prof, A, alpha, r_d, eps2=2.5e-3):
    """Ускорения всех частиц (N, 3).

    Взаимная гравитация (смягчение Пламмера, eps = sqrt(eps2) кпк)
    плюс радиальное регуляторное поле RMEP.
    """
    d = pos[:, None, :] - pos[None, :, :]          # (N, N, 3)
    r2 = np.sum(d * d, axis=2) + eps2
    np.fill_diagonal(r2, np.inf)                   # без самодействия
    inv_r3 = 1.0 / r2 ** 1.5
    acc_g = -G_KPC * np.sum((m * inv_r3)[:, :, None] * d, axis=0)
    # регуляторное поле RMEP: радиальное ускорение из кривой вращения
    r = np.linalg.norm(pos, axis=1)
    v2 = prof.v2(r, A, alpha, r_d)
    a_reg = -v2 / np.maximum(r, 1e-6)              # направлено к центру
    return acc_g + pos * (a_reg / np.maximum(r, 1e-6))[:, None]


def total_energy(pos, vel, m, prof, A, alpha, r_d, eps2=2.5e-3):
    """Полная энергия: кинетическая + гравитация + потенциал поля RMEP."""
    K = 0.5 * float(np.sum(m * np.sum(vel ** 2, axis=1)))
    d = pos[:, None, :] - pos[None, :, :]
    r2 = np.sum(d * d, axis=2) + eps2
    iu = np.triu_indices(len(m), 1)
    U_g = -G_KPC * float(np.sum(m[iu[0]] * m[iu[1]] / r2[iu]))
    r_c = alpha * r_d
    r = np.linalg.norm(pos, axis=1)
    U_r = float(np.sum(m * (-A ** 2 * r_c / (r_c + np.maximum(r, 1e-6)))))
    return K + U_g + U_r


def run_demo(n_steps=200, dt=0.005, eps=0.05, m_particle=1e3):
    """Демо: кольцо из 500 частиц на круговых орбитах в поле RMEP."""
    prof = RMEPProfile()
    A, alpha, r_d = 120.0, 0.5, 3.0
    N = 500
    radii = np.linspace(2.0, 10.0, N)
    angles = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    pos = np.zeros((N, 3))
    pos[:, 0] = radii * np.cos(angles)
    pos[:, 1] = radii * np.sin(angles)
    vel = np.zeros((N, 3))
    vc = np.sqrt(prof.v2(radii, A, alpha, r_d))
    vel[:, 0] = -vc * np.sin(angles)               # круговые орбиты
    vel[:, 1] = vc * np.cos(angles)
    m = np.full(N, m_particle)

    eps2 = eps ** 2
    acc = acceleration(pos, m, prof, A, alpha, r_d, eps2)
    E0 = total_energy(pos, vel, m, prof, A, alpha, r_d, eps2)
    drift = []
    for _ in range(n_steps):
        vel += acc * (dt / 2)                      # kick
        pos += vel * dt                            # drift
        acc = acceleration(pos, m, prof, A, alpha, r_d, eps2)
        vel += acc * (dt / 2)                      # kick
        drift.append((total_energy(pos, vel, m, prof, A, alpha, r_d, eps2)
                      - E0) / abs(E0))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(drift)
    ax.set_xlabel("step")
    ax.set_ylabel("total energy drift (rel.)")
    ax.set_title("N-body skeleton: energy conservation (leapfrog)")
    fig.tight_layout()
    fig.savefig("nbody_energy_drift.png", dpi=120)
    plt.close(fig)

    max_drift = max(abs(d) for d in drift)
    print("Saved: nbody_energy_drift.png")
    print(f"Max relative total-energy drift over {n_steps} steps: "
          f"{max_drift:.2e}  (< 0.1% expected for this demo setup)")
    return drift


if __name__ == "__main__":
    run_demo()
