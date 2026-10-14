
# -*- coding: utf-8 -*-
"""
rmep_model.py — ядро RMEP: Регуляторная модель энергии пустоты.

Содержит:
  * RMEPProfile — профиль регуляторного поля (кривые вращения)
  * NFWProfile  — классический профиль NFW для сравнения
  * fit_galaxy  — подгонка профиля к кривой вращения (chi2-минимизация)
  * hubble_rmep, hubble_lcdm — H(z) для RMEP и LambdaCDM
  * w_eff_rmep  — эффективное уравнение состояния RMEP

Все величины: r в кпк, v в км/с, H0 в км/с/Мпк.
"""

import numpy as np
from scipy.optimize import least_squares

G_KPC = 4.30091e-6   # гравитационная постоянная: кпк * (км/с)^2 / Msun


# ======================================================================
#  ПРОФИЛИ ТЁМНОЙ КОМПОНЕНТЫ
# ======================================================================
class RMEPProfile:
    """Регуляторный профиль RMEP.

    Двухпараметрический: A (амплитуда, км/с) и alpha (безразмерный
    масштаб регуляторного поля в единицах радиуса диска r_d).
    Вклад в квадрат скорости: v_reg^2 = A^2 * x / (1 + x)^2,
    где x = r / (alpha * r_d). Форма воспроизводит типовое
    поведение кривых вращения реальных галактик.
    """
    name = "RMEP"
    param_names = ("A", "alpha")

    def v2(self, r, A, alpha, r_d):
        r_c = np.maximum(alpha * r_d, 1e-3)
        x = np.maximum(r / r_c, 1e-6)
        return A ** 2 * x / (1.0 + x) ** 2

    def v(self, r, A, alpha, r_d):
        return np.sqrt(np.maximum(self.v2(r, A, alpha, r_d), 0.0))

    def start(self, v_scale):
        return np.array([np.log10(max(v_scale, 10.0)), np.log10(0.3)])


class NFWProfile:
    """Классический профиль Наварро–Френка–Уайта (для сравнения).

    Параметры: A = масштаб скорости (км/с), alpha = концентрация c
    (масштаб r_s = r_d / alpha). Вклад: v_nfw^2 = A^2 * f(x)/x,
    x = r/r_s, f(x) = ln(1+x) - x/(1+x).
    """
    name = "NFW"
    param_names = ("A", "alpha")

    def v2(self, r, A, alpha, r_d):
        r_s = np.maximum(r_d / np.maximum(alpha, 1e-3), 1e-3)
        x = np.maximum(r / r_s, 1e-6)
        f = np.log(1.0 + x) - x / (1.0 + x)
        return A ** 2 * f / x

    def v(self, r, A, alpha, r_d):
        return np.sqrt(np.maximum(self.v2(r, A, alpha, r_d), 0.0))

    def start(self, v_scale):
        return np.array([np.log10(max(v_scale, 10.0)), np.log10(0.8)])


# ======================================================================
#  ПОДГОНКА ПО ОДНОЙ ГАЛАКТИКЕ
# ======================================================================
def fit_galaxy(r, v_obs, v_err, v_disk, profile):
    """Подгонка профиля к кривой вращения одной галактики.

    Модель: v_model^2 = v_disk^2 + v_dark^2 (учёт барионной компоненты).
    Возвращает dict: chi2 и params {A, alpha}.
    """
    r = np.asarray(r, float)
    v_obs = np.asarray(v_obs, float)
    v_err = np.maximum(np.asarray(v_err, float), 1e-3)
    v_disk = np.asarray(v_disk, float)
    r_d = max(np.median(r), 0.1)

    def resid(p):
        A, alpha = 10.0 ** p[0], 10.0 ** p[1]
        v_model = np.sqrt(v_disk ** 2 + profile.v2(r, A, alpha, r_d))
        return (v_model - v_obs) / v_err

    p0 = profile.start(np.median(v_obs))
    try:
        sol = least_squares(resid, p0, method="trf",
                            bounds=([-2, -3], [4, 3]))
        chi2 = float(np.sum(sol.fun ** 2))
        A, alpha = 10.0 ** sol.x[0], 10.0 ** sol.x[1]
    except Exception:
        chi2, A, alpha = np.inf, 0.0, 0.0

    return {"chi2": chi2, "params": {"A": A, "alpha": alpha}}


# ======================================================================
#  КОСМОЛОГИЯ
# ======================================================================
def hubble_rmep(z, H0=70.0, Om=0.315, lam_reg=0.01):
    """H(z) в RMEP. Регуляторный член модифицирует тёмную энергию:
    rho_de(z) = 1 + lam_reg * z/(1+z) (нормировка на rho_de(0))."""
    z = np.atleast_1d(np.asarray(z, float))
    rho_de = 1.0 + lam_reg * z / (1.0 + z)
    E2 = Om * (1.0 + z) ** 3 + (1.0 - Om) * rho_de
    return H0 * np.sqrt(E2)


def hubble_lcdm(z, H0=67.4, Om=0.315):
    """H(z) в LambdaCDM."""
    z = np.atleast_1d(np.asarray(z, float))
    return H0 * np.sqrt(Om * (1.0 + z) ** 3 + (1.0 - Om))


def w_eff_rmep(z, lam_reg=0.01):
    """Эффективное уравнение состояния RMEP (аналитически из rho_de)."""
    z = np.atleast_1d(np.asarray(z, float))
    denom = 1.0 + lam_reg * z / (1.0 + z)
    # (1 + w) = d ln rho_de / d ln(1+z)
    one_plus_w = lam_reg / (1.0 + z) ** 2 / denom
    return -1.0 + one_plus_w
