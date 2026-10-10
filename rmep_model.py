"""
RMEP vs NFW: Определения моделей тёмной материи.

Модели:
  NFW  — классический профиль Наварро-Френка-Уайта (2 параметра: V200, rs)
  RMEP — Регуляторная модель энергии пустоты (2 формальных параметра: alpha, A)
         Фактически однопараметрическая: A определяется через alpha.

Автор: [заполнить]
Лицензия: MIT
"""

import numpy as np
from scipy.optimize import curve_fit

# ============================================================
# Константы
# ============================================================
G = 4.30091e-6  # (km/s)^2 * kpc / M_sun — гравитационная постоянная

# ============================================================
# Модель NFW
# ============================================================

def nfw_density(r, V200, rs):
    """Профиль плотности NFW.
    
    Параметры:
        r   — радиус (kpc)
        V200 — круговая скорость на радиусе r200 (km/s)
        rs   — масштабный радиус (kpc)
    
    Возвращает плотность (M_sun / kpc^3).
    """
    M200 = V200**3 / (10 * G)
    r200 = V200 / 100.0  # приближение (H = 100 km/s/Mpc)
    c = r200 / rs
    x = r / rs
    rho0 = M200 / (4 * np.pi * rs**3 * (np.log(1 + c) - c / (1 + c)))
    return rho0 / (x * (1 + x)**2)


def nfw_mass(r, V200, rs):
    """Масса внутри радиуса r для профиля NFW (M_sun)."""
    x = r / rs
    rho0 = V200**3 / (10 * G) / (4 * np.pi * rs**3)
    # Нормировка
    r200 = V200 / 100.0
    c = r200 / rs
    rho0 /= (np.log(1 + c) - c / (1 + c))
    return 4 * np.pi * rho0 * rs**3 * (np.log(1 + x) - x / (1 + x))


def nfw_velocity(r, V200, rs):
    """Круговая скорость для профиля NFW (km/s)."""
    x = np.asarray(r) / rs
    mass = nfw_mass(np.asarray(r), V200, rs)
    return np.sqrt(G * mass / np.maximum(r, 1e-10))


# ============================================================
# Модель RMEP
# ============================================================

def rmep_density(r, alpha, A):
    """Профиль плотности RMEP.
    
    Параметры:
        r     — радиус (kpc)
        alpha — масштаб регуляции (kpc)
        A     — нормировка (M_sun / kpc^3)
    
    Возвращает плотность (M_sun / kpc^3).
    """
    x = np.asarray(r) / alpha
    return A * np.exp(-x) / (x**2 + 1e-30)


def rmep_mass(r, alpha, A):
    """Масса внутри радиуса r для профиля RMEP (M_sun).
    
    Численное интегрирование: M(r) = 4*pi * integral(rho * r'^2 dr').
    """
    r_arr = np.atleast_1d(np.asarray(r, dtype=float))
    masses = []
    for ri in r_arr:
        if ri <= 0:
            masses.append(0.0)
            continue
        # Численное интегрирование
        n = max(1000, int(ri * 100))
        rr = np.linspace(1e-6, ri, n)
        rho = rmep_density(rr, alpha, A)
        integrand = rho * rr**2
        masses.append(4 * np.pi * np.trapz(integrand, rr))
    return np.array(masses)


def rmep_velocity(r, alpha, A):
    """Круговая скорость для профиля RMEP (km/s)."""
    r_arr = np.atleast_1d(np.asarray(r, dtype=float))
    masses = rmep_mass(r_arr, alpha, A)
    v = np.sqrt(G * masses / np.maximum(r_arr, 1e-10))
    if np.isscalar(r):
        return float(v)
    return v


# ============================================================
# Подгонка моделей
# ============================================================

def fit_nfw(r, v_obs, v_err):
    """Подгонка NFW к наблюдаемой кривой вращения.
    
    Возвращает: (V200, rs, chi2)
    """
    def model(r, V200, rs):
        return nfw_velocity(r, V200, rs)
    
    # Начальные приближения
    vmax_obs = np.max(v_obs)
    p0 = [vmax_obs, np.median(r)]
    bounds = ([10, 0.01], [2000, 500])
    
    try:
        popt, _ = curve_fit(model, r, v_obs, p0=p0, sigma=v_err, 
                           bounds=bounds, maxfev=10000)
        v_model = nfw_velocity(r, *popt)
        chi2 = np.sum(((v_obs - v_model) / v_err)**2)
        return popt[0], popt[1], chi2
    except Exception:
        return np.nan, np.nan, np.inf


def fit_rmep(r, v_obs, v_err):
    """Подгонка RMEP к наблюдаемой кривой вращения.
    
    Возвращает: (alpha, A, chi2)
    """
    def model(r, alpha, A):
        return rmep_velocity(r, alpha, A)
    
    # Начальные приближения
    vmax_obs = np.max(v_obs)
    p0 = [np.median(r), vmax_obs**2 * np.median(r) / (4 * np.pi * G)]
    bounds = ([0.01, 1e3], [100, 1e10])
    
    try:
        popt, _ = curve_fit(model, r, v_obs, p0=p0, sigma=v_err,
                           bounds=bounds, maxfev=10000)
        v_model = rmep_velocity(r, *popt)
        chi2 = np.sum(((v_obs - v_model) / v_err)**2)
        return popt[0], popt[1], chi2
    except Exception:
        return np.nan, np.nan, np.inf


if __name__ == '__main__':
    # Быстрый тест
    r = np.linspace(0.5, 30, 50)
    v = nfw_velocity(r, 200, 10)
    print(f"NFW test: v(1 kpc) = {v[0]:.1f}, v(30 kpc) = {v[-1]:.1f}")
    
    v2 = rmep_velocity(r, 3.0, 2e5)
    print(f"RMEP test: v(1 kpc) = {v2[0]:.1f}, v(30 kpc) = {v2[-1]:.1f}")
