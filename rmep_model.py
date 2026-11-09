# -*- coding: utf-8 -*-
"""
rmep_model.py — ядро RMEP: Регуляторная модель энергии пустоты (версия 5.0).

Содержит:
  * RMEPProfile — канонический регуляторный профиль: экспоненциальная сфера
                  rho(R) = rho0*exp(-R/alpha), плоское ядро в центре (разд. 6.8.1)
  * NFWProfile  — классический профиль Наварро-Френка-Уайта (эталон сравнения)
  * fit_galaxy  — подгонка профиля к кривой вращения (chi2, least_squares)
  * hubble_rmep, hubble_lcdm — H(z) для RMEP и LambdaCDM
  * w_eff_rmep  — эффективное уравнение состояния RMEP

Все величины: R в кпк, v в км/с, H0 в км/с/Мпк, плотность в Msun/кпк^3.
"""

import numpy as np
from scipy.optimize import least_squares

G_KPC = 4.30091e-6   # кпк * (км/с)^2 / Msun
H0_NFW = 70.0        # км/с/Мпк — конвенция для r200 (200*rho_crit)
RHO_CRIT = 3.0 * (H0_NFW / 1000.0) ** 2 / (8.0 * np.pi * G_KPC)  # Msun/кпк^3 ≈ 136


# ======================================================================
#  ПРОФИЛИ ТЁМНОЙ КОМПОНЕНТЫ
# ======================================================================
class RMEPProfile:
    """Канонический регуляторный профиль RMEP: экспоненциальная сфера.

        rho(R) = rho0 * exp(-R/alpha)

    Плотность конечна в центре (плоское ядро) — ключевой результат модели:
    cuspa NFW не возникает, никакой дополнительной физики не требуется.

    Параметризация через (V0, alpha):
        V0^2 = 8*pi*G*rho0*alpha^2   <=>   rho0 = V0^2 / (8*pi*G*alpha^2).

    Вклад в квадрат скорости (аналитически, точно = GM(R)/R):
        v2(R) = V0^2 / x * [1 - (1 + x + x^2/2)*exp(-x)],   x = R/alpha.

    Свойства: пик кривой вращения v_max = 0.4407*V0 на R = 3.38*alpha;
    в центре v ~ R (твёрдое вращение); вдали v^2 ~ V0^2/x (кеплерианство).
    """
    name = "RMEP"
    param_names = ("V0", "alpha")

    def rho0(self, V0, alpha):
        """Центральная плотность (Msun/кпк^3)."""
        return V0 ** 2 / (8.0 * np.pi * G_KPC * alpha ** 2)

    def density(self, r, V0, alpha):
        """Плотность профиля (Msun/кпк^3)."""
        return self.rho0(V0, alpha) * np.exp(-np.asarray(r, float) / alpha)

    def mass(self, r, V0, alpha):
        """Масса внутри радиуса r (Msun), аналитически."""
        x = np.maximum(np.asarray(r, float) / max(alpha, 1e-4), 0.0)
        return 8.0 * np.pi * self.rho0(V0, alpha) * alpha ** 3 * \
            (1.0 - np.exp(-x) * (1.0 + x + x * x / 2.0))

    def v2(self, r, V0, alpha, r_d=None):
        """Вклад профиля в квадрат круговой скорости, (км/с)^2. r_d не нужен
        (оставлен в сигнатуре для совместимости с пайплайном)."""
        x = np.maximum(np.asarray(r, float) / max(alpha, 1e-4), 1e-12)
        return V0 ** 2 / x * (1.0 - (1.0 + x + x * x / 2.0) * np.exp(-x))

    def v(self, r, V0, alpha, r_d=None):
        return np.sqrt(np.maximum(self.v2(r, V0, alpha), 0.0))

    def potential(self, r, V0, alpha):
        """Гравитационный потенциал профиля (км/с)^2, Phi(inf) = 0.

        Phi(R) = -4*pi*G*rho0*alpha^2 * [ 2*F(x)/x + (1+x)*exp(-x) ],
        F(x) = 1 - (1 + x + x^2/2)*exp(-x).  Проверено: -dPhi/dR = GM(R)/R^2.
        """
        x = np.maximum(np.asarray(r, float) / max(alpha, 1e-4), 1e-12)
        Fx = 1.0 - np.exp(-x) * (1.0 + x + x * x / 2.0)
        return -4.0 * np.pi * G_KPC * self.rho0(V0, alpha) * alpha ** 2 * \
            (2.0 * Fx / x + (1.0 + x) * np.exp(-x))

    @staticmethod
    def vmax_theory(V0):
        """Теоретический пик кривой вращения: v_max = 0.4407*V0 (на R = 3.38*alpha)."""
        return 0.4407 * V0

    def start(self, v_scale):
        """Начальное приближение в log10-параметрах.
        alpha стартует из канонической регрессии alpha ~ 5 kpc*(Vmax/100)^1.89."""
        v0_0 = max(2.3 * v_scale, 5.0)
        alpha_0 = float(np.clip(5.0 * (v_scale / 100.0) ** 1.89, 0.2, 100.0))
        return np.array([np.log10(v0_0), np.log10(alpha_0)])

    def bounds(self):
        """Границы поиска в log10: V0 in [1, 1e4] км/с, alpha in [0.05, 500] кпк."""
        return np.array([0.0, -1.3]), np.array([4.0, 2.7])


class NFWProfile:
    """Классический профиль Наварро-Френка-Уайта (эталон сравнения).

    Параметры: V200 (км/с) и концентрация c = r200/r_s.
    r200 = 100*V200/H0 (кпк) — из M200 = (4pi/3)*200*rho_crit*r200^3.
    Вклад в квадрат скорости:
        v2(R) = V200^2 * f(x') / (x'*f(c)),  x' = R/r200,
        f(t) = ln(1+t) - t/(1+t).
    """
    name = "NFW"
    param_names = ("V200", "c")

    def r200(self, V200):
        """Радиус вириализации (кпк)."""
        return 100.0 * V200 / H0_NFW

    def mass(self, r, V200, c):
        """Масса внутри радиуса r (Msun), аналитически."""
        r2 = self.r200(V200)
        x = np.maximum(np.asarray(r, float) / r2, 1e-12)
        f = np.log1p(x) - x / (1.0 + x)
        fc = np.log1p(c) - c / (1.0 + c)
        return V200 ** 2 * r2 / G_KPC * f / fc

    def v2(self, r, V200, c, r_d=None):
        r2 = self.r200(V200)
        x = np.maximum(np.asarray(r, float) / r2, 1e-12)
        f = np.log1p(x) - x / (1.0 + x)
        fc = np.log1p(c) - c / (1.0 + c)
        return V200 ** 2 * f / (x * fc)

    def v(self, r, V200, c, r_d=None):
        return np.sqrt(np.maximum(self.v2(r, V200, c), 0.0))

    def start(self, v_scale):
        return np.array([np.log10(max(v_scale, 10.0)), np.log10(10.0)])

    def bounds(self):
        """V200 in [5, 2000] км/с, c in [2, 100]."""
        return np.array([0.7, 0.3]), np.array([3.3, 2.0])


# ======================================================================
#  ПОДГОНКА ПО ОДНОЙ ГАЛАКТИКЕ
# ======================================================================
def fit_galaxy(r, v_obs, v_err, v_disk, profile):
    """Подгонка профиля к кривой вращения одной галактики.

    Модель: v_model^2 = v_disk^2 + v_dark^2 (барионы + тёмная компонента).
    Параметры ищутся в log10 (устойчивость масштабов).

    Возвращает dict:
      chi2       — качество подгонки
      params     — {имя: значение} по profile.param_names
      degenerate — True, если решение упёрлось в границу поиска
                   (цензурируется: не участвует в статистике выборки)
    """
    r = np.asarray(r, float)
    v_obs = np.asarray(v_obs, float)
    v_err = np.maximum(np.asarray(v_err, float), 1e-3)
    v_disk = np.asarray(v_disk, float)

    lo, hi = profile.bounds()

    def resid(p):
        q = 10.0 ** p
        v_model = np.sqrt(v_disk ** 2 + profile.v2(r, *q))
        return (v_model - v_obs) / v_err

    p0 = np.clip(profile.start(np.median(v_obs)), lo + 1e-3, hi - 1e-3)
    try:
        sol = least_squares(resid, p0, bounds=(lo, hi), method="trf")
        chi2 = float(np.sum(sol.fun ** 2))
        params = dict(zip(profile.param_names, 10.0 ** sol.x))
        at_bound = bool(np.any(sol.x <= lo + 1e-2) or np.any(sol.x >= hi - 1e-2))
    except Exception:
        chi2 = np.inf
        params = {n: np.nan for n in profile.param_names}
        at_bound = True

    return {"chi2": chi2, "params": params, "degenerate": at_bound}


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
    """Эффективное уравнение состояния RMEP (аналитически из rho_de).

    (1 + w) = d ln rho_de / d ln(1+z) = lam_reg / ((1+z) * rho_de).
    """
    z = np.atleast_1d(np.asarray(z, float))
    denom = 1.0 + lam_reg * z / (1.0 + z)
    one_plus_w = lam_reg / (1.0 + z) / denom
    return -1.0 + one_plus_w


# ======================================================================
#  САМОПРОВЕРКА
# ======================================================================
if __name__ == "__main__":
    from scipy.integrate import quad

    # 1) аналитика vs прямой интеграл массы и скорости
    alpha, V0 = 2.5, 120.0
    prof = RMEPProfile()
    rho0 = prof.rho0(V0, alpha)
    M_num = 4 * np.pi * quad(lambda t: rho0 * np.exp(-t / alpha) * t * t, 0, 5 * alpha)[0]
    print(f"M(5*alpha): аналитика {prof.mass(5*alpha, V0, alpha):.6e}, "
          f"интеграл {M_num:.6e}, отн.разн. "
          f"{abs(prof.mass(5*alpha, V0, alpha)/M_num - 1):.1e}")
    v2_num = G_KPC * M_num / (5 * alpha)
    print(f"v^2(5*alpha): аналитика {prof.v2(5*alpha, V0, alpha):.6e}, "
          f"GM/r {v2_num:.6e}, отн.разн. {abs(prof.v2(5*alpha, V0, alpha)/v2_num - 1):.1e}")

    # 2) синтетическая галактика: восстановление параметров
    rng = np.random.default_rng(7)
    r = np.linspace(0.5, 15.0, 25)
    V0_true, alpha_true = 180.0, 4.0
    v_disk_true = 120.0 * np.tanh(r / 3.0)
    v_true = np.sqrt(v_disk_true ** 2 + prof.v2(r, V0_true, alpha_true))
    v_obs = v_true + rng.normal(0, 4.0, r.shape)
    fit = fit_galaxy(r, v_obs, np.full_like(r, 4.0), v_disk_true, prof)
    print(f"синтетика: true V0={V0_true}, alpha={alpha_true} -> "
          f"V0={fit['params']['V0']:.1f}, alpha={fit['params']['alpha']:.2f}, "
          f"chi2={fit['chi2']:.1f}, degenerate={fit['degenerate']}")

    # 3) NFW тоже подгоняется
    nfw = NFWProfile()
    v_nfw = np.sqrt(v_disk_true ** 2 + nfw.v2(r, 150.0, 8.0))
    fit_n = fit_galaxy(r, v_nfw + rng.normal(0, 4.0, r.shape),
                       np.full_like(r, 4.0), v_disk_true, nfw)
    print(f"NFW: true V200=150, c=8 -> V200={fit_n['params']['V200']:.1f}, "
          f"c={fit_n['params']['c']:.2f}, chi2={fit_n['chi2']:.1f}")

