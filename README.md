# RMEP-dark-matter

**RMEP** (Regressive Model of Empty-space Energy) — an alternative model of the
dark component: a regulatory field replaces both dark matter in galaxies and
dark energy in cosmology. The code is prepared for peer review.

---

**RMEP** (Регуляторная модель энергии пустоты) — альтернативная модель тёмной
компоненты: регуляторное поле заменяет и тёмную материю в галактиках,
и тёмную энергию в космологии. Код подготовлен для рецензирования.

## Contents / Содержание

| File / Файл | Purpose / Назначение |
|---|---|
| `rmep_model.py` | Core: RMEP and NFW profiles, rotation-curve fitting, H(z) and w(z). Ядро: профили RMEP и NFW, подгонка кривых вращения, H(z) и w(z) |
| `fit_galaxies.py` | Fitting pipeline for 177 galaxies; produces `results.csv`. Пайплайн подгонки по 177 галактикам; создаёт results.csv |
| `plot_results.py` | Plots: A ~ alpha^k, chi2 distributions, RMEP/NFW. Графики: связь A ~ alpha^k, распределения chi2, сравнение RMEP/NFW |
| `cosmology_test.py` | Tests: H(z), SNe Ia, BAO, w(z) vs CMB distance priors (R, l_A, r_s) LambdaCDM. Тесты: H(z), сверхновые, BAO, w(z) против LambdaCDM |
| `nbody_skeleton.py` | N-body skeleton, leapfrog, energy check. N-body скелет, leapfrog, проверка энергии |
| `requirements.txt` | Dependencies. Зависимости (numpy, scipy, matplotlib, pandas) |

## Installation and usage / Установка и запуск

    pip install -r requirements.txt

    # Demo mode (synthetic sample of 177 galaxies) / Демо-режим (синтетическая выборка из 177 галактик):
    python fit_galaxies.py
    python plot_results.py
    python cosmology_test.py
    python nbody_skeleton.py

    # Real SPARC data (directory with *_rotmod.dat files) / Реальные данные SPARC (каталог с файлами *_rotmod.dat):
    python fit_galaxies.py sparc/

The SPARC data catalog is available on the official SPARC project website
(Lelli et al. 2016).
Каталог данных SPARC доступен на официальном сайте проекта SPARC (Lelli et al. 2016).

## Results / Результаты

**Rotation curves (demo mode):** mean chi2 of RMEP ~61 versus ~94 for NFW.
**Кривые вращения (демо-режим):** средний chi2 RMEP ~61 против ~94 у NFW.

**Cosmology (lam_reg = 0.01):** deviations from LambdaCDM within 0.1% for BAO
and 0.001 mag for SNe Ia; w(z) approaches -1 as z grows.
**Космология (lam_reg = 0.01):** отклонения от LambdaCDM в пределах 0,1% по BAO
и 0,001 mag по сверхновым; w(z) стремится к -1 с ростом z.

**N-body:** total-energy drift over 200 leapfrog steps below 0.1%.
**N-body:** дрейф полной энергии за 200 шагов leapfrog менее 0,1%.

## Important note / Важное замечание

The demo mode uses **synthetic** data (reproducible generation, seed = 42)
and verifies the pipeline only. Physical conclusions require running the code
on the real SPARC catalog.
Демо-режим использует **синтетические** данные (воспроизводимая генерация,
seed = 42) и проверяет только работоспособность конвейера. Физические выводы
требуют запуска на реальном каталоге SPARC.

## Status / Статус

Submitted for peer review. Contact: see the repository owner's profile.
Код передан на рецензирование. Контакты: см. профиль владельца репозитория.
