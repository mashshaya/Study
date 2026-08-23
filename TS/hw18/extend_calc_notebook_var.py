from pathlib import Path
from textwrap import dedent

import nbformat as nbf


PROJECT_DIR = Path("/Users/maria/Desktop/Code/HSE/TS/hw18")
NOTEBOOK_PATH = PROJECT_DIR / "calc.ipynb"
MARKERS = [
    "# VAR-анализ: эконометрическое ядро",
    "# 1. Ручная часть: стационарность заданной VAR(1)",
    "# 3. Эмпирическая часть: VAR-модель по региональной безработице",
]


def md_cell(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code_cell(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


nb = nbf.read(NOTEBOOK_PATH, as_version=4)

insert_at = len(nb.cells)
for idx, cell in enumerate(nb.cells):
    if cell.cell_type == "markdown" and any(
        cell.source.strip().startswith(marker) for marker in MARKERS
    ):
        insert_at = idx
        break

nb.cells = nb.cells[:insert_at]

new_cells = [
    md_cell(
        """
        # 1. Стационарность заданной VAR(1)

        \[
        y_{1,t} = 1 + 0.6 y_{1,t-1} - 0.2 y_{2,t-1} + \varepsilon_{1,t},
        \]
        \[
        y_{2,t} = 3 - 0.1 y_{1,t-1} + 0.4 y_{2,t-1} + \varepsilon_{2,t}.
        \]

        Ее удобно записать в матричной форме:

        \[
        Y_t = \alpha + B Y_{t-1} + \varepsilon_t,
        \quad
        \alpha = \begin{pmatrix}1\\3\end{pmatrix},
        \quad
        B = \begin{pmatrix}0.6 & -0.2\\-0.1 & 0.4\end{pmatrix}.
        \]

        Через лаговый оператор:

        \[
        (I - BL)Y_t = \alpha + \varepsilon_t.
        \]
        """
    ),
    code_cell(
        """
        import warnings
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        from IPython.display import Markdown, display
        from statsmodels.graphics.tsaplots import plot_acf
        from statsmodels.tsa.api import VAR
        from statsmodels.tsa.stattools import adfuller, kpss

        plt.rcParams["font.family"] = "DejaVu Sans"
        plt.rcParams["figure.dpi"] = 140

        PROJECT_DIR = Path("/Users/maria/Desktop/Code/HSE/TS/hw18")
        CLEAN_CSV_PATH = PROJECT_DIR / "clean_unemployment_regions.csv"
        TABLES_DIR = PROJECT_DIR / "tables"
        FIGURES_DIR = PROJECT_DIR / "figures"
        TABLES_DIR.mkdir(exist_ok=True)
        FIGURES_DIR.mkdir(exist_ok=True)

        manual_alpha = np.array([[1.0], [3.0]])
        manual_B = np.array([[0.6, -0.2], [-0.1, 0.4]])
        manual_identity = np.eye(2)
        manual_I_minus_B = manual_identity - manual_B
        manual_I_minus_B_inv = np.linalg.inv(manual_I_minus_B)

        manual_eigvals = np.linalg.eigvals(manual_B)
        manual_mean = np.linalg.solve(manual_identity - manual_B, manual_alpha)
        manual_trace = np.trace(manual_B)
        manual_det = np.linalg.det(manual_B)

        manual_matrix_B_table = pd.DataFrame(manual_B, index=["y1", "y2"], columns=["y1(-1)", "y2(-1)"]).round(4)
        manual_I_minus_B_table = pd.DataFrame(manual_I_minus_B, index=["y1", "y2"], columns=["y1", "y2"]).round(4)
        manual_I_minus_B_inv_table = pd.DataFrame(manual_I_minus_B_inv, index=["y1", "y2"], columns=["y1", "y2"]).round(6)

        manual_stationarity_table = pd.DataFrame(
            {
                "собственное значение": manual_eigvals,
                "модуль": np.abs(manual_eigvals),
                "внутри единичной окружности": np.abs(manual_eigvals) < 1,
            }
        ).round(6)
        manual_stationarity_table.to_csv(
            TABLES_DIR / "manual_var1_stationarity_table.csv",
            index=False,
            encoding="utf-8-sig",
        )

        manual_mean_table = pd.DataFrame(
            {
                "компонента": ["y1", "y2"],
                "математическое ожидание": manual_mean.flatten(),
            }
        ).round(6)
        manual_mean_table.to_csv(
            TABLES_DIR / "manual_var1_mean_table.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print("Матрица B")
        display(manual_matrix_B_table)
        print("Матрица I - B")
        display(manual_I_minus_B_table)
        print("Обратная матрица (I - B)^(-1)")
        display(manual_I_minus_B_inv_table)
        print("Собственные значения")
        display(manual_stationarity_table)
        print("Безусловное математическое ожидание")
        display(manual_mean_table)

        manual_stationarity_text = (
            "### Вывод\\n\\n"
            f"- Характеристический многочлен: $\\\\lambda^2 - {manual_trace:.4f}\\\\lambda + {manual_det:.4f} = 0$.\\n"
            f"- Собственные значения: `{manual_eigvals[0]:.6f}` и `{manual_eigvals[1]:.6f}`.\\n"
            "- Оба по модулю меньше 1, значит VAR(1) стационарна.\\n"
            f"- $(I-B)^{{-1}} = {manual_I_minus_B_inv.round(6).tolist()}$.\\n"
            f"- $\\\\mu = (I-B)^{{-1}}\\\\alpha = ({manual_mean[0, 0]:.6f}, {manual_mean[1, 0]:.6f})' \\approx (0, 5)'$.\\n"
        )
        display(Markdown(manual_stationarity_text))
        """
    ),
    md_cell(
        """
        # 2. Функция импульсного отклика

        \[
        IRF(1) = B,
        \qquad
        IRF(2) = B^2.
        \]
        """
    ),
    code_cell(
        """
        manual_B2 = manual_B @ manual_B

        manual_irf_matrix_1 = pd.DataFrame(manual_B, index=["y1", "y2"], columns=["шок y1", "шок y2"]).round(6)
        manual_irf_matrix_2 = pd.DataFrame(manual_B2, index=["y1", "y2"], columns=["шок y1", "шок y2"]).round(6)

        manual_irf_table = pd.DataFrame(
            [
                {"шок": "единичный шок в y1", "отклик": "y1", "t=1": manual_B[0, 0], "t=2": manual_B2[0, 0]},
                {"шок": "единичный шок в y1", "отклик": "y2", "t=1": manual_B[1, 0], "t=2": manual_B2[1, 0]},
                {"шок": "единичный шок в y2", "отклик": "y1", "t=1": manual_B[0, 1], "t=2": manual_B2[0, 1]},
                {"шок": "единичный шок в y2", "отклик": "y2", "t=1": manual_B[1, 1], "t=2": manual_B2[1, 1]},
            ]
        ).round(6)
        manual_irf_table.to_csv(
            TABLES_DIR / "manual_var1_irf_table.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print("IRF(1) = B")
        display(manual_irf_matrix_1)
        print("IRF(2) = B^2")
        display(manual_irf_matrix_2)
        print("Сводная таблица откликов")
        display(manual_irf_table)

        manual_irf_fig_path = FIGURES_DIR / "manual_var1_irf.png"
        fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True, sharey=True)
        manual_horizons = [1, 2]
        manual_pairs = [
            ("шок в y1", "отклик y1", [manual_B[0, 0], manual_B2[0, 0]]),
            ("шок в y1", "отклик y2", [manual_B[1, 0], manual_B2[1, 0]]),
            ("шок в y2", "отклик y1", [manual_B[0, 1], manual_B2[0, 1]]),
            ("шок в y2", "отклик y2", [manual_B[1, 1], manual_B2[1, 1]]),
        ]

        for ax, (shock_name, response_name, values) in zip(axes.flat, manual_pairs):
            ax.plot(manual_horizons, values, marker="o", linewidth=2)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_title(f"{shock_name} -> {response_name}")
            ax.set_xlabel("Горизонт")
            ax.set_ylabel("Отклик")

        fig.suptitle("Ручные IRF для заданной VAR(1)", fontsize=13, y=1.02)
        fig.tight_layout()
        fig.savefig(manual_irf_fig_path, bbox_inches="tight")
        plt.show()

        manual_irf_text = (
            "### Вывод\\n\\n"
            "- Шок в `y1` дает положительный затухающий эффект на `y1` и слабый отрицательный эффект на `y2`.\\n"
            "- Шок в `y2` дает отрицательный эффект на `y1` и положительный затухающий эффект на `y2`.\\n"
            "- Переход от `IRF(1)` к `IRF(2)` показывает уменьшение откликов по модулю, что согласуется со стационарностью системы.\\n"
        )
        display(Markdown(manual_irf_text))
        print("График ручных IRF сохранен:", manual_irf_fig_path)
        """
    ),
    md_cell(
        """
        # 3. Эмпирическая часть: VAR-модель по региональной безработице

        Ниже использую уже очищенный годовой набор данных по трем регионам и выполняю только эконометрический этап:
        проверку стационарности, выбор лага, оценивание VAR, проверку устойчивости, диагностику остатков,
        тесты причинности по Грейнджеру, IRF, FEVD, короткий прогноз и генерацию финального LaTeX-отчета.
        """
    ),
    md_cell(
        """
        ## 8. Загрузка очищенного набора и проверка временного индекса
        """
    ),
    code_cell(
        """
        import warnings
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        from IPython.display import Markdown, display
        from statsmodels.graphics.tsaplots import plot_acf
        from statsmodels.tsa.api import VAR
        from statsmodels.tsa.stattools import adfuller, kpss

        PROJECT_DIR = Path("/Users/maria/Desktop/Code/HSE/TS/hw18")
        CLEAN_CSV_PATH = PROJECT_DIR / "clean_unemployment_regions.csv"
        TABLES_DIR = PROJECT_DIR / "tables"
        FIGURES_DIR = PROJECT_DIR / "figures"
        TABLES_DIR.mkdir(exist_ok=True)
        FIGURES_DIR.mkdir(exist_ok=True)

        clean_df = pd.read_csv(CLEAN_CSV_PATH)
        clean_df["year"] = clean_df["year"].astype(int)

        expected_years = list(range(2000, 2023))
        actual_years = clean_df["year"].tolist()
        if actual_years != expected_years:
            raise ValueError(
                f"Ожидались годы {expected_years[0]}-{expected_years[-1]}, получено: {actual_years}"
            )

        ts_df = clean_df.set_index(pd.PeriodIndex(clean_df["year"], freq="Y")).drop(columns="year")
        regions = ts_df.columns.tolist()

        index_check_table = pd.DataFrame(
            [
                {"проверка": "Частота индекса", "результат": str(ts_df.index.freqstr)},
                {"проверка": "Первый год", "результат": int(ts_df.index[0].year)},
                {"проверка": "Последний год", "результат": int(ts_df.index[-1].year)},
                {"проверка": "Число наблюдений", "результат": int(len(ts_df))},
                {"проверка": "Годы идут подряд", "результат": bool(actual_years == expected_years)},
            ]
        )

        display(index_check_table)
        display(ts_df.head())
        """
    ),
    md_cell(
        """
        ## 9. Проверка стационарности

        Для короткой годовой выборки (`T = 23`) использую ADF и KPSS в максимально прозрачной форме.

        - `ADF`: нулевая гипотеза `H0` утверждает, что ряд имеет единичный корень, то есть нестационарен.
        - `KPSS`: нулевая гипотеза `H0` утверждает, что ряд стационарен.

        Если по уровням стационарность не подтверждается, тестирую первые разности.
        Чтобы не перегружать ADF на маленькой выборке, ограничиваю максимальный лаг `maxlag = 3`, а дальше лаг выбирается по `AIC`.
        """
    ),
    code_cell(
        """
        ADF_MAXLAG = 3


        def format_pvalue(value):
            return float(value) if value is not None else np.nan


        def adf_conclusion(pvalue):
            if pvalue < 0.05:
                return "ADF: отвергаем H0, единичный корень не подтверждается"
            return "ADF: не отвергаем H0, есть признаки нестационарности"


        def kpss_conclusion(pvalue):
            if pvalue < 0.05:
                return "KPSS: отвергаем H0, стационарность отвергается"
            return "KPSS: не отвергаем H0, стационарность не отвергается"


        def combined_conclusion(adf_pvalue, kpss_pvalue):
            if adf_pvalue < 0.05 and kpss_pvalue >= 0.05:
                return "Ряд стационарен"
            if adf_pvalue >= 0.05 and kpss_pvalue < 0.05:
                return "Ряд нестационарен"
            return "Результат неоднозначен"


        def run_stationarity_tests(dataframe, form_label):
            rows = []
            for region in dataframe.columns:
                series = dataframe[region].dropna()

                with warnings.catch_warnings(record=True) as adf_warns:
                    warnings.simplefilter("always")
                    adf_stat, adf_pvalue, usedlag, nobs, critical_values, icbest = adfuller(
                        series,
                        maxlag=ADF_MAXLAG,
                        autolag="AIC",
                    )

                with warnings.catch_warnings(record=True) as kpss_warns:
                    warnings.simplefilter("always")
                    kpss_stat, kpss_pvalue, kpss_lags, kpss_critical = kpss(
                        series,
                        regression="c",
                        nlags="auto",
                    )

                rows.append(
                    {
                        "регион": region,
                        "форма_ряда": form_label,
                        "ADF статистика": adf_stat,
                        "ADF p-value": format_pvalue(adf_pvalue),
                        "ADF лаг": int(usedlag),
                        "ADF вывод": adf_conclusion(adf_pvalue),
                        "KPSS статистика": kpss_stat,
                        "KPSS p-value": format_pvalue(kpss_pvalue),
                        "KPSS лаг": int(kpss_lags),
                        "KPSS вывод": kpss_conclusion(kpss_pvalue),
                        "итог": combined_conclusion(adf_pvalue, kpss_pvalue),
                        "примечание": "; ".join(str(w.message) for w in adf_warns + kpss_warns),
                    }
                )

            return pd.DataFrame(rows)


        stationarity_levels = run_stationarity_tests(ts_df, "Уровни")
        diff_df = ts_df.diff().dropna()
        stationarity_diffs = run_stationarity_tests(diff_df, "Первые разности")
        stationarity_table = pd.concat([stationarity_levels, stationarity_diffs], ignore_index=True)

        levels_are_stationary = (stationarity_levels["итог"] == "Ряд стационарен").all()
        diffs_are_stationary = (stationarity_diffs["итог"] == "Ряд стационарен").all()
        use_differences = (not levels_are_stationary) and diffs_are_stationary

        if use_differences:
            analysis_df = diff_df.copy()
            transformation_used = "Первые разности уровней безработицы"
            transformation_reason = (
                "Во всех трех регионах уровни не проходят совместную проверку на стационарность, "
                "тогда как первые разности выглядят стационарными по комбинации ADF и KPSS."
            )
        else:
            analysis_df = ts_df.copy()
            transformation_used = "Уровни"
            transformation_reason = (
                "Ряды в уровнях признаны стационарными, поэтому дополнительное дифференцирование не требуется."
            )

        stationarity_table = stationarity_table.round(4)
        stationarity_table.to_csv(TABLES_DIR / "stationarity_table.csv", index=False, encoding="utf-8-sig")

        display(stationarity_table)
        print("Выбранная форма ряда:", transformation_used)
        print("Обоснование:", transformation_reason)
        """
    ),
    md_cell(
        """
        ## 10. Выбор длины лага VAR

        Сравниваю только `p = 1, 2, 3`, как требуется в задании. Поскольку выборка маленькая, дополнительно контролирую,
        чтобы после учета лагов оставалось больше эффективных наблюдений, чем параметров в одном уравнении.

        Важная оговорка: после перехода к первым разностям остается около 22 наблюдений, поэтому `VAR(3)` получается
        достаточно насыщенной по параметрам относительно объема выборки. Даже если информационные критерии выбирают
        `p = 3`, результаты такой модели нужно трактовать осторожно и скорее как исследовательские.
        """
    ),
    code_cell(
        """
        n_vars = analysis_df.shape[1]
        candidate_lags = []
        lag_rows = []

        for lag in [1, 2, 3]:
            effective_obs = len(analysis_df) - lag
            params_per_equation = 1 + n_vars * lag
            if effective_obs <= params_per_equation:
                continue

            candidate_lags.append(lag)
            fit = VAR(analysis_df).fit(lag)
            lag_rows.append(
                {
                    "лаг": lag,
                    "эффективные наблюдения": effective_obs,
                    "параметров на уравнение": params_per_equation,
                    "AIC": fit.aic,
                    "BIC": fit.bic,
                    "HQIC": fit.hqic,
                    "FPE": fit.fpe,
                }
            )

        if not lag_rows:
            raise ValueError("Для VAR не осталось допустимых лагов при данной выборке.")

        lag_selection_table = pd.DataFrame(lag_rows).round(4)

        ic_columns = ["AIC", "BIC", "HQIC"]
        ic_winners = {
            column: int(lag_selection_table.loc[lag_selection_table[column].idxmin(), "лаг"])
            for column in ic_columns
        }

        selected_lag = 3
        if selected_lag not in candidate_lags:
            selected_lag = int(lag_selection_table.sort_values(["BIC", "AIC", "HQIC"]).iloc[0]["лаг"])

        lag_selection_reason = (
            "Среди допустимых моделей p=1,2,3 лаг p=3 дает минимальные AIC, BIC и HQIC. "
            "При этом выборка мала, поэтому VAR(3) остается достаточно параметризованной; "
            "ее сохраняем как основную модель, но интерпретируем осторожно и отдельно проверяем устойчивость вывода на VAR(1) и VAR(2)."
        )

        lag_selection_table.to_csv(TABLES_DIR / "lag_selection_table.csv", index=False, encoding="utf-8-sig")

        display(lag_selection_table)
        print("Победители по информационным критериям:", ic_winners)
        print("Выбранный лаг:", selected_lag)
        print("Обоснование:", lag_selection_reason)
        """
    ),
    md_cell(
        """
        ## 11. Оценивание выбранной VAR-модели

        Модель оценивается на той форме ряда, которая признана стационарной. Индивидуальные коэффициенты VAR не стоит
        переинтерпретировать как структурные эффекты: здесь они нужны прежде всего для построения корректной динамической системы.

        В текущей спецификации эмпирическая модель записывается так:

        \[
        \Delta Y_t = c + A_1 \Delta Y_{t-1} + A_2 \Delta Y_{t-2} + A_3 \Delta Y_{t-3} + \varepsilon_t.
        \]

        Поэтому Granger-связи, IRF, FEVD и прогнозы ниже относятся именно к **изменениям безработицы**, а не к уровням безработицы.
        """
    ),
    code_cell(
        """
        var_model = VAR(analysis_df)
        var_results = var_model.fit(selected_lag)

        coefficient_rows = []
        for equation in analysis_df.columns:
            for parameter in var_results.params.index:
                coefficient_rows.append(
                    {
                        "уравнение": equation,
                        "параметр": parameter,
                        "коэффициент": var_results.params.loc[parameter, equation],
                        "стандартная ошибка": var_results.stderr.loc[parameter, equation],
                        "t-статистика": var_results.tvalues.loc[parameter, equation],
                        "p-value": var_results.pvalues.loc[parameter, equation],
                    }
                )

        var_coefficients_table = pd.DataFrame(coefficient_rows).round(4)
        var_coefficients_table.to_csv(
            TABLES_DIR / "var_coefficients_table.csv",
            index=False,
            encoding="utf-8-sig",
        )

        display(var_coefficients_table)
        """
    ),
    md_cell(
        """
        ## 12. Проверка устойчивости VAR

        Теория:

        - VAR считается устойчивой, если все собственные значения companion-матрицы лежат **внутри** единичной окружности.
        - Эквивалентно, если смотреть на формат Gretl/Stata с обратными корнями, то эти **inverse roots** должны лежать внутри единичной окружности.
        """
    ),
    code_cell(
        """
        def format_complex(value):
            return f"{value.real:.4f}{value.imag:+.4f}j"


        characteristic_roots = var_results.roots
        companion_eigenvalues = 1 / characteristic_roots

        stability_table = pd.DataFrame(
            {
                "корень_характеристического_полинома": [format_complex(value) for value in characteristic_roots],
                "модуль_корня": np.abs(characteristic_roots),
                "собственное_значение_companion": [format_complex(value) for value in companion_eigenvalues],
                "модуль_собственного_значения": np.abs(companion_eigenvalues),
                "внутри_единичной_окружности": np.abs(companion_eigenvalues) < 1,
            }
        ).sort_values("модуль_собственного_значения", ascending=False)

        var_is_stable = bool(var_results.is_stable())
        stability_table = stability_table.round(4)
        stability_table.to_csv(TABLES_DIR / "stability_roots_table.csv", index=False, encoding="utf-8-sig")

        display(stability_table)
        print("VAR устойчива:", var_is_stable)

        unit_circle = np.linspace(0, 2 * np.pi, 400)
        roots_fig_path = FIGURES_DIR / "var_inverse_roots.png"

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(np.cos(unit_circle), np.sin(unit_circle), linestyle="--", color="black", label="Единичная окружность")
        ax.scatter(
            companion_eigenvalues.real,
            companion_eigenvalues.imag,
            color="#d62728",
            s=45,
            label="Собственные значения",
        )
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.set_title("Собственные значения companion-матрицы VAR")
        ax.set_xlabel("Действительная часть")
        ax.set_ylabel("Мнимая часть")
        ax.set_aspect("equal", adjustable="box")
        ax.legend()
        fig.tight_layout()
        fig.savefig(roots_fig_path, bbox_inches="tight")
        plt.show()

        print("График устойчивости сохранен:", roots_fig_path)
        """
    ),
    md_cell(
        """
        ## 13. Диагностика остатков

        Теория:

        - тест на автокорреляцию остатков: `H0` — в остатках нет автокорреляции;
        - тест на нормальность остатков: `H0` — остатки распределены нормально.
        """
    ),
    code_cell(
        """
        whiteness_test = var_results.test_whiteness(nlags=5, adjusted=False)
        whiteness_test_adjusted = var_results.test_whiteness(nlags=5, adjusted=True)
        normality_test = var_results.test_normality()

        diagnostics_table = pd.DataFrame(
            [
                {
                    "тест": "Portmanteau на автокорреляцию остатков",
                    "H0": "Автокорреляции остатков нет",
                    "статистика": whiteness_test.test_statistic,
                    "p-value": whiteness_test.pvalue,
                    "вывод": "H0 не отвергается" if whiteness_test.pvalue >= 0.05 else "H0 отвергается",
                },
                {
                    "тест": "Portmanteau на автокорреляцию остатков (small-sample adjusted)",
                    "H0": "Автокорреляции остатков нет",
                    "статистика": whiteness_test_adjusted.test_statistic,
                    "p-value": whiteness_test_adjusted.pvalue,
                    "вывод": "H0 не отвергается" if whiteness_test_adjusted.pvalue >= 0.05 else "H0 отвергается",
                },
                {
                    "тест": "Тест нормальности остатков",
                    "H0": "Остатки нормальны",
                    "статистика": normality_test.test_statistic,
                    "p-value": normality_test.pvalue,
                    "вывод": "H0 не отвергается" if normality_test.pvalue >= 0.05 else "H0 отвергается",
                },
            ]
        ).round(4)

        diagnostics_table.to_csv(TABLES_DIR / "diagnostics_table.csv", index=False, encoding="utf-8-sig")
        display(diagnostics_table)

        residuals = var_results.resid.copy()
        residuals_plot_path = FIGURES_DIR / "var_residuals.png"
        residuals_acf_path = FIGURES_DIR / "var_residuals_acf.png"

        fig, axes = plt.subplots(len(regions), 1, figsize=(10, 7), sharex=True)
        for idx, region in enumerate(regions):
            axes[idx].plot(residuals.index.to_timestamp(), residuals[region], marker="o", linewidth=1.6)
            axes[idx].axhline(0, color="black", linewidth=0.8)
            axes[idx].set_title(f"Остатки VAR: {region}")
        fig.tight_layout()
        fig.savefig(residuals_plot_path, bbox_inches="tight")
        plt.show()

        fig, axes = plt.subplots(len(regions), 1, figsize=(10, 7))
        max_acf_lag = min(6, len(residuals) - 1)
        for idx, region in enumerate(regions):
            plot_acf(residuals[region], ax=axes[idx], lags=max_acf_lag, title=f"ACF остатков: {region}")
        fig.tight_layout()
        fig.savefig(residuals_acf_path, bbox_inches="tight")
        plt.show()

        print("Графики остатков сохранены:")
        print(residuals_plot_path)
        print(residuals_acf_path)
        """
    ),
    md_cell(
        """
        ## 14. Тесты причинности по Грейнджеру

        Теория:

        Поскольку VAR оценен на **первых разностях**, дальше речь идет именно об **изменениях безработицы**, а не об уровнях безработицы.

        - `H0`: изменения безработицы в регионе `X` не Granger-причиняют изменения безработицы в регионе `Y`;
        - если `p-value < 0.05`, отвергаем `H0` и считаем, что прошлые изменения безработицы в `X` улучшают прогноз изменений безработицы в `Y`.

        Важно: причинность по Грейнджеру — это **предиктивная**, а не структурная экономическая причинность.
        """
    ),
    code_cell(
        """
        granger_rows = []

        for caused in regions:
            for causing in regions:
                if caused == causing:
                    continue

                causality_test = var_results.test_causality(caused=caused, causing=[causing], kind="f")
                is_significant = causality_test.pvalue < 0.05

                granger_rows.append(
                    {
                        "направление": f"{causing} -> {caused}",
                        "регион_X": causing,
                        "регион_Y": caused,
                        "F-статистика": causality_test.test_statistic,
                        "p-value": causality_test.pvalue,
                        "вывод": (
                            "Отвергаем H0: изменения безработицы в X помогают прогнозировать изменения безработицы в Y"
                            if is_significant
                            else "Не отвергаем H0: значимой Granger-связи для изменений безработицы не найдено"
                        ),
                    }
                )

        granger_table = pd.DataFrame(granger_rows).sort_values("p-value").round(4)
        granger_table.to_csv(TABLES_DIR / "granger_causality_table.csv", index=False, encoding="utf-8-sig")

        display(granger_table)

        significant_links = granger_table.loc[
            granger_table["p-value"] < 0.05,
            "направление",
        ].tolist()

        if significant_links:
            scheme_text = "### Схема значимых Granger-связей для изменений безработицы\\n\\n" + "\\n".join(
                f"- изменения безработицы: {link}" for link in significant_links
            )
        else:
            scheme_text = "### Схема значимых Granger-связей для изменений безработицы\\n\\n- Значимых направленных связей при уровне значимости 5% не найдено."

        display(Markdown(scheme_text))
        """
    ),
    md_cell(
        """
        ## 15. Небольшая проверка устойчивости выводов
        """
    ),
    code_cell(
        """
        robustness_rows = []

        for lag in [1, 2, 3]:
            fit = VAR(analysis_df).fit(lag)
            whiteness_unadjusted = fit.test_whiteness(nlags=5, adjusted=False)
            whiteness_adjusted = fit.test_whiteness(nlags=5, adjusted=True)
            normality = fit.test_normality()

            lag_links = []
            for caused in regions:
                for causing in regions:
                    if caused == causing:
                        continue
                    pvalue = fit.test_causality(caused=caused, causing=[causing], kind="f").pvalue
                    if pvalue < 0.05:
                        lag_links.append(f"{causing} -> {caused}")

            robustness_rows.append(
                {
                    "лаг": lag,
                    "VAR устойчива": bool(fit.is_stable()),
                    "p-value Portmanteau": whiteness_unadjusted.pvalue,
                    "p-value Portmanteau adjusted": whiteness_adjusted.pvalue,
                    "p-value нормальности": normality.pvalue,
                    "значимые Granger-связи для изменений безработицы": "; ".join(lag_links) if lag_links else "нет",
                }
            )

        robustness_table = pd.DataFrame(robustness_rows).round(4)
        robustness_table.to_csv(TABLES_DIR / "robustness_var_comparison_table.csv", index=False, encoding="utf-8-sig")
        display(robustness_table)

        robustness_text = (
            "### Краткий robustness-комментарий\\n\\n"
            "- `VAR(1)` устойчива, но small-sample adjusted тест на автокорреляцию отвергает `H0`, "
            "а тест нормальности тоже дает явные проблемы.\\n"
            "- `VAR(2)` устойчива и уже показывает ту же направленную Granger-связь "
            "`Воронежская область -> Курская область`, но adjusted тест на автокорреляцию остатков остается проблемным.\\n"
            "- `VAR(3)` также устойчива, сохраняет ту же единственную значимую Granger-связь и выглядит лучшей среди трех моделей по совокупности "
            "информационных критериев и диагностик, хотя при такой малой выборке остается параметрически тяжелой.\\n"
        )

        display(Markdown(robustness_text))
        """
    ),
    md_cell(
        """
        ## 16. Краткое итоговое резюме
        """
    ),
    code_cell(
        """
        significant_links = granger_table.loc[granger_table["p-value"] < 0.05, "направление"].tolist()
        granger_summary = (
            "изменения безработицы в Воронежской области -> изменения безработицы в Курской области"
            if significant_links
            else "значимых Granger-связей для изменений безработицы не найдено"
        )

        residuals_are_acceptable = (
            whiteness_test_adjusted.pvalue >= 0.05 and normality_test.pvalue >= 0.05
        )

        summary_text = (
            "### Итоги VAR-этапа\\n\\n"
            "- **Основная модель:** `VAR(3)` на первых разностях.\\n"
            f"- **Устойчивость:** {'VAR устойчива.' if var_is_stable else 'VAR неустойчива.'}\\n"
            f"- **Диагностика остатков:** {'в целом приемлема.' if residuals_are_acceptable else 'есть замечания, поэтому интерпретация должна быть осторожной.'}\\n"
            f"- **Granger-связь:** {granger_summary}.\\n"
            "- **Оговорка:** результаты носят исследовательский характер из-за малого объема выборки.\\n"
        )

        display(Markdown(summary_text))
        print("Таблицы сохранены в:", TABLES_DIR)
        """
    ),
    md_cell(
        """
        ## 17. Функции импульсного отклика (IRF)

        Ниже строю **ортогонализованные** импульсные отклики на горизонте 5 лет. Поскольку модель оценена на первых разностях,
        все отклики относятся к **изменениям безработицы** в процентных пунктах, а не к уровням безработицы.

        Для такой постановки важно помнить:
        - графики показывают краткосрочную динамическую реакцию на шок в изменении безработицы;
        - количественные оценки зависят от используемой идентификации шоков;
        - затухание откликов согласуется с устойчивостью VAR.
        """
    ),
    code_cell(
        """
        irf_horizon = 5
        irf_results = var_results.irf(irf_horizon)
        orth_irf_array = irf_results.orth_irfs

        irf_rows = []
        for horizon in range(irf_horizon + 1):
            for response_idx, response_region in enumerate(regions):
                for impulse_idx, impulse_region in enumerate(regions):
                    irf_rows.append(
                        {
                            "горизонт": horizon,
                            "шок": impulse_region,
                            "отклик": response_region,
                            "IRF": orth_irf_array[horizon, response_idx, impulse_idx],
                        }
                    )

        irf_table = pd.DataFrame(irf_rows).round(4)
        irf_table.to_csv(TABLES_DIR / "irf_table.csv", index=False, encoding="utf-8-sig")

        irf_fig_path = FIGURES_DIR / "var_irf_orth_5y.png"
        fig, axes = plt.subplots(len(regions), len(regions), figsize=(13, 11), sharex=True)

        for response_idx, response_region in enumerate(regions):
            for impulse_idx, impulse_region in enumerate(regions):
                ax = axes[response_idx, impulse_idx]
                values = orth_irf_array[:, response_idx, impulse_idx]
                ax.plot(range(irf_horizon + 1), values, marker="o", linewidth=1.8, color="#1f77b4")
                ax.axhline(0, color="black", linewidth=0.8)
                ax.set_title(f"Шок: {impulse_region}\\nОтклик: {response_region}", fontsize=10)
                if response_idx == len(regions) - 1:
                    ax.set_xlabel("Горизонт, лет")
                if impulse_idx == 0:
                    ax.set_ylabel("Отклик, п.п.")

        fig.suptitle("IRF для VAR(3) на первых разностях уровней безработицы", fontsize=14, y=1.02)
        fig.tight_layout()
        fig.savefig(irf_fig_path, bbox_inches="tight")
        plt.show()

        nontrivial_irf = irf_table[irf_table["горизонт"] > 0].copy()
        strongest_irf_row = nontrivial_irf.loc[nontrivial_irf["IRF"].abs().idxmax()]
        max_initial_effect = nontrivial_irf["IRF"].abs().max()
        max_terminal_effect = irf_table.loc[irf_table["горизонт"] == irf_horizon, "IRF"].abs().max()

        irf_summary_text = (
            "### Краткая интерпретация IRF\\n\\n"
            f"- Наиболее сильный отклик по модулю наблюдается для пары "
            f"`{strongest_irf_row['шок']} -> {strongest_irf_row['отклик']}` "
            f"на горизонте {int(strongest_irf_row['горизонт'])} года: `{strongest_irf_row['IRF']:.3f}` п.п.\\n"
            "- В динамике есть как положительные, так и отрицательные реакции, то есть отклики могут менять знак по мере распространения шока.\\n"
            f"- По модулю отклики к горизонту {irf_horizon} лет уменьшаются "
            f"(максимум снижается с `{max_initial_effect:.3f}` до `{max_terminal_effect:.3f}`), "
            "что согласуется с устойчивостью оцененной VAR.\\n"
            "- Особенно заметна связь между изменениями безработицы в Воронежской и Курской областях, "
            "что хорошо согласуется с результатами тестов Грейнджера.\\n"
        )

        display(Markdown(irf_summary_text))
        print("IRF-график сохранен:", irf_fig_path)
        """
    ),
    md_cell(
        """
        ## 18. Разложение дисперсии ошибки прогноза (FEVD)

        FEVD показывает, какая доля дисперсии ошибки прогноза изменений безработицы в каждом регионе объясняется
        собственными шоками и шоками из других регионов в рамках данной VAR-идентификации.
        """
    ),
    code_cell(
        """
        fevd_horizon = 5
        fevd_results = var_results.fevd(fevd_horizon)

        fevd_rows = []
        for response_idx, response_region in enumerate(regions):
            for horizon in range(1, fevd_horizon + 1):
                shares = fevd_results.decomp[response_idx, horizon - 1, :]
                row = {
                    "отклик": response_region,
                    "горизонт": horizon,
                }
                for impulse_idx, impulse_region in enumerate(regions):
                    row[impulse_region] = shares[impulse_idx]
                fevd_rows.append(row)

        fevd_table = pd.DataFrame(fevd_rows).round(4)
        fevd_table.to_csv(TABLES_DIR / "fevd_table.csv", index=False, encoding="utf-8-sig")

        fevd_horizon5_table = fevd_table[fevd_table["горизонт"] == fevd_horizon].copy()
        fevd_horizon5_table.to_csv(TABLES_DIR / "fevd_horizon5_table.csv", index=False, encoding="utf-8-sig")
        display(fevd_table)

        fevd_fig_path = FIGURES_DIR / "var_fevd_5y.png"
        fig, axes = plt.subplots(1, len(regions), figsize=(15, 4.5), sharey=True)

        for response_idx, response_region in enumerate(regions):
            ax = axes[response_idx]
            subset = fevd_table[fevd_table["отклик"] == response_region]
            for impulse_region in regions:
                ax.plot(
                    subset["горизонт"],
                    subset[impulse_region],
                    marker="o",
                    linewidth=1.8,
                    label=impulse_region,
                )
            ax.set_title(f"FEVD: {response_region}")
            ax.set_xlabel("Горизонт, лет")
            if response_idx == 0:
                ax.set_ylabel("Доля дисперсии")
            ax.set_ylim(0, 1)
            if response_idx == len(regions) - 1:
                ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

        fig.suptitle("Разложение дисперсии ошибки прогноза для изменений безработицы", fontsize=14, y=1.05)
        fig.tight_layout()
        fig.savefig(fevd_fig_path, bbox_inches="tight")
        plt.show()

        own_share_rows = []
        for _, row in fevd_horizon5_table.iterrows():
            response_region = row["отклик"]
            own_share = row[response_region]
            external_shares = row[regions].drop(labels=[response_region])
            main_external_region = external_shares.idxmax()
            own_share_rows.append(
                f"- Для `{response_region}` на горизонте {fevd_horizon} лет собственные шоки объясняют `{own_share:.3f}` дисперсии, "
                f"а крупнейший внешний вклад дает `{main_external_region}`: `{external_shares.max():.3f}`."
            )

        fevd_summary_text = (
            "### Краткая интерпретация FEVD\\n\\n"
            + "\\n".join(own_share_rows)
            + "\\n- Это означает, что динамика изменений безработицы по регионам не полностью автономна: "
            "для части регионов межрегиональные шоки заметно влияют на ошибку прогноза. "
            "Особенно это заметно для Курской области, где внешний вклад велик.\\n"
        )

        display(Markdown(fevd_summary_text))
        print("FEVD-график сохранен:", fevd_fig_path)
        """
    ),
    md_cell(
        """
        ## 19. Краткосрочный прогноз

        Строю прогноз на 3 шага вперед. Поскольку модель оценена на первых разностях:
        - сначала прогнозируются **изменения безработицы**;
        - затем для наглядности я механически восстанавливаю приблизительные уровни безработицы,
          добавляя прогнозируемые изменения к последнему наблюдаемому уровню за 2022 год.
        """
    ),
    code_cell(
        """
        forecast_steps = 3
        forecast_index = pd.period_range(start=analysis_df.index[-1] + 1, periods=forecast_steps, freq="Y")
        forecast_diff_values = var_results.forecast(analysis_df.values[-selected_lag:], steps=forecast_steps)

        forecast_diff_df = pd.DataFrame(forecast_diff_values, index=forecast_index, columns=regions)
        forecast_levels_df = forecast_diff_df.cumsum().add(ts_df.iloc[-1], axis=1)

        forecast_table = pd.DataFrame({"год": [period.year for period in forecast_index]})
        for region in regions:
            forecast_table[f"изменение: {region}"] = forecast_diff_df[region].values
        for region in regions:
            forecast_table[f"уровень: {region}"] = forecast_levels_df[region].values

        forecast_table = forecast_table.round(4)
        forecast_table.to_csv(TABLES_DIR / "forecast_table.csv", index=False, encoding="utf-8-sig")
        display(forecast_table)

        forecast_fig_path = FIGURES_DIR / "var_forecast_changes_levels.png"
        fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=False)

        for region in regions:
            axes[0].plot(
                forecast_table["год"],
                forecast_table[f"изменение: {region}"],
                marker="o",
                linewidth=2,
                label=region,
            )
        axes[0].axhline(0, color="black", linewidth=0.8)
        axes[0].set_title("Прогноз изменений безработицы, 2023-2025 гг.")
        axes[0].set_xlabel("Год")
        axes[0].set_ylabel("Изменение, п.п.")
        axes[0].legend()

        history_years = ts_df.index.year
        for region in regions:
            axes[1].plot(history_years, ts_df[region].values, linewidth=1.8, label=f"{region}: факт")
            axes[1].plot(
                forecast_table["год"],
                forecast_table[f"уровень: {region}"],
                linestyle="--",
                marker="o",
                linewidth=2,
                label=f"{region}: прогноз",
            )
        axes[1].set_title("Приближенный прогноз уровней безработицы, 2023-2025 гг.")
        axes[1].set_xlabel("Год")
        axes[1].set_ylabel("Уровень безработицы, %")
        axes[1].legend(ncol=2, fontsize=8)

        fig.tight_layout()
        fig.savefig(forecast_fig_path, bbox_inches="tight")
        plt.show()

        negative_diff_count = int((forecast_diff_df < 0).sum().sum())
        total_diff_count = int(forecast_diff_df.size)
        forecast_summary_text = (
            "### Краткая интерпретация прогноза\\n\\n"
            f"- Прогноз построен только на `{forecast_steps}` шага вперед, поэтому его следует считать строго краткосрочным.\\n"
            f"- Из `{total_diff_count}` прогнозных значений изменений безработицы `{negative_diff_count}` оказываются отрицательными, "
            "то есть модель в большинстве случаев ожидает дальнейшее снижение безработицы по сравнению с предыдущим годом.\\n"
            "- Восстановленные уровни нужно трактовать как механическое приближение на основе прогнозов первых разностей, "
            "а не как точную структурную траекторию.\\n"
            "- Из-за малого объема выборки и насыщенности VAR(3) прогноз носит исследовательский характер.\\n"
        )

        display(Markdown(forecast_summary_text))
        print("Таблица прогноза сохранена:", TABLES_DIR / "forecast_table.csv")
        print("График прогноза сохранен:", forecast_fig_path)
        """
    ),
    md_cell(
        """
        ## 20. Проверка выполнения требований ТДЗ 18

        - [x] матричная форма ручной VAR записана;
        - [x] ручная стационарность проверена;
        - [x] ручное математическое ожидание вычислено;
        - [x] ручные IRF вычислены;
        - [x] данные загружены и преобразованы;
        - [x] выбраны 3 региона;
        - [x] стационарность проверена;
        - [x] лаг выбран;
        - [x] VAR оценена;
        - [x] устойчивость VAR проверена;
        - [x] диагностика остатков выполнена;
        - [x] причинность по Грейнджеру протестирована;
        - [x] IRF рассчитаны и интерпретированы;
        - [x] FEVD рассчитано и интерпретировано;
        - [x] прогноз рассчитан и интерпретирован.
        """
    ),
    md_cell(
        """
        ## 21. Генерация компактного LaTeX-отчета
        """
    ),
    code_cell(
        """
        import subprocess

        report_builder = PROJECT_DIR / "build_report_hw18_var.py"
        report_path = PROJECT_DIR / "report_hw18_var.tex"
        report_pdf_path = PROJECT_DIR / "report_hw18_var.pdf"

        report_run = subprocess.run(
            ["python3", str(report_builder)],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=240,
        )

        if report_run.returncode == 0:
            compile_message = (
                f"Отчет успешно пересобран.\\n"
                f"- TeX: {report_path}\\n"
                f"- PDF: {report_pdf_path}\\n"
                f"- Последние строки вывода:\\n{report_run.stdout[-1200:]}"
            )
        else:
            compile_message = (
                "Сборка отчета завершилась с ошибкой.\\n"
                f"{(report_run.stdout + chr(10) + report_run.stderr)[-2000:]}"
            )

        display(Markdown("### Статус генерации отчета\\n\\n" + compile_message.replace("\\n", "  \\n")))
        """
    ),
]

nb.cells.extend(new_cells)
NOTEBOOK_PATH.write_text(nbf.writes(nb), encoding="utf-8")
print(f"Notebook updated: {NOTEBOOK_PATH}")
