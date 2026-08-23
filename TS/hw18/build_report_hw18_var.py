from pathlib import Path
import subprocess
import shutil

import numpy as np
import pandas as pd


PROJECT_DIR = Path("/Users/maria/Desktop/Code/HSE/TS/hw18")
TABLES_DIR = PROJECT_DIR / "tables"
FIGURES_DIR = PROJECT_DIR / "figures"
REPORT_TEX = PROJECT_DIR / "report_hw18_var.tex"
REPORT_PDF = PROJECT_DIR / "report_hw18_var.pdf"


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    result = str(text)
    for old, new in replacements.items():
        result = result.replace(old, new)
    return result


def latex_table_block(
    dataframe: pd.DataFrame,
    caption: str,
    label: str,
    resize: bool = True,
    width: str = "\\textwidth",
    size: str = "\\small",
) -> str:
    table_tex = dataframe.to_latex(
        index=False,
        escape=True,
        float_format=lambda value: f"{value:.4f}",
    )
    if resize:
        body = "\\resizebox{" + width + "}{!}{%\n" + table_tex + "}\n"
    else:
        body = size + "\n" + table_tex
    return (
        "\\begin{table}[H]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"{body}"
        "\\end{table}\n"
    )


def latex_figure_block(relative_path: str, caption: str, label: str, width: str = "0.92\\textwidth") -> str:
    return (
        "\\begin{figure}[H]\n"
        "\\centering\n"
        f"\\includegraphics[width={width}]{{{relative_path}}}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{figure}\n"
    )


stationarity = pd.read_csv(TABLES_DIR / "stationarity_table.csv")
lags = pd.read_csv(TABLES_DIR / "lag_selection_table.csv")
diagnostics = pd.read_csv(TABLES_DIR / "diagnostics_table.csv")
granger = pd.read_csv(TABLES_DIR / "granger_causality_table.csv")
fevd5 = pd.read_csv(TABLES_DIR / "fevd_horizon5_table.csv")
forecast = pd.read_csv(TABLES_DIR / "forecast_table.csv")
robustness = pd.read_csv(TABLES_DIR / "robustness_var_comparison_table.csv")
manual_stationarity = pd.read_csv(TABLES_DIR / "manual_var1_stationarity_table.csv")
manual_mean = pd.read_csv(TABLES_DIR / "manual_var1_mean_table.csv")
manual_irf = pd.read_csv(TABLES_DIR / "manual_var1_irf_table.csv")

selected_regions = [
    "Белгородская область",
    "Курская область",
    "Воронежская область",
]

significant_granger = granger.loc[granger["p-value"] < 0.05, "направление"].tolist()
granger_line = significant_granger[0] if significant_granger else "значимых связей не найдено"

belgorod_own = float(fevd5.loc[fevd5["отклик"] == "Белгородская область", "Белгородская область"].iloc[0])
kursk_own = float(fevd5.loc[fevd5["отклик"] == "Курская область", "Курская область"].iloc[0])
kursk_vor = float(fevd5.loc[fevd5["отклик"] == "Курская область", "Воронежская область"].iloc[0])
kursk_bel = float(fevd5.loc[fevd5["отклик"] == "Курская область", "Белгородская область"].iloc[0])
vor_own = float(fevd5.loc[fevd5["отклик"] == "Воронежская область", "Воронежская область"].iloc[0])
vor_bel = float(fevd5.loc[fevd5["отклик"] == "Воронежская область", "Белгородская область"].iloc[0])

forecast_negative = int(
    (
        forecast[[
            "изменение: Белгородская область",
            "изменение: Курская область",
            "изменение: Воронежская область",
        ]] < 0
    ).sum().sum()
)

stationarity_short = stationarity[[
    "регион",
    "форма_ряда",
    "ADF p-value",
    "KPSS p-value",
    "итог",
]].copy()
diagnostics_short = diagnostics[["тест", "p-value", "вывод"]].copy()
granger_short = granger[["направление", "p-value", "вывод"]].copy()
forecast_changes = forecast[[
    "год",
    "изменение: Белгородская область",
    "изменение: Курская область",
    "изменение: Воронежская область",
]].copy()
forecast_levels = forecast[[
    "год",
    "уровень: Белгородская область",
    "уровень: Курская область",
    "уровень: Воронежская область",
]].copy()

manual_alpha = np.array([[1.0], [3.0]])
manual_B = np.array([[0.6, -0.2], [-0.1, 0.4]])
manual_B2 = manual_B @ manual_B
manual_eigvals = np.linalg.eigvals(manual_B)
manual_mu = np.linalg.solve(np.eye(2) - manual_B, manual_alpha)


report_sections = [
    (
        "\\section{Стационарность заданной VAR(1)}\n"
        "Заданная система имеет вид:\n"
        "\\[\n"
        "y_{1,t} = 1 + 0.6 y_{1,t-1} - 0.2 y_{2,t-1} + \\varepsilon_{1,t},\n"
        "\\]\n"
        "\\[\n"
        "y_{2,t} = 3 - 0.1 y_{1,t-1} + 0.4 y_{2,t-1} + \\varepsilon_{2,t}.\n"
        "\\]\n"
        "В матричной форме:\n"
        "\\[\n"
        "Y_t = \\alpha + B Y_{t-1} + \\varepsilon_t,\n"
        "\\qquad\n"
        "\\alpha = \\begin{pmatrix}1\\\\3\\end{pmatrix},\n"
        "\\qquad\n"
        "B = \\begin{pmatrix}0.6 & -0.2\\\\-0.1 & 0.4\\end{pmatrix}.\n"
        "\\]\n"
        "Через лаговый оператор это записывается как\n"
        "\\[\n"
        "(I - BL)Y_t = \\alpha + \\varepsilon_t.\n"
        "\\]\n"
        "Стационарность проверяется по корням характеристического уравнения:\n"
        "\\[\n"
        "\\det(\\lambda I - B) = "
        "\\det\\begin{pmatrix}\\lambda - 0.6 & 0.2\\\\0.1 & \\lambda - 0.4\\end{pmatrix}"
        " = (\\lambda - 0.6)(\\lambda - 0.4) - 0.02"
        " = \\lambda^2 - \\lambda + 0.22 = 0.\n"
        "\\]\n"
        "Далее\n"
        "\\[\n"
        "I-B = \\begin{pmatrix}0.4 & 0.2\\\\0.1 & 0.6\\end{pmatrix},\n"
        "\\qquad\n"
        "(I-B)^{-1} = \\begin{pmatrix}2.727273 & -0.909091\\\\-0.454545 & 1.818182\\end{pmatrix}.\n"
        "\\]\n"
        "И математическое ожидание:\n"
        "\\[\n"
        "\\mu = (I - B)^{-1}\\alpha = "
        "\\begin{pmatrix}2.727273 & -0.909091\\\\-0.454545 & 1.818182\\end{pmatrix}"
        "\\begin{pmatrix}1\\\\3\\end{pmatrix}"
        "= \\begin{pmatrix}0\\\\5\\end{pmatrix}.\n"
        "\\]\n"
        + latex_table_block(
            manual_stationarity,
            "Собственные значения матрицы $B$",
            "tab:manual_stationarity",
            resize=False,
            size="\\normalsize",
        )
        + latex_table_block(
            manual_mean,
            "Безусловное математическое ожидание",
            "tab:manual_mean",
            resize=False,
            size="\\normalsize",
        )
        + f"Собственные значения равны {manual_eigvals[0]:.6f} и {manual_eigvals[1]:.6f}; оба по модулю меньше единицы. "
        f"Следовательно, VAR(1) стационарна и имеет конечное постоянное среднее $({manual_mu[0,0]:.6f}, {manual_mu[1,0]:.6f})' = (0, 5)'$.\n"
    ),
    (
        "\\section{Функция импульсного отклика}\n"
        "Для VAR(1) ручные IRF определяются степенями матрицы перехода:\n"
        "\\[\n"
        "IRF(1) = B = \\begin{pmatrix}0.6 & -0.2\\\\-0.1 & 0.4\\end{pmatrix},\n"
        "\\qquad\n"
        "IRF(2) = B^2 = \\begin{pmatrix}0.38 & -0.20\\\\-0.10 & 0.18\\end{pmatrix}.\n"
        "\\]\n"
        + latex_table_block(
            manual_irf,
            "Ручные значения IRF для заданной VAR(1)",
            "tab:manual_irf",
            resize=False,
            size="\\normalsize",
        )
        + latex_figure_block("figures/manual_var1_irf.png", "Схема ручных IRF для заданной VAR(1)", "fig:manual_irf", width="0.78\\textwidth")
        + "Шок в $y_1$ дает положительный затухающий эффект на $y_1$ и слабый отрицательный эффект на $y_2$. "
        "Шок в $y_2$ дает отрицательный эффект на $y_1$ и положительный затухающий эффект на $y_2$. "
        "Поскольку система стационарна, отклики по мере роста горизонта уменьшаются по модулю.\n"
    ),
    (
        "\\section{Данные и выбор регионов}\n"
        "\\begin{itemize}\n"
        "\\item переменная: уровень безработицы, в процентах;\n"
        "\\item частота: годовая;\n"
        "\\item число наблюдений в уровнях: 23;\n"
        "\\item модель далее строится не по уровням, а по первым разностям.\n"
        "\\end{itemize}\n"
        + latex_figure_block("figures/unemployment_selected_regions_combined.png", "Исходные ряды уровней безработицы", "fig:data")
        + "Используются годовые данные по уровню безработицы за 2000--2022 гг. "
        "Выбраны три соседних региона: Белгородская, Курская и Воронежская области. "
        "Такой выбор позволяет сохранить географическую и экономическую сопоставимость рядов.\n"
    ),
    (
        "\\section{Предварительный анализ}\n"
        "На графиках видно общее снижение безработицы с отдельными всплесками. "
        "Следовательно, в уровнях ряды выглядят не вполне стационарными, и это нужно проверить формально. "
        "Дополнительная оговорка: после перехода к первым разностям остается около 22 наблюдений, то есть выборка мала.\n"
    ),
    (
        "\\section{Стационарность рядов}\n"
        + latex_table_block(stationarity_short, "Тесты стационарности", "tab:stationarity")
        + "По уровням ADF не отвергает единичный корень, а KPSS отвергает стационарность. "
        "По первым разностям картина становится обратной: ряды можно считать стационарными. "
        "Поэтому итоговая VAR оценивается на первых разностях.\n"
    ),
    (
        "\\section{Выбор лага и спецификация VAR}\n"
        "Эмпирическая модель записывается как\n"
        "\\[\n"
        "\\Delta Y_t = c + A_1 \\Delta Y_{t-1} + A_2 \\Delta Y_{t-2} + A_3 \\Delta Y_{t-3} + \\varepsilon_t.\n"
        "\\]\n"
        + latex_table_block(lags, "Выбор длины лага", "tab:lags")
        + latex_table_block(robustness, "Небольшая проверка устойчивости вывода по лагам", "tab:robustness")
        + "Информационные критерии AIC, BIC и HQIC выбирают лаг $p = 3$. "
        "Итоговая спецификация --- \\textbf{VAR(3) на первых разностях уровней безработицы}. "
        "Это решение поддерживается диагностикой и сравнением с VAR(1) и VAR(2), но из-за малой выборки VAR(3) нужно трактовать осторожно.\n"
    ),
    (
        "\\section{Диагностика модели}\n"
        + latex_table_block(diagnostics_short, "Диагностика остатков VAR(3)", "tab:diag")
        + latex_figure_block("figures/var_inverse_roots.png", "Собственные значения companion-матрицы", "fig:roots", width="0.60\\textwidth")
        + "VAR(3) устойчива: собственные значения лежат внутри единичной окружности. "
        "Тесты на автокорреляцию и нормальность остатков на уровне 5\\% не дают критических возражений. "
        "Для учебной задачи модель можно считать приемлемой, хотя малая выборка остается ограничением.\n"
    ),
    (
        "\\section{Причинность по Грейнджеру}\n"
        + latex_table_block(granger_short, "Тесты Грейнджера", "tab:granger")
        + f"Так как модель оценена на первых разностях, выводы относятся к \\textbf{{изменениям безработицы}}. "
        f"Единственная статистически значимая связь: \\textbf{{{latex_escape(granger_line)}}}. "
        "Это предиктивная, а не структурная причинность.\n"
    ),
    (
        "\\section{Функции импульсного отклика}\n"
        + latex_figure_block("figures/var_irf_orth_5y.png", "IRF для VAR(3) на первых разностях", "fig:irf")
        + "IRF интерпретируются для \\textbf{изменений безработицы}. "
        "Наиболее заметная межрегиональная реакция наблюдается для отклика Курской области на шок в Воронежской области. "
        "Знаки откликов могут меняться, но на дальних горизонтах реакции затухают, что согласуется с устойчивостью VAR.\n"
    ),
    (
        "\\section{Разложение дисперсии ошибки прогноза}\n"
        + latex_table_block(fevd5, "FEVD на горизонте 5 лет", "tab:fevd5")
        + latex_figure_block("figures/var_fevd_5y.png", "Разложение дисперсии ошибки прогноза", "fig:fevd")
        + "FEVD относится к \\textbf{изменениям безработицы}. "
        f"Для Белгородской области доминируют собственные шоки ({belgorod_own:.3f}). "
        f"Для Курской области вклад внешних шоков велик: Воронежская область дает {kursk_vor:.3f}, Белгородская --- {kursk_bel:.3f}, что вместе больше собственной доли {kursk_own:.3f}. "
        f"Для Воронежской области собственная доля {vor_own:.3f}, но влияние Белгородской области тоже заметно ({vor_bel:.3f}). "
        "Значит, межрегиональная динамика особенно важна для Курской области.\n"
    ),
    (
        "\\section{Прогноз}\n"
        + latex_table_block(forecast_changes, "Прогноз изменений безработицы", "tab:forecast_changes")
        + latex_table_block(forecast_levels, "Приближенный прогноз уровней безработицы", "tab:forecast_levels")
        + latex_figure_block("figures/var_forecast_changes_levels.png", "Краткосрочный прогноз по VAR(3)", "fig:forecast")
        + "Сначала прогнозируются изменения безработицы, а затем уровни восстанавливаются механически от значения 2022 года. "
        f"Из 9 прогнозных значений изменений безработицы {forecast_negative} отрицательны, то есть модель чаще ожидает дальнейшее снижение безработицы. "
        "Однако прогноз следует считать только краткосрочным и исследовательским из-за малого объема выборки.\n"
    ),
    (
        "\\section{Итоговый вывод}\n"
        "Заданная VAR(1) является стационарной, ее импульсные отклики затухают, а безусловное среднее конечно. "
        "В эмпирической части модель \\textbf{VAR(3) на первых разностях} для Белгородской, Курской и Воронежской областей "
        "показывает устойчивую, но достаточно чувствительную к малой выборке динамику. "
        "Наиболее содержательный результат --- связь от изменений безработицы в Воронежской области к изменениям безработицы в Курской области; "
        "это подтверждается и тестами Грейнджера, и IRF, и FEVD. Следовательно, динамика безработицы в выбранных регионах частично взаимосвязана, "
        "но количественные выводы и прогноз следует трактовать осторожно из-за малого числа наблюдений.\n"
    ),
]


tex = (
    "\\documentclass[12pt,a4paper]{article}\n"
    "\\usepackage[a4paper,margin=2.2cm]{geometry}\n"
    "\\usepackage[T2A]{fontenc}\n"
    "\\usepackage[utf8]{inputenc}\n"
    "\\usepackage[russian]{babel}\n"
    "\\usepackage{amsmath}\n"
    "\\usepackage{graphicx}\n"
    "\\usepackage{booktabs}\n"
    "\\usepackage{float}\n"
    "\\usepackage{hyperref}\n"
    "\\usepackage{array}\n"
    "\\usepackage{caption}\n"
    "\\usepackage{enumitem}\n"
    "\\captionsetup{font=small}\n"
    "\\setlist[itemize]{noitemsep, topsep=3pt}\n"
    "\\title{ТДЗ 18: VAR-модель региональной безработицы}\n"
    "\\author{}\n"
    "\\date{}\n"
    "\\begin{document}\n"
    "\\maketitle\n"
    "\\thispagestyle{empty}\n"
    + "\n".join(report_sections)
    + "\n\\end{document}\n"
)

REPORT_TEX.write_text(tex, encoding="utf-8")
print(f"Written {REPORT_TEX}")

engine = shutil.which("pdflatex") or shutil.which("xelatex") or shutil.which("lualatex")
if engine is None:
    raise SystemExit("No LaTeX engine found")

for _ in range(2):
    result = subprocess.run(
        [engine, "-interaction=nonstopmode", REPORT_TEX.name],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=False,
        timeout=180,
    )
    if result.returncode != 0:
        stdout = result.stdout.decode("utf-8", errors="replace")
        stderr = result.stderr.decode("utf-8", errors="replace")
        tail = (stdout + "\n" + stderr)[-4000:]
        raise RuntimeError(f"LaTeX compilation failed:\n{tail}")

if not REPORT_PDF.exists():
    raise RuntimeError("PDF was not created")

print(f"Written {REPORT_PDF}")
