# Proposal EDA — экспортированные артефакты

Сгенерировано из ноутбука `project_analysis.ipynb`. Содержимое соответствует разделам EDA и базовым проверкам preferred specification.

| Файл | Содержание |
|---|---|
| `01_life_expectancy_distribution.png` | Гистограмма распределения `life_expectancy` (85 субъектов РФ, 2015) |
| `02_emissions_distribution.png` | Гистограмма `ln_emissions_h1_pc` |
| `03_life_expectancy_vs_emissions.png` | Scatter `life_expectancy ~ ln_emissions_h1_pc` |
| `04_emissions_vs_mining_instrument.png` | Scatter `ln_emissions_h1_pc ~ ln_emp_mining_pc` |
| `05_correlation_heatmap.png` | Корреляционная матрица preferred baseline |
| `06_descriptive_statistics.csv` | Дескриптивная статистика |
| `06_descriptive_statistics.xlsx` | То же в Excel |

**Источник данных**: `validated_dataset_extended.csv` (N = 85 субъектов РФ; 2015 год; показатели занятости — 2017 как приближение).

**Preferred baseline**:
- DV: `life_expectancy`
- Эндогенный регрессор: `ln_emissions_h1_pc`
- IV: `ln_emp_mining_pc`
- Контроли: `ln_grp_pc`, `urban_share`, `ln_population_avg`

**Дополнительные проверки**:
- `age_struct_share` используется только в robustness-check.
- `visits_per_capita` намеренно не входит в preferred baseline.
