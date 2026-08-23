# Project Summary

## Что было сделано

- Через MOEX ISS собрана дневная история опционов на IMOEX-related contracts за `2024-01-03` -> `2026-05-19`.
- Проведён sanity-check источника и discovery historical `SECID`.
- Из нескольких семейств (`IM`, `MM`, `MX`) выбрано семейство `MM` как основное для анализа.
- Собран отдельный `MM`-датасет опционов.
- Загружены daily candles underlying (`IMOEX`) за `2024-01-03` -> `2026-05-21`.
- Выполнен merge опционов с underlying.
- Подготовлен modelling sample с базовыми признаками:
  - `DTE_DAYS`
  - `moneyness`
  - `log_moneyness`
  - `hv_21d`
  - `hv_63d`
  - `vol_regime`
  - `trend_regime`
  - `regime_label`
- Построен baseline `Black-76` pricing с двумя volatility input:
  - `hv_21d`
  - `hv_63d`
- Выполнен error analysis по regime, moneyness и DTE.

## Главные датасеты

- Финальный merged dataset:
  [imoex_mm_options_with_underlying_2024_2026.parquet](/Users/maria/Desktop/Code/HSE/COURSEBOOK/data/final/imoex_mm_options_with_underlying_2024_2026.parquet)
- Modelling sample:
  [imoex_mm_options_modelling_sample.parquet](/Users/maria/Desktop/Code/HSE/COURSEBOOK/data/final/imoex_mm_options_modelling_sample.parquet)
- Baseline pricing results:
  [model_pricing_baseline.parquet](/Users/maria/Desktop/Code/HSE/COURSEBOOK/results/model_pricing_baseline.parquet)

## Размер и покрытие

- `MM` options dataset after merge:
  - `19,574` строк
  - `92` уникальных `SECID`
  - `604` уникальных торговых даты
- Modelling sample after soft filter:
  - `19,327` строк
  - сохранено `98.7%` выборки
- Underlying coverage after merge:
  - `100%`
  - пропусков по `UNDERLYING_CLOSE` нет

## Качество данных

- `SETTLEPRICE` заполнен полностью и используется как `market_price`.
- Дублей по `SECID x TRADEDATE` нет.
- Датасет подходит для research MVP.
- Ограничение:
  trade-поля (`OPEN/HIGH/LOW/CLOSE/VOLUME`) заметно слабее и часто нулевые, поэтому основной упор разумно делать на `SETTLEPRICE`, а не на сделочную ликвидность.

## Выбор семейства

- `MM` оставлено как основное семейство.
- Причина:
  - лучшее покрытие по датам;
  - более однородный масштаб страйков;
  - названия контрактов явно указывают на опционы на фьючерс.
- `IM` и `MX` не используются как основной рабочий набор.

## Baseline pricing results

### Overall

- `Black-76 + hv_21d`
  - `N = 18,926`
  - `MAE ≈ 326.29`
  - `RMSE ≈ 452.59`
  - `mean_error ≈ -309.58`
- `Black-76 + hv_63d`
  - `N = 18,077`
  - `MAE ≈ 309.00`
  - `RMSE ≈ 435.45`
  - `mean_error ≈ -293.00`

### Краткий вывод по baseline

- `hv_63d` работает немного лучше, чем `hv_21d`.
- Baseline систематически недооценивает рыночные цены:
  `mean_error` отрицательный у обеих версий.

## Error analysis

- Ошибки растут в `high_vol` regime.
- Самые тяжёлые состояния для baseline:
  - `high_vol_up`
  - `high_vol_down`
  - `mid_vol_down`
- По `moneyness` baseline хуже работает на хвостах:
  - `deep_OTM`
  - `deep_ITM`
- По `DTE` baseline заметно хуже на длинных опционах (`91+` дней), чем на коротких.

## Итоговый исследовательский вывод

- Датасет пригоден для курсовой как research MVP.
- `Black-76 + historical volatility` годится как baseline benchmark.
- Но этого недостаточно как финальной модели рынка.
- Следующий логичный шаг:
  - `GARCH volatility forecast`
  - или implied-vol based benchmark
  - затем сравнение по `MAE`, `RMSE`, `IV error` и по market regimes.
