# Final Research Summary

- Лучший простой baseline по итогам сравнения моделей: `Black-76 + hv_63d`.
- `GARCH` в текущем MVP не дал улучшения по `MAE` и `RMSE` относительно `hv_63d`.
- Ошибки моделей растут в `high_vol` regime.
- Ошибки выше на длинных `DTE` и в крайних сегментах moneyness.
- OLS-проверка подтверждает, что `DTE` и `abs_log_moneyness` статистически связаны с размером pricing error.
- Значит `Black-76` с простыми volatility inputs полезен как baseline benchmark, но не полностью описывает рыночную поверхность опционов.
- Следующий естественный шаг: implied-vol benchmark или более сильная volatility / surface model.
