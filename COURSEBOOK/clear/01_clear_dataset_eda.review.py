# CELL 1
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path('/Users/maria/Desktop/Code/HSE/COURSEBOOK')
data_path = project_root / 'clear/mm_options_with_mxi_underlying_2024_2026.parquet'
df = pd.read_parquet(data_path)
df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'], errors='coerce')
df['expiry_date'] = pd.to_datetime(df['expiry_date'], errors='coerce')
df.shape

# CELL 2
summary = {
    'rows': len(df),
    'unique_option_secids': int(df['SECID'].nunique()),
    'unique_underlying_secids': int(df['UNDERLYING_SECID'].dropna().nunique()),
    'trade_date_min': df['TRADEDATE'].min().date(),
    'trade_date_max': df['TRADEDATE'].max().date(),
    'expiry_min': df['expiry_date'].min().date(),
    'expiry_max': df['expiry_date'].max().date(),
    'option_types': df['option_type'].value_counts(dropna=False).to_dict(),
}
pd.Series(summary)

# CELL 3
print('Underlying contracts:', sorted(df['UNDERLYING_SHORTNAME'].dropna().unique().tolist()))
display(df['UNDERLYING_SHORTNAME'].value_counts().to_frame('rows'))
display(df.groupby(df['TRADEDATE'].dt.year).size().to_frame('rows'))

# CELL 4
eda = {
    'unique_strikes': int(df['strike'].nunique()),
    'unique_expiries': int(df['expiry_date'].nunique()),
    'missing_market_price_share': float(df['market_price'].isna().mean()),
    'missing_underlying_price_share': float(df['underlying_price'].isna().mean()),
    'duplicate_secid_tradedate_rows': int(df.duplicated(['SECID', 'TRADEDATE']).sum()),
}
pd.Series(eda)

missing = df[['market_price', 'underlying_price', 'moneyness', 'log_moneyness', 'UNDERLYING_CLOSE', 'UNDERLYING_VOLUME']].isna().mean().sort_values(ascending=False)
missing.to_frame("missing_share")

# CELL 5
verdict = [
    '1. Датасет пересобран с MXI-фьючерсным underlying, а не с индексом IMOEX spot.',
    '2. Покрытие по option rows полное: underlying_price есть для всех строк.',
    '3. Для моделирования базовой цены underlying разумно использовать UNDERLYING_SETTLEPRICE / underlying_price.',
    '4. Поля UNDERLYING_CLOSE и часть trade-полей фьючерса местами пустые, но settlement-based pricing dataset выглядит рабочим.',
    '5. Это более корректная база для следующего шага, чем предыдущий merged dataset с IMOEX spot.',
]
print('\n'.join(verdict))

