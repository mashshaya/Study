import pandas as pd
import numpy as np
import requests
import io
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews
import matplotlib.pyplot as plt

# Function to fetch indicator data for a given country from World Bank API

def fetch_worldbank_indicator(indicator_code, country_code='RUS'):
    """Fetches annual indicator data for a country from the World Bank API and returns a pandas Series indexed by year."""
    url = f'https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}?format=json&per_page=1000'
    response = requests.get(url)
    data_json = response.json()
    pages = data_json[0]['pages']
    records = []
    # iterate through pages
    for page in range(1, pages + 1):
        resp = requests.get(f'{url}&page={page}')
        page_data = resp.json()[1]
        for entry in page_data:
            year = int(entry['date']) if entry['date'] != '' else None
            value = entry['value']
            if value is not None:
                records.append({'year': year, 'value': float(value)})
    # create DataFrame
    df = pd.DataFrame(records).dropna().drop_duplicates(subset='year').sort_values('year')
    df = df.set_index('year')['value']
    return df

# Approximate Phillips-Perron test implementation
def phillips_perron_test(series, trend='c', lags=None):
    """
    Approximate Phillips–Perron test for a unit root.
    The implementation is based on the approach used in statsmodels' adfuller but uses Newey-West long-run variance.

    Parameters
    ----------
    series : array-like
        Time series data.
    trend : {'n', 'c', 'ct'}
        'n' for no constant, 'c' for constant, 'ct' for constant and trend.
    lags : int or None
        Newey-West lag truncation. If None, uses floor(4*(n/100)**(2/9)) as per Andrews (1991).

    Returns
    -------
    statistic, p_value
        Test statistic and approximated p-value using MacKinnon's approximate p-values for the ADF test (as a proxy).
    """
    y = np.asarray(series).astype(float)
    y = y[~np.isnan(y)]
    n = len(y)
    y_diff = np.diff(y)
    y_lag = y[:-1]
    # prepare design matrix
    X = y_lag[:, None]
    if trend == 'c':
        X = sm.add_constant(X, has_constant='add')
    elif trend == 'ct':
        trend_arr = np.arange(1, n)
        X = np.column_stack([np.ones(n - 1), trend_arr, y_lag])
    # Fit OLS on differences
    model = sm.OLS(y_diff, X).fit()
    rho = model.params[-1]  # coefficient on y_{t-1}
    # Compute test statistic
    se = model.bse[-1]
    statistic = (rho) / se
    # Long-run variance using Newey-West
    # choose lag if not provided
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    eps = model.resid
    s0 = np.sum(eps ** 2) / (n - 1)
    s1 = 0
    for j in range(1, lags + 1):
        weight = 1 - j / (lags + 1)
        gamma = np.sum(eps[j:] * eps[:-j]) / (n - 1)
        s1 += 2 * weight * gamma
    lrv = s0 + s1
    sigma = lrv
    # compute standard error of rho
    se_rho_pp = np.sqrt(sigma) / np.sqrt(np.sum((y_lag - y_lag.mean()) ** 2))
    statistic = (rho - 1) / se_rho_pp
    # Use MacKinnon approximate p-values from adfuller
    # We'll use statsmodels adfuller to compute approximate p-value on the same series with the same trend
    adf_result = adfuller(y, regression='nc' if trend == 'n' else ('c' if trend == 'c' else 'ct'), autolag='AIC')
    p_value = adf_result[1]
    return statistic, p_value

# Prepare the data for the selected indicators
indicators = {
    'NY.GDP.MKTP.CD': 'GDP, current US$',
    'FP.CPI.TOTL.ZG': 'Inflation (consumer prices, annual %)',
    'SP.DYN.LE00.IN': 'Life expectancy at birth, total (years)'
}

# Fetch data for each indicator
series_data = {}
for code in indicators.keys():
    series = fetch_worldbank_indicator(code)
    series_data[code] = series

# Restrict periods for analysis (20–60 years)
periods = {
    'NY.GDP.MKTP.CD': (1988, 2024),
    'FP.CPI.TOTL.ZG': (1993, 2024),
    'SP.DYN.LE00.IN': (1960, 2023)
}

# Create cleaned and aligned series for each indicator
clean_series = {}
for code, (start, end) in periods.items():
    ser = series_data[code].loc[start:end]
    clean_series[code] = ser

# Plot series and save images
plt.style.use('ggplot')
for code, series in clean_series.items():
    plt.figure(figsize=(8, 3))
    plt.plot(series.index, series.values, marker='o', linewidth=1)
    plt.title(f"{indicators[code]} (Russia)")
    plt.xlabel("Year")
    plt.ylabel(indicators[code])
    plt.tight_layout()
    plt.savefig(f'/home/oai/share/{code}_series_new.png')
    plt.close()

# Perform stationarity tests: ADF, PP, KPSS for levels and first differences
results = {}
for code, series in clean_series.items():
    y = series.values
    # level tests
    adf_c = adfuller(y, regression='c', autolag='AIC')  # constant
    adf_ct = adfuller(y, regression='ct', autolag='AIC')  # constant + trend
    pp_c = phillips_perron_test(y, trend='c')
    pp_ct = phillips_perron_test(y, trend='ct')
    kpss_c = kpss(y, regression='c', nlags='auto')
    kpss_ct = kpss(y, regression='ct', nlags='auto')
    # first differences
    dy = np.diff(y)
    adf_diff_c = adfuller(dy, regression='c', autolag='AIC')
    adf_diff_ct = adfuller(dy, regression='ct', autolag='AIC')
    pp_diff_c = phillips_perron_test(dy, trend='c')
    pp_diff_ct = phillips_perron_test(dy, trend='ct')
    kpss_diff_c = kpss(dy, regression='c', nlags='auto')
    kpss_diff_ct = kpss(dy, regression='ct', nlags='auto')
    results[code] = {
        'level': {
            'ADF_c': {'stat': adf_c[0], 'p': adf_c[1]},
            'ADF_ct': {'stat': adf_ct[0], 'p': adf_ct[1]},
            'PP_c': {'stat': pp_c[0], 'p': pp_c[1]},
            'PP_ct': {'stat': pp_ct[0], 'p': pp_ct[1]},
            'KPSS_c': {'stat': kpss_c[0], 'p': kpss_c[1]},
            'KPSS_ct': {'stat': kpss_ct[0], 'p': kpss_ct[1]},
        },
        'diff': {
            'ADF_c': {'stat': adf_diff_c[0], 'p': adf_diff_c[1]},
            'ADF_ct': {'stat': adf_diff_ct[0], 'p': adf_diff_ct[1]},
            'PP_c': {'stat': pp_diff_c[0], 'p': pp_diff_c[1]},
            'PP_ct': {'stat': pp_diff_ct[0], 'p': pp_diff_ct[1]},
            'KPSS_c': {'stat': kpss_diff_c[0], 'p': kpss_diff_c[1]},
            'KPSS_ct': {'stat': kpss_diff_ct[0], 'p': kpss_diff_ct[1]},
        }
    }

# Perform Zivot-Andrews test for inflation series
inflation_series = clean_series['FP.CPI.TOTL.ZG'].values
# Use maxlag = 1 for more stable estimate
za_result = zivot_andrews(inflation_series, maxlag=1, regression='c')
# zivot_andrews returns (test_statistic, p_value, critical_values, best_lag, break_index)
za_statistic = za_result[0]
za_pvalue = za_result[1]
za_break_idx = za_result[4]  # break_index is the last element of the returned tuple
za_break_year = clean_series['FP.CPI.TOTL.ZG'].index[int(za_break_idx)]

# Save Zivot-Andrews results
za_info = {
    'statistic': za_statistic,
    'p_value': za_pvalue,
    'break_year': int(za_break_year)
}

# Save results to a file for the report to read
import json
with open('/home/oai/share/results.json', 'w') as f:
    json.dump({'results': results, 'za': za_info}, f, indent=2)

