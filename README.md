# Weekly Sales Forecasting: 45 Walmart Stores

Eight-week-ahead sales forecasts for 45 Walmart stores. I compared a seasonal naive baseline, ETS, ARIMA, an ETS+ARIMA average and gradient boosting, and checked whether prediction intervals cover the real values as often as they should.

**Main result:** the average of ETS and ARIMA had the lowest error on a held-out 8-week window: **2.6% WAPE**, against **4.9%** for "same week last year". Gradient boosting scored 4.5% and did not beat the statistical models.

## Data

- Weekly sales for 45 stores, 2010-02-05 to 2012-10-26 (143 weeks each, 6,435 rows).
- Source: a public Walmart weekly sales dataset (6,435 rows), downloaded from [this GitHub copy](https://github.com/Chulumanco135/Walmart-sales-analysis/blob/main/data/Walmart_Sales.csv). I could not trace the original publisher. To run the notebook, download the file and place it at `data/Walmart_Sales.csv`.
- Columns: Store, Date, Weekly_Sales, Holiday_Flag, Temperature, Fuel_Price, CPI, Unemployment.
- Checks run: no missing values, no duplicate store-weeks, no negative sales, every gap between weeks is exactly 7 days.
- 269 store-weeks are unusually high or low, and 89% of them fall in November and December. They are real holiday peaks, so I kept them.
- The dataset's `Holiday_Flag` misses most of the biggest pre-Christmas weeks (for example 24 Dec 2010 and 23 Dec 2011), so it is a weak signal.

## Method

**Split.** Forecast horizon is 8 weeks. The last 8 weeks (September to October 2012) are the test set; the cutoff is 2012-08-31, leaving 135 training weeks per store.

**Backtests.** Three rolling-origin backtests inside the training period (origins 8, 16 and 24 weeks before the cutoff). At each origin I trained on earlier data only and forecast the next 8 weeks. The test set was not used for model choice.

**Models**

| Model | Description |
| --- | --- |
| Seasonal naive | Same week last year |
| ETS | Exponential smoothing with additive 52-week seasonality, with and without trend |
| ARIMA | Fitted on the year-on-year difference, forecast = last year's value + forecast change. Order chosen by AIC summed over stores (compared only within the same differencing) |
| Ensemble | Average of ETS (no trend) and ARIMA(0,1,1) |
| Gradient boosting | `HistGradientBoostingRegressor` on lagged sales and calendar features |

**Avoiding leakage.** Lag features are at least 8 weeks old (`lag_8`, `lag_52`, and the average of the 4 weeks ending 8 weeks ago), because that is the newest data available when forecasting 8 weeks ahead. Each store's sales are scaled by that store's average over **training weeks only**.

**Metrics.** WAPE = sum of |actual - forecast| / sum of actual. Bias = sum of (forecast - actual) / sum of actual (negative means forecasts are too low).

## Results

Held-out 8 weeks:

| Model | WAPE | Bias |
| --- | --- | --- |
| ETS + ARIMA average | 2.6% | -0.8% |
| ETS | 2.9% | -2.0% |
| ARIMA(0,1,1) | 3.0% | +0.4% |
| Gradient boosting | 4.5% | +1.2% |
| Seasonal naive | 4.9% | -1.7% |

In the three backtests, ETS alone and the ensemble tie at 4.1% mean WAPE, and seasonal naive scores 5.9%.

**Prediction intervals.** Intervals are built from the relative errors of the backtests (forecast x (1 + error quantile)). Share of test store-weeks that fell inside the interval:

| Model | Target | Actual coverage |
| --- | --- | --- |
| ETS | 80% | 88.9% |
| ETS | 90% | 96.9% |
| Ensemble | 80% | 93.1% |
| Ensemble | 90% | 98.1% |

Coverage is above target, so the intervals are safe but wider than necessary.

## Limitations

- The test window and all three backtests are outside November and December, so these results say nothing about the holiday peak.
- A single 8-week holdout is a small sample. The 0.3-point gap between the ensemble and ETS may be luck.
- Gradient boosting was not tuned and uses only a few features.
- I did not use the Temperature, Fuel_Price, CPI and Unemployment columns; the models use only sales history and the holiday flag.
- Intervals use the same relative width for every store.
- This project predicts what will happen. It does not estimate what causes sales to change.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook walmart_sales.ipynb
```

The notebook loops over 45 stores and several models, so a full run can take a few minutes.

## Files

- `walmart_sales.ipynb`: the full analysis
- `requirements.txt`: Python packages
