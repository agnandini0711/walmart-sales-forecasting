# Weekly Sales Forecasting: 45 Walmart Stores

Eight-week-ahead sales forecasts for 45 Walmart stores. I compared a seasonal naive baseline, ETS, ARIMA, an ETS+ARIMA average and gradient boosting, and checked whether prediction intervals cover the real values as often as they should.

**Main result:** the average of ETS and ARIMA had the lowest error on a held-out 8-week window: **2.6% WAPE**, against **4.9%** for "same week last year". In the three backtests ETS and the average tied at 4.1%, against 5.9% for seasonal naive. Gradient boosting scored 4.5% on the holdout and 5.2% in the backtests, and did not beat the statistical models.

**Read this with care:** the holdout and all three backtests fall outside November and December, and the holdout is a single 8-week window. These results say nothing about holiday-peak performance, and the small gaps between the top models may not be robust.

## Data

- Weekly sales for 45 stores, 2010-02-05 to 2012-10-26 (143 weeks each, 6,435 rows).
- Source: a public Walmart weekly sales dataset, downloaded from [this GitHub copy](https://github.com/Chulumanco135/Walmart-sales-analysis/blob/main/data/Walmart_Sales.csv). I could not trace the original publisher. To run the notebook, download the file and place it at `data/Walmart_Sales.csv`.
- Columns: Store, Date, Weekly_Sales, Holiday_Flag, Temperature, Fuel_Price, CPI, Unemployment.
- Checks run: no missing values, no duplicate store-weeks, no negative sales, every gap between weeks is exactly 7 days.
- 269 store-weeks are unusually high or low, and 89% of them fall in November and December, so they are mostly consistent with holiday peaks. I kept them, because removing them would delete the demand a forecaster has to predict.
- The dataset's `Holiday_Flag` misses three of the five biggest total-sales weeks (24 Dec 2010, 23 Dec 2011 and 17 Dec 2010), so it is a weak signal.

## Method

**Split.** Forecast horizon is 8 weeks. The last 8 weeks (7 September to 26 October 2012) are the test set; the cutoff is 2012-08-31, leaving 135 training weeks per store.

**Backtests.** Three rolling-origin backtests inside the training period (origins 8, 16 and 24 weeks before the cutoff, covering roughly 23 March to 31 August 2012). At each origin I trained on earlier data only and forecast the next 8 weeks. Models were chosen on the backtests; the test set was scored once and not used for any choice.

**Models**

| Model | Description |
| --- | --- |
| Seasonal naive | Same week last year |
| ETS | Additive Holt-Winters exponential smoothing (statsmodels `ExponentialSmoothing`) with 52-week seasonality, with and without trend |
| ARIMA | Fitted on the year-on-year difference, forecast = last year's value + forecast change. Candidate orders were shortlisted by AIC summed over stores (compared only within the same differencing); the final order, (0,1,1), was chosen on the backtests |
| Ensemble | Average of ETS (no trend) and ARIMA(0,1,1) |
| Gradient boosting | `HistGradientBoostingRegressor` on lagged sales and calendar features, untuned |

**Avoiding leakage.** Lag features are at least 8 weeks old (`lag_8`, `lag_52`, and the average of the 4 weeks ending 8 weeks ago), because that is the newest data available when forecasting 8 weeks ahead. Each store's sales are scaled by that store's average over **training weeks only**; in the gradient boosting backtests the scaling is recomputed for each fold. As a check, the ARIMA order search was repeated using only the 111 weeks before the earliest backtest, and (0,1,1) was still the best `d=1` order.

**Metrics.** WAPE = sum of |actual - forecast| / sum of actual. Bias = sum of (forecast - actual) / sum of actual (negative means forecasts are too low). MASE = absolute error divided by each store's in-sample seasonal-naive error (training weeks only).

## Results

| Model | Backtest WAPE (mean of 3 folds) | Holdout WAPE | Holdout bias | Holdout MASE |
| --- | --- | --- | --- | --- |
| ETS + ARIMA average | 4.1% | 2.6% | -0.8% | 0.455 |
| ETS | 4.1% | 2.9% | -2.0% | 0.504 |
| ARIMA(0,1,1) | 4.3% | 3.0% | +0.4% | 0.523 |
| Gradient boosting | 5.2% | 4.5% | +1.2% | 0.736 |
| Seasonal naive | 5.9% | 4.9% | -1.7% | 0.787 |

- The ensemble is the best observed result on the holdout, not a proven winner: it tied with ETS in the backtests, and the 0.3-point holdout gap comes from one 8-week window.
- Every model scores lower error on the holdout than in the backtests, so the holdout appears to be the easier window. MASE values below 1 should not be read as "better than a typical naive forecast", because the scale includes holiday peaks while the holdout does not.
- Gradient boosting beats seasonal naive in the backtests but trails ETS and the ensemble in every fold. This is a simple untuned model and says nothing about machine learning in general.

**Prediction intervals.** Intervals are built from the relative errors of the backtests (forecast x (1 + error quantile)) and checked on the holdout, which was not used to build them.

| Model | Target | Holdout coverage | Width (% of forecast) |
| --- | --- | --- | --- |
| ETS | 80% | 88.9% | 12.5 |
| ETS | 90% | 96.9% | 17.1 |
| Ensemble | 80% | 93.1% | 12.2 |
| Ensemble | 90% | 98.1% | 17.5 |

- Coverage is above target everywhere, so the intervals are conservative (wider than needed) on this window. A likely reason is that the backtest windows were more volatile than the calm holdout.
- The over-coverage comes almost entirely from the lower bound. The upper bounds are close to nominal: actual sales exceeded the ETS 80% interval's upper end in 10.3% of store-weeks (nominal 10%) and the 90% interval's in 2.8% (nominal 5%).
- Coverage was not tested on holiday peaks, and the holdout covers only 8 distinct weeks, so these figures should not be assumed to hold elsewhere.
- Scaling each store's width by its backtest error gives narrower intervals for easy stores and wider ones for hard stores at the same average width and overall coverage. I kept the single pooled width as the main result.

## Other findings

- **Stationarity.** ADF and KPSS agree that the seasonal difference alone leaves many stores non-stationary, and that one further regular difference makes almost all stationary (42 of 45 under both tests). This supports the `d=1` ARIMA order chosen on the backtests.
- **Peak weeks.** Seasonal-naive error for November (5.1%) and December (6.4%) 2011 was not unusually large compared with other months (4.5% to 5.9%, with April highest at 8.9% because Easter moved). This is one Q4 and one baseline, and says nothing about ETS, ARIMA or the intervals at the peak.
- **Store 35** has the largest holdout error (7.1% ETS WAPE) and is the most volatile store. Its year-on-year growth swung from about -29% in early 2011 to +8% by October 2012. Seasonal naive and ETS, which copy last year's pattern, under-forecast it in all 8 holdout weeks, while ARIMA followed the turn (1.8%). The data cannot say what caused the swings.

## Planning illustration

This is illustrative only; the data has no inventory levels, costs, lead times or service targets, and "actual above plan" is not a stockout rate. For ETS on the holdout:

| Plan | Actual above plan | Average (plan - actual) / actual |
| --- | --- | --- |
| Point forecast | 71.9% of store-weeks | -1.9% |
| 80% upper bound | 10.3% | +4.2% |
| 90% upper bound | 2.8% | +7.1% |

Planning above the point forecast cuts how often demand exceeds the plan but raises the average overshoot. ETS ran about 2% low on this window, so ordering to its point forecast would have fallen short in most store-weeks.

## Limitations

- The holdout and all three backtests are outside November and December, so these results say nothing about the holiday peak. With under two seasonal cycles of earlier history, ETS cannot be fitted for a Q4 2011 test.
- A single 8-week holdout is a small sample. The backtest scores are also slightly optimistic, because the best model was chosen from several candidates on the same three folds.
- Gradient boosting was not tuned and uses only a few features.
- I did not use Temperature, Fuel_Price, CPI or Unemployment, because their future values are not known 8 weeks ahead. The models use only sales history and the holiday flag.
- The main intervals use the same relative width for every store.
- This project predicts what will happen. It does not estimate what causes sales to change.

## How to run

Results were produced with Python 3.14.0, pandas 3.0.6, numpy 2.5.4, scikit-learn 1.9.1, statsmodels 0.15.0 and matplotlib 3.11.2 (pinned in `requirements.txt`).

```bash
pip install -r requirements.txt
jupyter notebook walmart_sales.ipynb
```

The notebook fits ETS and ARIMA for each of the 45 stores across several models and folds, so a full run can take several minutes. Saved outputs are included in the notebook.

## Files

- `walmart_sales.ipynb`: the full analysis, with outputs
- `requirements.txt`: Python packages
