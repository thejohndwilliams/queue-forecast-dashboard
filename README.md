# Queue Demand Forecast Dashboard

Browser-based forecasting tool that runs **Holt-Winters exponential smoothing** and **Beta-Binomial escalation rate estimation** entirely client-side. No server, no dependencies beyond two CDN scripts (Chart.js + PapaParse).

## How to Use

1. Open `Queue_Forecast_Dashboard.html` in any modern browser.
2. Drag a CSV with columns `date, A_t, C_t, E_t, B_t` onto the upload zone.
3. The dashboard generates a 30-day indexed forecast instantly.

Optional columns: `vertical` (for breakdown by account type) and `issue_tag`.

## Views

- **Forecast** — Arrivals vs. closures indexed against historical baseline, with backlog trajectory.
- **Verticals** — Proportional share allocation across account types.
- **Seasonality** — Day-of-week demand patterns and closure efficiency ratios.
- **Data Table** — Sortable forecast detail with net flow indicators.

## Parameters

Smoothing constants (alpha, beta, gamma) and forecast horizon are adjustable in the dashboard header. Default: `alpha=0.2, beta=0.1, gamma=0.2, horizon=30, season_period=7`.

## Sanitization

All displayed values are **indices, ratios, or percentages** — never absolute ticket counts. Safe for external sharing, portfolio use, or presentation outside secure environments.

## Files

| File | Purpose |
|------|---------|
| `Queue_Forecast_Dashboard.html` | Self-contained interactive dashboard |
| `queue_forecast.py` | Python forecasting engine (source implementation) |
| `Sample_Daily_Table.csv` | 174 days of synthetic test data |

## Tech

- **Forecasting**: Holt-Winters additive seasonal (m=7) with trend damping
- **Escalation**: Beta-Binomial conjugate prior with exponential decay weighting
- **Frontend**: Vanilla JS + Chart.js 4.5 + PapaParse 5.4
- **Zero dependencies**: Single HTML file, no build step, no server
