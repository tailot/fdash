# Integrated Financial Analysis Master Dashboard

Interactive web application built with **Python** and **Streamlit** that acts as a **Financial Analysis Master Dashboard**.

The application **natively** calculates quantitative and volumetric metrics (including the TimesFM-3 multi-day quantile probabilistic forecast) and enriches them with local heuristic qualitative analysis.

---

## 🌟 Key Features

The native calculation includes:

1. **Quantitative Forecast — `quant_engine.py`**
   - Universe: top N NASDAQ stocks by market cap (Nasdaq screener, Wikipedia Nasdaq-100 fallback), dual share classes removed; or a manual list.
   - 5-year adjusted prices on a common calendar (tickers with history <99% discarded, isolated missing values filled).
   - **TimesFM-3 on log returns** (1024-day context window): deciles → median, P10, P90; step sigma `(P90−P10)/(2·Z80)`, cumulative as square root of sum of squares; `P10/Median/P90 = last · exp(μ ∓ Z80·σ)`.
   - Output: `Last Price`, `Forecast Date`, `P10`, `Median`, `P90`, `Median Return %`, `Uncertainty (sigma) %`, sorted by median return. Additional dashboard columns: `TimesFM Trend` (BUY/SELL/EQUILIBRIUM if |μ| > threshold·σ, adjustable threshold) and `Sigma %` (annualized volatility).
   - **Without `timesfm3` installed**, the app uses the *Naive (0%)* baseline and flags it (`Model` column + notice): no fabricated trends.
2. **Minute Volumes — `volume_engine.py`**
   - 1-minute data (~8 days), 1-5 days, `DATA_FINE` (End Date), regular hours, `candela` (candle) / `clv` / `tick` methods, equilibrium threshold.
   - Volume Profile on typical price (H+L+C)/3: `% in loss/gain`, POC, VWAP, value area (VAL/VAH), most crowded zones above/below, top 5 zones, daily detail, conclusions (60% / 40% thresholds).
3. **Data**: the dashboard calculates everything natively (no CSV loading or reading from folders).
4. **Local Heuristic Enrichment — `heuristic_enrichment.py`**: `News Sentiment` and `Summary / Verdict`, with defined rule thresholds. Missing data are marked as "N/A", never filled with fictitious values.
5. **Report**: integrated table (*outer* join on `Ticker`), download `report_finale_integrato.csv`, comparative charts and, for native calculation, detail per ticker (Volume Profile, tables).

> ⚠️ Note: buy/sell is an **estimation** from candles (not real order flow); "in loss/gain" considers only the volumes of the selected period; the "today's top N" universe has survivorship bias. This is not investment advice.

---

## 🚀 Requirements and Local Installation (Without Docker)

### 1. Prerequisites
- **Python 3.10+** installed on the system.

### 2. Installing Dependencies
Clone or download the repository, then install the required packages:

```bash
pip install -r requirements.txt            # without TimesFM-3 (Naive baseline)
pip install -r requirements-timesfm.txt    # optional: real TimesFM-3
```

### 3. Running the Application
To launch the Streamlit dashboard locally:

```bash
streamlit run app.py
```

The application will automatically open in your browser at `http://localhost:8501`.

---

## 🐳 Running with Docker

You can run the application in an isolated and automated environment using **Docker** or **Docker Compose**.

### Option A: With Docker Compose (Recommended)

1. Start the container:
   ```bash
   docker compose up --build
   ```
2. Open your browser at: `http://localhost:8501`
3. To stop the container:
   ```bash
   docker compose down
   ```

### Option B: With Docker CLI

1. Build the Docker image:
   ```bash
   docker build -t financial-dashboard .
   ```
2. Run the container:
   ```bash
   docker run -d -p 8501:8501 --name financial_dashboard financial-dashboard
   ```
3. Open your browser at: `http://localhost:8501`

---

## ⚙️ Application Usage & Sidebar Menu Customizations

The application sidebar allows you to customize every aspect of the quantitative forecast, volume microstructure analysis, backtesting, and UI language.

### 🌐 Options & Language Selection
- **Language / Lingua:** Switch the application interface between 5 supported languages:
  - English (`en`)
  - Italian (`it`)
  - Spanish (`es`)
  - Chinese (`zh`)
  - French (`fr`)

### 📊 Universe Configuration
- **Stocks Universe Selection:** Choose between two modes:
  - **Manual List:** Enter custom stock symbols (comma-separated, e.g., `MSTR, AAPL, NVDA, TSLA, MSFT`).
  - **Top N Nasdaq by Market Cap:** Use a slider (range: 20 to 100 stocks, step: 10) to automatically analyze the top Nasdaq tickers by market capitalization.

### 📈 Forecast Settings (TimesFM-3)
- **Horizon (trading days):** Select the forecast horizon between 1 and 20 trading days (default: 10).
- **Trend Threshold (x cumulated sigma):** Set the multiplier (range: 0.0 to 1.0, default: 0.20) for classifying trends. A BUY or SELL trend signal is triggered when the expected return magnitude exceeds `threshold × sigma`.
- **Use TimesFM-3 Checkbox:** Enable/disable TimesFM-3 probabilistic neural network inference. When unchecked or unavailable, the system defaults to the Naive (0%) baseline model.

### 🔍 Volume Microstructure Settings (1-Minute intraday data)
- **Buy/Sell Estimation Method:** Select the algorithm used to split volume into buy vs. sell flows:
  - `clv` (Close Location Value / Accumulation-Distribution formula)
  - `candela` (Candle body direction)
  - `tick` (Tick-by-tick / minute-level price change direction)
- **Analyzed Days:** Set the intraday historical depth to analyze from 1 to 5 trading days.
- **End Date (`YYYY-MM-DD`):** Specify a target historical end date (leave empty to fetch up to the latest available trading day).
- **Regular Hours Only:** Checkbox to filter out pre-market and post-market trading sessions.
- **Equilibrium Threshold %:** Percentage threshold (default: 2.0%) to determine whether volume delta is in equilibrium vs. buyer/seller dominant.
- **Volume Profile Bins:** Set the number of price histogram bins for Volume Profile analysis (range: 10 to 100, default: 30).
- **Value Area %:** Define the percentage of total volume included in the Value Area (VAL–VAH range; default: 70%).
- **Reference Price:** Set a custom reference price (default: `0.0`, which uses the last close price) to evaluate buyers in profit vs. loss.

### 🧪 Backtesting Parameters
- **Number of past dates:** In the backtesting section of the main dashboard, set the number of historical non-overlapping evaluation dates (5 to 100 dates) for point-in-time walk-forward backtesting.

### 🚀 Running Analysis & Exporting
1. Adjust the desired options in the sidebar and click **"Run Native Analysis"**.
2. Review the localized heuristic analysis, integrated report table, comparative charts, and detailed Volume Profile charts per ticker.
3. Click **"Download report_finale_integrato.csv"** to export the integrated dataset.

---

## 🧪 Running Unit Tests

The tests are offline (synthetic data and simulated forecaster, no access to Yahoo/Nasdaq) and verify the calculation formulas:

```bash
pytest test_modules.py
```

---

## 📁 Project Structure

```
.
├── app.py                   # Main Streamlit dashboard
├── quant_engine.py          # TimesFM-3 quantitative pipeline (universe, prices, log returns, quantiles)
├── volume_engine.py         # Volume pipeline (buy/sell, Volume Profile, value area)
├── heuristic_enrichment.py  # Local heuristic enrichment module
├── test_modules.py          # Unit test suite
├── requirements.txt         # Python dependencies
├── requirements-timesfm.txt # Optional dependencies for TimesFM-3
├── Dockerfile               # Docker image configuration
├── docker-compose.yml       # Docker Compose configuration
└── README.md                # Project usage guide
```

---

## ⚠️ Disclaimer

This project was created strictly for educational and research purposes. It does not constitute financial advice, investment recommendations, or an incentive to invest real money. The author assumes no responsibility or liability for any direct or indirect damages or financial losses incurred as a result of using this software.
