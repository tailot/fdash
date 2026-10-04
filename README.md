# Integrated Financial Analysis Master Dashboard

Interactive web application built with **Python** and **Streamlit** that acts as a **Financial Analysis Master Dashboard**.

The application **natively** calculates quantitative and volumetric metrics (including the TimesFM-3 multi-day quantile probabilistic forecast). It also uses the **Gemini 2.5 Flash** API (`gemini-2.5-flash`) to generate qualitative analyses on news sentiment and an integrated financial verdict.

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
4. **Enrichment (Gemini 2.5 Flash or local heuristic)**: `News Sentiment` and `Summary / Verdict`, with the same analysis thresholds. Missing data are marked as "n/a", never filled with fictitious values. Note: Gemini does not have access to real-time news.
5. **Report**: integrated table (*outer* join on `Ticker`), download `report_finale_integrato.csv`, comparative charts and, for native calculation, detail per ticker (Volume Profile, tables).

> ⚠️ Note: buy/sell is an **estimation** from candles (not real order flow); "in loss/gain" considers only the volumes of the selected period; the "today's top N" universe has survivorship bias. This is not investment advice.

---

## 🚀 Requirements and Local Installation (Without Docker)

### 1. Prerequisites
- **Python 3.10+** installed on the system.
- (Optional) Google Gemini API key to enable real-time AI analysis.

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

## ⚙️ Application Usage

1. **Enter Gemini API Key (optional):**
   - In the sidebar, enter your API key to use the `gemini-2.5-flash` model. If left empty, the app will generate a fallback integrated heuristic analysis.
2. **Set parameters** in the sidebar: tickers (manual list or top N Nasdaq), horizon, buy/sell method, days, etc., then click **"Run Native Analysis"**.
3. **Export the Report:**
   - Click the **"Download report_finale_integrato.csv"** button to export the integrated data.

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
├── gemini_enrichment.py     # Gemini 2.5 Flash API integration
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
