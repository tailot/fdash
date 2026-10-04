# Integrated Master Financial Analysis Dashboard

Interactive web application built in **Python** and **Streamlit** that serves as a **Master Financial Analysis Dashboard**.

The application natively calculates quantitative and volumetric metrics (including TimesFM-3 multi-day probabilistic quantile forecasting). It also utilizes the **Gemini 2.5 Flash** API (`gemini-2.5-flash`) to generate qualitative news sentiment analysis and an integrated financial verdict.

The user interface supports **multilingual localization** in 5 major global languages: **English, Italian, Spanish, Chinese, and French**.

---

## 🌟 Key Features

Native calculations include:

1. **Quantitative Forecasting — `quant_engine.py`**
   - Universe: top N NASDAQ stocks by market cap (Nasdaq screener, Wikipedia Nasdaq-100 fallback), dual share classes eliminated; or manual list.
   - Adjusted prices over 5 years on a common calendar (stocks with <99% history discarded, isolated gaps filled).
   - **TimesFM-3 on log returns** (1024-day context window): deciles → median, P10, P90; step sigma `(P90−P10)/(2·Z80)`, cumulative sigma as root sum of squares; `P10/Median/P90 = last · exp(μ ∓ Z80·σ)`.
   - Output: `Last Price`, `Forecast Date`, `P10`, `Median`, `P90`, `Median Return %`, `Uncertainty (sigma) %`, sorted by median return. Additional dashboard columns: `TimesFM Trend` (BUY/SELL/EQUILIBRIO if |μ| > threshold·σ, adjustable threshold) and `Sigma %` (annualized volatility).
   - **Without `timesfm3` installed**, the app uses the *Naive (0%)* baseline and flags it (`Model` column + warning): no fabricated trends.
2. **1-Minute Volumes — `volume_engine.py`**
   - 1-minute intraday data (~8 days), 1-5 days, `end_date`, regular trading hours, `candle` / `clv` / `tick` methods, equilibrium threshold.
   - Volume Profile on typical price (H+L+C)/3: `% in loss/profit`, POC, VWAP, Value Area (VAL/VAH), most crowded zones above/below price, top 5 zones, daily breakdown, textual conclusions (60% / 40% thresholds).
3. **Multilingual UI & AI Enrichment — `i18n.py` & `gemini_enrichment.py`**:
   - Supports 5 widely spoken languages selectable directly from the sidebar: English 🇬🇧, Italian 🇮🇹, Spanish 🇪🇸, Chinese 🇨🇳, French 🇫🇷.
   - Qualitative columns `Sentiment News` and `Summary / Verdict` are dynamically generated in the user's selected language using Gemini 2.5 Flash or localized heuristic fallback rules. Missing data is marked as "N/A" and never hallucinated.
4. **Data**: The dashboard calculates everything natively (no CSV uploading or folder reads required).
5. **Report**: Integrated table (outer join on `Ticker`), CSV report download `report_finale_integrato.csv`, comparative charts, and per-ticker Volume Profile detail breakdowns.

> ⚠️ Note: Buy/sell is an **estimate** from 1-minute candles (not actual order flow); "in loss/profit" considers only volumes in the period; today's top N universe carries survivorship bias. Statistical analysis, not investment advice.

---

## 🚀 Requirements & Local Setup (Without Docker)

### 1. Prerequisites
- **Python 3.10+** installed on the system.
- (Optional) Google Gemini API Key to enable AI news sentiment enrichment.

### 2. Dependency Installation
Clone or download the repository, then install the required packages:

```bash
pip install -r requirements.txt            # without TimesFM-3 (Naive baseline)
pip install -r requirements-timesfm.txt    # optional: real TimesFM-3 model
```

### 3. Running the Application
To launch the Streamlit dashboard locally:

```bash
streamlit run app.py
```

The application will open automatically in your browser at `http://localhost:8501`.

---

## 🐳 Running via Docker

You can run the application isolated in a container using **Docker** or **Docker Compose**.

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

## ⚙️ Using the Application

1. **Select Language:**
   - Choose your preferred language (English, Italian, Spanish, Chinese, French) from the sidebar language dropdown.
2. **Enter Gemini API Key (Optional):**
   - In the sidebar, enter your API key to enable `gemini-2.5-flash`. If left empty, the app will generate integrated local heuristic analysis.
3. **Configure Parameters:**
   - Set universe parameters (manual list or top N Nasdaq), forecast horizon, volume estimation method, analyzed days, etc., then click **"Run Native Analysis"**.
4. **Export Report:**
   - Click the **"Download report_finale_integrato.csv"** button to export integrated results.

---

## 🧪 Running Unit Tests

Tests are offline (synthetic data and mocked forecaster, no live network calls to Yahoo/Nasdaq) and verify calculation formulas and multilingual i18n support:

```bash
python3 -m pytest test_modules.py
```

---

## 📁 Project Structure

```
.
├── app.py                   # Main Streamlit dashboard UI
├── i18n.py                  # Internationalization module (EN, IT, ES, ZH, FR)
├── quant_engine.py          # Quantitative pipeline (universe, prices, log returns, quantiles)
├── volume_engine.py         # Volume microstructure pipeline (buy/sell estimation, Volume Profile)
├── gemini_enrichment.py     # Gemini 2.5 Flash API integration & multilingual fallback
├── test_modules.py          # Unit test suite
├── requirements.txt         # Standard Python dependencies
├── requirements-timesfm.txt # Optional dependencies for TimesFM-3
├── Dockerfile               # Docker build configuration
├── docker-compose.yml       # Docker Compose setup
└── README.md                # Project documentation
```
