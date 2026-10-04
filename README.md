# Dashboard Master di Analisi Finanziaria Integrata

Applicazione web interattiva realizzata in **Python** e **Streamlit** che agisce da **Dashboard Master di Analisi Finanziaria**.

L'applicazione calcola **nativamente** le metriche quantitative e volumetriche (inclusa la previsione probabilistica quantilica multi-giorno TimesFM-3). Utilizza inoltre l'API di **Gemini 2.5 Flash** (`gemini-2.5-flash`) per generare analisi qualitative sul sentiment delle notizie e un verdetto finanziario integrato.

---

## 🌟 Funzionalità Principali

Il calcolo nativo comprende:

1. **Previsione quantitativa — `quant_engine.py`**
   - Universo: i top N titoli NASDAQ per market cap (screener Nasdaq, fallback Wikipedia Nasdaq-100), doppie classi eliminate; oppure lista manuale.
   - Prezzi aggiustati 5y su calendario comune (titoli con storico <99% scartati, buchi isolati riempiti).
   - **TimesFM-3 sui rendimenti log** (contesto 1024 gg): decili → mediana, P10, P90; sigma per passo `(P90−P10)/(2·Z80)`, cumulata come radice della somma dei quadrati; `P10/Mediana/P90 = ultimo · exp(μ ∓ Z80·σ)`.
   - Output: `Ultimo Prezzo`, `Data previsione`, `P10`, `Mediana`, `P90`, `Rend. mediano %`, `Incertezza (sigma) %`, ordinati per rendimento mediano. Colonne aggiuntive della dashboard: `Trend TimesFM` (BUY/SELL/EQUILIBRIO se |μ| > soglia·σ, soglia regolabile) e `Sigma %` (volatilità annualizzata).
   - **Senza `timesfm3` installato** l'app usa la baseline *Naive (0%)* e lo segnala (colonna `Modello` + avviso): niente trend inventati.
2. **Volumi al minuto — `volume_engine.py`**
   - Dati a 1 minuto (~8 giorni), 1-5 giorni, `DATA_FINE`, orari regolari, metodi `candela` / `clv` / `tick`, soglia di equilibrio.
   - Volume Profile su prezzo tipico (H+L+C)/3: `% in perdita/guadagno`, POC, VWAP, area di valore (VAL/VAH), zone più affollate sopra/sotto, top 5 zone, dettaglio per giorno, conclusioni (soglie 60% / 40%).
3. **Dati**: la dashboard calcola tutto nativamente (nessun caricamento di CSV né lettura da cartella).
4. **Arricchimento (Gemini 2.5 Flash o euristica locale)**: `Sentiment News` e `Sintesi / Verdetto`, con le stesse soglie di analisi. I dati mancanti sono "n/d", mai riempiti con valori fittizi. Nota: Gemini non ha accesso alle notizie in tempo reale.
5. **Report**: tabella integrata (join *outer* su `Ticker`), download `report_finale_integrato.csv`, grafici comparativi e, per il calcolo nativo, il dettaglio per ticker (Volume Profile, tabelle).

> ⚠️ Nota: buy/sell è una **stima** dalle candele (non vero order flow); "in perdita/guadagno" considera solo i volumi del periodo; l'universo "top N di oggi" ha survivorship bias. Non è un consiglio di investimento.

---

## 🚀 Requisiti e Installazione Locale (Senza Docker)

### 1. Prerequisiti
- **Python 3.10+** installato sul sistema.
- (Opzionale) Chiave API Google Gemini per abilitare l'analisi AI in tempo reale.

### 2. Installazione Dipendenze
Clona o scarica la repository, quindi installa i pacchetti richiesti:

```bash
pip install -r requirements.txt            # senza TimesFM-3 (baseline Naive)
pip install -r requirements-timesfm.txt    # opzionale: TimesFM-3 reale
```

### 3. Avvio dell'Applicazione
Per lanciare la dashboard Streamlit localmente:

```bash
streamlit run app.py
```

L'applicazione si aprirà automaticamente nel browser all'indirizzo `http://localhost:8501`.

---

## 🐳 Avvio tramite Docker

È possibile avviare l'applicazione in modo isolato ed automatizzato utilizzando **Docker** o **Docker Compose**.

### Opzione A: Con Docker Compose (Consigliato)

1. Avvia il container:
   ```bash
   docker compose up --build
   ```
2. Apri il browser all'indirizzo: `http://localhost:8501`
3. Per fermare il container:
   ```bash
   docker compose down
   ```

### Opzione B: Con Docker CLI

1. Costruisci l'immagine Docker:
   ```bash
   docker build -t financial-dashboard .
   ```
2. Esegui il container:
   ```bash
   docker run -d -p 8501:8501 --name financial_dashboard financial-dashboard
   ```
3. Apri il browser all'indirizzo: `http://localhost:8501`

---

## ⚙️ Uso dell'Applicazione

1. **Inserisci la Chiave API Gemini (opzionale):**
   - Nella barra laterale (*Sidebar*), inserisci la tua API Key per utilizzare il modello `gemini-2.5-flash`. Se lasciata vuota, l'app genererà un'analisi euristica integrata di fallback.
2. **Imposta i parametri** nella barra laterale: ticker (lista manuale o top N Nasdaq), orizzonte, metodo buy/sell, giorni, ecc., poi premi **«Esegui Analisi Nativa»**.
3. **Esporta il Report:**
   - Clicca sul pulsante **"Scarica report_finale_integrato.csv"** per esportare i dati integrati.

---

## 🧪 Esecuzione dei Test Unitari

I test sono offline (dati sintetici e forecaster simulato, nessun accesso a Yahoo/Nasdaq) e verificano le formule di calcolo:

```bash
pytest test_modules.py
```

---

## 📁 Struttura del Progetto

```
.
├── app.py                   # Dashboard principale Streamlit
├── quant_engine.py          # Pipeline quantitativa TimesFM-3 (universo, prezzi, rendimenti log, quantili)
├── volume_engine.py         # Pipeline volumi (buy/sell, Volume Profile, area di valore)
├── gemini_enrichment.py     # Integrazione API Gemini 2.5 Flash
├── test_modules.py          # Suite di test unitari
├── requirements.txt         # Dipendenze Python
├── requirements-timesfm.txt # Dipendenze opzionali per TimesFM-3
├── Dockerfile               # Configurazione per la creazione dell'immagine Docker
├── docker-compose.yml       # Configurazione Docker Compose
└── README.md                # Guida all'uso del progetto
```
