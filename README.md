# Dashboard Master di Analisi Finanziaria Integrata

Applicazione web interattiva realizzata in **Python** e **Streamlit** che agisce da **Dashboard Master di Analisi Finanziaria**.

L'applicazione calcola **nativamente** le metriche quantitative e volumetriche estratte dai Colab specialistici (inclusa la previsione probabilistica quantilica multi-giorno TimesFM-3) o, in alternativa, supporta l'ingestion tramite caricamento di file CSV o lettura da cartella locale. Utilizza inoltre l'API di **Gemini 2.5 Flash** (`gemini-2.5-flash`) per generare analisi qualitative sul sentiment delle notizie e un verdetto finanziario integrato.

---

## 🌟 Funzionalità Principali

1. **Calcolo Nativo (Motore Python):**
   - **Analisi Quantitativa (TimesFM-3 / Proiezioni Quantiliche):** Calcola Ticker, Ultimo Prezzo, Trend (`BUY`/`SELL`/`EQUILIBRIO`), Volatilità Annualizzata (`Sigma %`), e le proiezioni probabilistiche multi-giorno (`Data previsione`, `P10`, `Mediana`, `P90`, `Rend. mediano %`, `Incertezza (sigma) %`).
   - **Analisi Microstruttura dei Volumi (1m Intraday):** Scarica i dati ad 1 minuto e stima il Delta Volumi Intra (`Buy - Sell %`) usando i metodi *CLV*, *Candela* o *Tick*, e calcola la `% Trader in Perdita` tramite il Volume Profile.
2. **Ingestion Flessibile dei Dati:**
   - **Calcolo Nativo:** Genera automaticamente tutte le metriche per i ticker desiderati.
   - **Caricamento File CSV:** Permette di caricare separatamente l'output del Colab 1 (TimesFM-3) e del Colab 2 (Volumi).
   - **Lettura da Cartella Locale:** Legge automaticamente i CSV salvati nella cartella `./data/`.
3. **Arricchimento tramite Gemini API (`gemini-2.5-flash`):**
   - Interroga il modello Gemini di Google per generare le colonne qualitative:
     - **Sentiment News:** Sintesi del sentiment e contesto di mercato.
     - **Sintesi / Verdetto:** Giudizio finale integrato tra modello predittivo, volumi intraday e sentiment.
4. **Visualizzazione e Download:**
   - Tabella pulita con intestazioni complete:
     `Ticker` | `Ultimo Prezzo` | `Trend TimesFM` | `Sigma %` | `Data previsione` | `P10` | `Mediana` | `P90` | `Rend. mediano %` | `Incertezza (sigma) %` | `Delta Volumi Intra` | `% Trader in Perdita` | `Sentiment News` | `Sintesi / Verdetto`
   - Pulsante per scaricare il report completo in CSV (`report_finale_integrato.csv`).
   - Grafici interattivi e comparativi dei volumi e dei trend.

---

## 🚀 Requisiti e Installazione Locale (Senza Docker)

### 1. Prerequisiti
- **Python 3.10+** installato sul sistema.
- (Opzionale) Chiave API Google Gemini per abilitare l'analisi AI in tempo reale.

### 2. Installazione Dipendenze
Clona o scarica la repository, quindi installa i pacchetti richiesti:

```bash
pip install -r requirements.txt
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
2. **Scegli la Modalità di Funzionamento:**
   - **Calcolo Nativo (Consigliato):** Inserisci i ticker separati da virgola (es. `MSTR, AAPL, NVDA, TSLA, MSFT`), seleziona il metodo di stima dei volumi (*CLV*, *Candela*, *Tick*) e premi **"Esegui Analisi Nativa"**.
   - **Carica File CSV:** Trascina i due file CSV generati dai Colab.
   - **Leggi da Cartella Locale:** Legge i file `colab1_timesfm.csv` e `colab2_volumi.csv` posizionati nella cartella `./data/`.
3. **Esporta il Report:**
   - Clicca sul pulsante **"Scarica report_finale_integrato.csv"** per esportare i dati integrati.

---

## 🧪 Esecuzione dei Test Unitari

Per verificare il corretto funzionamento di tutti i moduli (motore quantitativo, motore volumi, arricchimento AI e join dati):

```bash
pytest test_modules.py
```

---

## 📁 Struttura del Progetto

```
.
├── app.py                   # Dashboard principale Streamlit
├── quant_engine.py          # Motore di analisi quantitativa (TimesFM / Trend / Sigma % / Quantili)
├── volume_engine.py         # Motore microstruttura volumi intraday e Volume Profile
├── gemini_enrichment.py     # Integrazione API Gemini 2.5 Flash
├── test_modules.py          # Suite di test unitari
├── requirements.txt         # Dipendenze Python
├── Dockerfile               # Configurazione per la creazione dell'immagine Docker
├── docker-compose.yml       # Configurazione Docker Compose
├── data/                    # Cartella contenente i CSV locali di esempio
└── README.md                # Guida all'uso del progetto
```
