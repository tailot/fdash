# Copyright (c) 2026 Vincenzo Tilotta
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

"""
Database Persistence Engine for Financial Analysis Dashboard.

Uses SQLite to persist generated analysis runs, including quantitative forecasts,
volume microstructure, parameters, and metadata.
"""

import sqlite3
import json
import os
import datetime
import numpy as np
import pandas as pd

DB_FILE = os.environ.get("DASHBOARD_DB_PATH", "analysis_history.db")


def _json_default(obj):
    if isinstance(obj, (datetime.datetime, datetime.date, pd.Timestamp)):
        return obj.isoformat()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    return str(obj)


def get_connection(db_file: str = DB_FILE) -> sqlite3.Connection:
    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_file: str = DB_FILE):
    """Initializes SQLite database and creates tables if they don't exist."""
    with get_connection(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                run_label TEXT NOT NULL,
                universe_type TEXT NOT NULL,
                n_tickers INTEGER NOT NULL,
                horizon INTEGER NOT NULL,
                model_used TEXT NOT NULL,
                params_json TEXT NOT NULL,
                quant_json TEXT NOT NULL,
                vol_json TEXT NOT NULL,
                info_json TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS triggered_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                condition TEXT NOT NULL,
                target REAL NOT NULL,
                price REAL NOT NULL
            )
        """)
        conn.commit()


def save_run(nat_data: dict, params: dict, db_file: str = DB_FILE) -> int:
    """
    Saves a native analysis run to SQLite database.
    nat_data contains 'quant', 'info', 'vol', 'errori', 'metodo'.
    """
    init_db(db_file)

    df_quant = nat_data.get("quant")
    quant_json = df_quant.to_json(orient="records", date_format="iso") if isinstance(df_quant, pd.DataFrame) else "[]"

    vol_rows = nat_data.get("vol", [])
    vol_serializable = []
    for r in vol_rows:
        r_copy = {}
        for k, v in r.items():
            if isinstance(v, pd.DataFrame):
                r_copy[k] = v.to_dict(orient="records")
            else:
                r_copy[k] = v
        vol_serializable.append(r_copy)
    vol_json = json.dumps(vol_serializable, default=_json_default)

    info = nat_data.get("info", {})
    info_clean = {}
    for k, v in info.items():
        if k == "prezzi":  # Exclude raw prices DataFrame from main run record
            continue
        info_clean[k] = v
    info_json = json.dumps(info_clean, default=_json_default)

    timestamp = str(info.get("run_ts", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    universe_type = str(info.get("fonte", "unknown"))
    n_tickers = len(df_quant) if isinstance(df_quant, pd.DataFrame) else 0
    model_used = str(df_quant["Modello"].iloc[0]) if isinstance(df_quant, pd.DataFrame) and "Modello" in df_quant else "Unknown"
    horizon = params.get("horizon", 5)

    ticker_list = []
    if isinstance(df_quant, pd.DataFrame) and "Ticker" in df_quant.columns:
        ticker_list = df_quant["Ticker"].tolist()

    is_manual_list = (universe_type == "manual list" or params.get("is_manual", False))
    if is_manual_list and 0 < len(ticker_list) < 6:
        symbols_str = ", ".join(ticker_list)
        run_label = f"{timestamp} | [{symbols_str}] ({n_tickers} stocks) | {model_used}"
    else:
        run_label = f"{timestamp} | {universe_type} ({n_tickers} stocks) | {model_used}"

    with get_connection(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO analysis_runs (
                timestamp, run_label, universe_type, n_tickers, horizon, model_used,
                params_json, quant_json, vol_json, info_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp, run_label, universe_type, n_tickers, horizon, model_used,
            json.dumps(params, default=_json_default), quant_json, vol_json, info_json
        ))
        conn.commit()
        return cursor.lastrowid


def list_runs(db_file: str = DB_FILE) -> list:
    """Returns a list of saved runs with metadata (ID, timestamp, label, ticker count, etc.)."""
    init_db(db_file)
    with get_connection(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, timestamp, run_label, universe_type, n_tickers, horizon, model_used
            FROM analysis_runs
            ORDER BY id DESC
        """)
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def get_run(run_id: int, db_file: str = DB_FILE) -> dict:
    """Retrieves a specific run by ID and reconstructs DataFrames and native analysis structure."""
    init_db(db_file)
    with get_connection(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM analysis_runs WHERE id = ?", (run_id,))
        row = cursor.fetchone()
        if not row:
            return None

        row_dict = dict(row)
        quant_list = json.loads(row_dict["quant_json"])
        df_quant = pd.DataFrame(quant_list) if quant_list else pd.DataFrame()

        vol_list = json.loads(row_dict["vol_json"])
        vol_rows = []
        for r in vol_list:
            r_copy = dict(r)
            if "zone_top5" in r_copy and isinstance(r_copy["zone_top5"], list):
                r_copy["zone_top5"] = pd.DataFrame(r_copy["zone_top5"])
            if "dettaglio_giorni" in r_copy and isinstance(r_copy["dettaglio_giorni"], list):
                r_copy["dettaglio_giorni"] = pd.DataFrame(r_copy["dettaglio_giorni"])
            vol_rows.append(r_copy)

        info = json.loads(row_dict["info_json"])
        params = json.loads(row_dict["params_json"])

        nat_data = {
            "quant": df_quant,
            "info": info,
            "vol": vol_rows,
            "errori": [],
            "metodo": params.get("metodo", "clv")
        }

        return {
            "metadata": row_dict,
            "params": params,
            "nativo": nat_data
        }


def delete_run(run_id: int, db_file: str = DB_FILE) -> bool:
    """Deletes a run record by ID."""
    init_db(db_file)
    with get_connection(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM analysis_runs WHERE id = ?", (run_id,))
        conn.commit()
        return cursor.rowcount > 0


def save_alert_event(event: dict, db_file: str = DB_FILE) -> int:
    """Saves a triggered alert event to the database."""
    init_db(db_file)
    ts = str(event.get("time") or event.get("ts") or datetime.datetime.now().strftime("%H:%M:%S"))
    with get_connection(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO triggered_alerts (timestamp, symbol, condition, target, price)
            VALUES (?, ?, ?, ?, ?)
        """, (
            ts,
            str(event.get("symbol") or ""),
            str(event.get("condition") or ""),
            float(event.get("target") or 0.0),
            float(event.get("price") or 0.0)
        ))
        conn.commit()
        return cursor.lastrowid


def list_alert_events(db_file: str = DB_FILE) -> list:
    """Lists saved triggered alert events from the database."""
    init_db(db_file)
    with get_connection(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, timestamp as time, symbol, condition, target, price
            FROM triggered_alerts
            ORDER BY id DESC
        """)
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def clear_alert_events(db_file: str = DB_FILE) -> bool:
    """Deletes all triggered alert records from the database."""
    init_db(db_file)
    with get_connection(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM triggered_alerts")
        conn.commit()
        return True
