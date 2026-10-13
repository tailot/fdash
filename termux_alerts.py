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
Termux Standalone Terminal Price Alert Application.

This module provides a terminal-only price alert monitoring system designed
to run exclusively inside Termux on Android. Audio notifications leverage
Termux capabilities (TTS text-to-speech, media player commands, terminal bells).
"""

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

CONFIG_PATH = os.environ.get("TERMUX_ALERTS_FILE", "termux_alerts_config.json")
POLL_INTERVAL_SECONDS = int(os.environ.get("TERMUX_ALERT_POLL_SECONDS", "60"))
CONDITIONS = (">=", "<=")


# ----------------------------------------------------------------------------- Termux Environment Enforcement
def is_termux_terminal() -> bool:
    """
    Checks if the current environment is Termux running in terminal mode (TTY).
    Returns True only when executed inside Termux with interactive TTY attached.
    """
    in_termux = (
        "TERMUX_VERSION" in os.environ
        or "com.termux" in os.environ.get("PREFIX", "")
        or os.path.exists("/data/data/com.termux")
    )
    is_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    return bool(in_termux and is_tty)


def enforce_termux_environment():
    """Exits the script immediately if not running inside Termux in terminal mode."""
    if not is_termux_terminal():
        sys.stderr.write(
            "\n[ERRORE] Questo script e' progettato per funzionare ESCLUSIVAMENTE "
            "all'interno dell'ambiente Termux in modalita' terminale.\n"
            "Esecuzione interrotta.\n\n"
        )
        sys.exit(1)


# ----------------------------------------------------------------------------- Rules & Persistence
def normalize_rules(rules_list: list) -> list:
    """Cleans and validates rule objects."""
    normalized = []
    if not isinstance(rules_list, list):
        return normalized
    for r in rules_list:
        if not isinstance(r, dict):
            continue
        symbol = str(r.get("symbol") or "").strip().upper()
        condition = str(r.get("condition") or "").strip()
        try:
            target = float(r.get("target"))
        except (ValueError, TypeError):
            continue
        if not symbol or condition not in CONDITIONS or target <= 0:
            continue
        active = bool(r.get("active", True))
        normalized.append({
            "symbol": symbol,
            "condition": condition,
            "target": target,
            "active": active
        })
    return normalized


def load_rules(path: str = None) -> list:
    """Loads alert rules from JSON file."""
    path = path or CONFIG_PATH
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return normalize_rules(data.get("rules", []))
    except (OSError, ValueError):
        return []


def save_rules(rules: list, path: str = None) -> bool:
    """Saves alert rules to JSON file atomically."""
    path = path or CONFIG_PATH
    clean = normalize_rules(rules)
    try:
        folder = os.path.dirname(os.path.abspath(path))
        if folder:
            os.makedirs(folder, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=folder or ".", prefix=".tmp_rules_", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"rules": clean}, f, indent=2)
        os.replace(tmp, path)
        return True
    except OSError:
        return False


def rule_key(rule: dict) -> str:
    """Unique key identifier for a rule."""
    return f"{rule['symbol']}|{rule['condition']}|{rule['target']:.6f}"


def is_triggered(rule: dict, price: float) -> bool:
    """Evaluates whether a price satisfies the rule condition."""
    if rule["condition"] == ">=":
        return price >= rule["target"]
    return price <= rule["target"]


def evaluate_rules(rules: list, prices: dict, fired: set):
    """
    Evaluates rules against prices.
    Returns (events, new_fired_set).
    Rules trigger once when condition is met and re-arm when condition turns false.
    """
    events, new_fired = [], set()
    now_str = datetime.now().strftime("%H:%M:%S")
    for rule in rules:
        if not rule.get("active", True):
            continue
        price = prices.get(rule["symbol"])
        rkey = rule_key(rule)
        if price is None:
            if rkey in fired:
                new_fired.add(rkey)
            continue
        if is_triggered(rule, price):
            new_fired.add(rkey)
            if rkey not in fired:
                events.append({
                    "time": now_str,
                    "symbol": rule["symbol"],
                    "condition": rule["condition"],
                    "target": rule["target"],
                    "price": float(price)
                })
    return events, new_fired


# ----------------------------------------------------------------------------- Price Downloading
def fetch_last_prices(symbols: list) -> dict:
    """Fetches the latest available price for a list of symbols using yfinance."""
    symbols = list(dict.fromkeys(s.strip().upper() for s in symbols if s and s.strip()))
    if not symbols:
        return {}
    try:
        raw = yf.download(
            symbols,
            period="1d",
            interval="1m",
            prepost=True,
            auto_adjust=True,
            progress=False,
            threads=False
        )
        if raw is None or raw.empty:
            return {}
        close = raw["Close"]
        if isinstance(close, pd.Series):
            close = close.to_frame(symbols[0])
        prices = {}
        for s in symbols:
            if s in close.columns:
                series = close[s].dropna()
                if not series.empty:
                    prices[s] = float(series.iloc[-1])
        return prices
    except Exception:
        return {}


# ----------------------------------------------------------------------------- Termux Audio Features
def generate_beep_wav(freq: float = 880.0, beep_s: float = 0.25, gap_s: float = 0.1,
                      repeats: int = 3, rate: int = 22050) -> bytes:
    """Generates a WAV audio buffer for an alert beep sound."""
    t = np.linspace(0, beep_s, int(rate * beep_s), endpoint=False)
    tone = 0.6 * np.sin(2 * np.pi * freq * t) * np.minimum(1, np.minimum(t, beep_s - t) * 80)
    gap = np.zeros(int(rate * gap_s))
    signal = np.concatenate([np.concatenate([tone, gap]) for _ in range(repeats)])
    pcm = (signal * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def play_termux_sound(text_message: str = "Alert prezzo scattato!") -> dict:
    """
    Plays audio alerts utilizing Termux capabilities:
    1. termux-tts-speak (Text-To-Speech)
    2. termux-media-player or command line audio players
    3. Terminal acoustic bell (\a) fallback
    """
    results = {"tts": False, "media": False, "bell": False}

    # 1. Termux TTS Text-To-Speech
    tts_cmd = shutil.which("termux-tts-speak")
    if tts_cmd:
        try:
            subprocess.run([tts_cmd, text_message], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            results["tts"] = True
        except Exception:
            pass

    # 2. Termux Media Player / CLI Audio Player with generated WAV
    player_cmd = (
        shutil.which("termux-media-player")
        or shutil.which("play")
        or shutil.which("mpv")
        or shutil.which("aplay")
        or shutil.which("paplay")
    )

    wav_bytes = generate_beep_wav()
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
            tmp_audio.write(wav_bytes)
            tmp_audio_path = tmp_audio.name

        if player_cmd:
            if "termux-media-player" in player_cmd:
                cmd = [player_cmd, "play", tmp_audio_path]
            else:
                cmd = [player_cmd, tmp_audio_path]
            subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            results["media"] = True

        time.sleep(0.3)
        try:
            os.remove(tmp_audio_path)
        except OSError:
            pass
    except Exception:
        pass

    # 3. Terminal acoustic bell (\a)
    try:
        for _ in range(5):
            sys.stdout.write("\a")
            sys.stdout.flush()
            time.sleep(0.1)
        results["bell"] = True
    except Exception:
        pass

    return results


# ----------------------------------------------------------------------------- Terminal UI Commands
def display_rules_and_prices(rules: list, prices: dict = None):
    """Prints current alert rules and live prices to the terminal."""
    print("\n" + "=" * 60)
    print("           REGOLE ALERT E ULTIMI PREZZI REGISTRATI")
    print("=" * 60)
    if not rules:
        print(" Nessuna regola configurata.")
        print("=" * 60 + "\n")
        return

    prices = prices or {}
    print(f"{'#':<3} {'Simbolo':<10} {'Condizione':<12} {'Target':<12} {'Prezzo Attuale':<15} {'Stato':<8}")
    print("-" * 60)
    for idx, r in enumerate(rules, start=1):
        sym = r["symbol"]
        cond_str = "Rialzo (>=)" if r["condition"] == ">=" else "Ribasso (<=)"
        target_str = f"{r['target']:.2f}"
        cur_p = prices.get(sym)
        p_str = f"{cur_p:.2f}" if cur_p is not None else "N/A"
        active_str = "ATTIVO" if r.get("active", True) else "DISATTIVATO"
        print(f"{idx:<3} {sym:<10} {cond_str:<12} {target_str:<12} {p_str:<15} {active_str:<8}")
    print("=" * 60 + "\n")


def prompt_add_rule(rules: list) -> list:
    """Prompt user to add a new alert rule."""
    print("\n--- AGGIUNGI NUOVA REGOLA ALERT ---")
    symbol = input("Inserisci Simbolo Ticker (es. AAPL, BTC-USD, MSTR): ").strip().upper()
    if not symbol:
        print("Simbolo non valido. Operazione annullata.")
        return rules

    print("Tipo di direzione da tracciare:")
    print("  1) Rialzo (Prezzo >= Target)")
    print("  2) Ribasso (Prezzo <= Target)")
    choice = input("Seleziona opzione (1 o 2): ").strip()
    if choice == "1":
        condition = ">="
    elif choice == "2":
        condition = "<="
    else:
        print("Opzione non valida. Operazione annullata.")
        return rules

    target_raw = input("Inserisci Prezzo Target: ").strip()
    try:
        target = float(target_raw)
        if target <= 0:
            raise ValueError
    except ValueError:
        print("Prezzo target non valido. Operazione annullata.")
        return rules

    new_rule = {"symbol": symbol, "condition": condition, "target": target, "active": True}
    rules.append(new_rule)
    save_rules(rules)
    print(f"Regola per {symbol} ({condition} {target:.2f}) aggiunta con successo!")
    return rules


def prompt_toggle_rule(rules: list) -> list:
    """Prompt user to toggle rule active state."""
    if not rules:
        print("\nNessuna regola da modificare.")
        return rules
    display_rules_and_prices(rules)
    idx_raw = input("Inserisci il numero della regola da attivare/disattivare: ").strip()
    try:
        idx = int(idx_raw) - 1
        if 0 <= idx < len(rules):
            rules[idx]["active"] = not rules[idx].get("active", True)
            save_rules(rules)
            status = "ATTIVATA" if rules[idx]["active"] else "DISATTIVATA"
            print(f"Regola per {rules[idx]['symbol']} impostata su: {status}")
        else:
            print("Numero regola non valido.")
    except ValueError:
        print("Input non valido.")
    return rules


def prompt_remove_rule(rules: list) -> list:
    """Prompt user to remove a rule."""
    if not rules:
        print("\nNessuna regola da rimuovere.")
        return rules
    display_rules_and_prices(rules)
    idx_raw = input("Inserisci il numero della regola da rimuovere: ").strip()
    try:
        idx = int(idx_raw) - 1
        if 0 <= idx < len(rules):
            removed = rules.pop(idx)
            save_rules(rules)
            print(f"Regola per {removed['symbol']} rimossa con successo!")
        else:
            print("Numero regola non valido.")
    except ValueError:
        print("Input non valido.")
    return rules


def start_live_monitoring(rules: list):
    """Runs continuous real-time market polling loop for active alert rules."""
    print("\n" + "=" * 60)
    print("      AVVIO MONITORAGGIO ALERT IN TEMPO REALE (TERMUX)")
    print("      Premere Ctrl+C per interrompere e tornare al menu")
    print("=" * 60 + "\n")

    fired = set()
    poll_count = 0

    try:
        while True:
            current_rules = load_rules()
            active_rules = [r for r in current_rules if r.get("active", True)]
            symbols = [r["symbol"] for r in active_rules]

            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            poll_count += 1
            print(f"[{now_str}] Check #{poll_count} - Download prezzi per {len(symbols)} simboli...")

            if symbols:
                prices = fetch_last_prices(symbols)
                for sym in symbols:
                    p = prices.get(sym)
                    p_str = f"{p:.2f}" if p is not None else "N/A"
                    print(f"  -> {sym}: {p_str}")

                events, fired = evaluate_rules(active_rules, prices, fired)
                for ev in events:
                    direction = "RIALZO" if ev["condition"] == ">=" else "RIBASSO"
                    msg = (
                        f"ALERT {ev['symbol']}! Prezzo {ev['price']:.2f} "
                        f"ha raggiunto il target a {direction} {ev['target']:.2f}"
                    )
                    print("\n" + "!" * 60)
                    print(f" *** TRIGGER SCATTATO: {msg} ***")
                    print("!" * 60 + "\n")
                    play_termux_sound(msg)
            else:
                print(" Nessuna regola attiva al momento.")

            print(f"Prossimo controllo tra {POLL_INTERVAL_SECONDS} secondi...\n")
            time.sleep(POLL_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\n\nMonitoraggio interrotto dall'utente. Ritorno al menu principale.\n")


def run_main_menu():
    """Main interactive terminal loop for Termux."""
    enforce_termux_environment()

    while True:
        rules = load_rules()
        print("\n" + "=" * 50)
        print("     TERMUX PRICE ALERT MONITOR (TERMINAL MODE)")
        print("=" * 50)
        print(" 1) Visualizza Regole e Prezzi")
        print(" 2) Aggiungi Nuova Regola Alert")
        print(" 3) Attiva / Disattiva Regola")
        print(" 4) Rimuovi Regola")
        print(" 5) Avvia Monitoraggio Continuo")
        print(" 6) Test Emissione Suono Termux")
        print(" 0) Esci")
        print("=" * 50)

        choice = input("Seleziona un'opzione [0-6]: ").strip()
        if choice == "1":
            active_symbols = [r["symbol"] for r in rules if r.get("active", True)]
            prices = fetch_last_prices(active_symbols) if active_symbols else {}
            display_rules_and_prices(rules, prices)
        elif choice == "2":
            prompt_add_rule(rules)
        elif choice == "3":
            prompt_toggle_rule(rules)
        elif choice == "4":
            prompt_remove_rule(rules)
        elif choice == "5":
            start_live_monitoring(rules)
        elif choice == "6":
            print("\nEsecuzione test suono Termux...")
            res = play_termux_sound("Test suono alert Termux in corso")
            print(f"Risultato test suono: {res}\n")
        elif choice == "0":
            print("\nChiusura Termux Price Alert Monitor. Arrivederci!")
            break
        else:
            print("\nOpzione non valida. Riprova.")


if __name__ == "__main__":
    run_main_menu()
