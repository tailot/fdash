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

This module provides a pure standard-library price alert monitoring system
designed to run exclusively inside Termux on Android without requiring heavy
third-party dependencies (e.g. pandas, numpy, yfinance). Notifications and audio
leverage sound playback (generated WAV tone, media players, terminal bells) and
Android system notifications (`termux-notification`), supporting background screen
lock execution via Termux WakeLock.
"""

import io
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

CONFIG_PATH = os.environ.get("TERMUX_ALERTS_FILE", "termux_alerts_config.json")
POLL_INTERVAL_SECONDS = int(os.environ.get("TERMUX_ALERT_POLL_SECONDS", "60"))
CONDITIONS = (">=", "<=")
ALERT_MODES = ("sound", "vibration", "both")
DEFAULT_ALERT_MODE = os.environ.get("TERMUX_ALERT_MODE", "both").lower()
if DEFAULT_ALERT_MODE not in ALERT_MODES:
    DEFAULT_ALERT_MODE = "both"


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
            "\n[ERROR] This script is designed to run EXCLUSIVELY inside the Termux environment in terminal mode.\n"
            "Execution terminated.\n\n"
        )
        sys.exit(1)


# ----------------------------------------------------------------------------- Termux WakeLock Handling
def acquire_wake_lock() -> bool:
    """Acquires a CPU Wake Lock in Termux so market polling and sound playback continue when screen is locked."""
    cmd = shutil.which("termux-wake-lock")
    if cmd:
        try:
            subprocess.run([cmd], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception:
            pass
    return False


def release_wake_lock() -> bool:
    """Releases the CPU Wake Lock in Termux."""
    cmd = shutil.which("termux-wake-unlock")
    if cmd:
        try:
            subprocess.run([cmd], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception:
            pass
    return False


# ----------------------------------------------------------------------------- Termux System Notification & Vibration
def vibrate_termux(duration_ms: int = 1000) -> bool:
    """Vibrates device using Termux API (`termux-vibrate`)."""
    cmd = shutil.which("termux-vibrate") or "termux-vibrate"
    vibrate_args = [cmd, "-d", str(duration_ms), "-f"]
    try:
        subprocess.run(vibrate_args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, Exception):
        return False


def send_termux_notification(title: str, content: str, vibrate_pattern: str = None) -> bool:
    """
    Sends an Android system notification via Termux API (`termux-notification`).
    Optionally accepts a vibration pattern (e.g. "1000,500,1000").
    """
    cmd = shutil.which("termux-notification") or "termux-notification"
    notif_args = [
        cmd,
        "--title", title,
        "--content", content,
        "--id", "123"
    ]
    if vibrate_pattern:
        notif_args.extend(["--vibrate", vibrate_pattern])
    try:
        subprocess.run(notif_args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, Exception):
        return False


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


def load_config(path: str = None) -> tuple:
    """Loads rules and alert_mode setting from JSON file. Returns (rules, alert_mode)."""
    path = path or CONFIG_PATH
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rules = normalize_rules(data.get("rules", []))
        mode = str(data.get("alert_mode", DEFAULT_ALERT_MODE)).lower()
        if mode not in ALERT_MODES:
            mode = DEFAULT_ALERT_MODE
        return rules, mode
    except (OSError, ValueError):
        return [], DEFAULT_ALERT_MODE


def save_config(rules: list, alert_mode: str = "both", path: str = None) -> bool:
    """Saves rules and alert_mode setting to JSON file atomically."""
    path = path or CONFIG_PATH
    clean = normalize_rules(rules)
    mode = str(alert_mode).lower() if str(alert_mode).lower() in ALERT_MODES else DEFAULT_ALERT_MODE
    try:
        folder = os.path.dirname(os.path.abspath(path))
        if folder:
            os.makedirs(folder, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=folder or ".", prefix=".tmp_rules_", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"alert_mode": mode, "rules": clean}, f, indent=2)
        os.replace(tmp, path)
        return True
    except OSError:
        return False


def load_rules(path: str = None) -> list:
    """Loads alert rules from JSON file."""
    rules, _ = load_config(path)
    return rules


def save_rules(rules: list, path: str = None) -> bool:
    """Saves alert rules to JSON file atomically."""
    _, mode = load_config(path)
    return save_config(rules, mode, path)


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


# ----------------------------------------------------------------------------- Standard Library Price Downloading
def fetch_single_price(symbol: str) -> tuple:
    """Fetches the latest price for a single ticker symbol using Yahoo Finance JSON chart API."""
    sym = symbol.strip().upper()
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval=1m&range=1d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            if response.status == 200:
                payload = json.loads(response.read().decode("utf-8"))
                result = payload.get("chart", {}).get("result", [])
                if result:
                    meta = result[0].get("meta", {})
                    if "regularMarketPrice" in meta and meta["regularMarketPrice"] is not None:
                        return sym, float(meta["regularMarketPrice"])
                    quotes = result[0].get("indicators", {}).get("quote", [{}])[0].get("close", [])
                    valid_closes = [c for c in quotes if c is not None]
                    if valid_closes:
                        return sym, float(valid_closes[-1])
    except Exception:
        pass
    return sym, None


def fetch_last_prices(symbols: list) -> dict:
    """Fetches the latest available price for a list of symbols in parallel using standard library ThreadPoolExecutor."""
    symbols = list(dict.fromkeys(s.strip().upper() for s in symbols if s and s.strip()))
    if not symbols:
        return {}
    prices = {}
    with ThreadPoolExecutor(max_workers=min(len(symbols), 8)) as executor:
        futures = [executor.submit(fetch_single_price, s) for s in symbols]
        for future in as_completed(futures):
            sym, price = future.result()
            if price is not None:
                prices[sym] = price
    return prices


# ----------------------------------------------------------------------------- Pure Standard Library Audio Generation
def generate_beep_wav(freq: float = 880.0, beep_s: float = 0.25, gap_s: float = 0.1,
                      repeats: int = 3, rate: int = 22050) -> bytes:
    """Generates a PCM WAV audio buffer using only standard library math, wave, and struct modules."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)

        beep_samples = int(rate * beep_s)
        gap_samples = int(rate * gap_s)
        frames = bytearray()

        for _ in range(repeats):
            for i in range(beep_samples):
                t = i / rate
                env = min(1.0, min(t, beep_s - t) * 80.0)
                val = int(0.6 * env * math.sin(2 * math.pi * freq * t) * 32767)
                val = max(-32768, min(32767, val))
                frames.extend(struct.pack("<h", val))
            frames.extend(struct.pack("<h", 0) * gap_samples)

        w.writeframes(frames)
    return buf.getvalue()


def play_termux_sound() -> dict:
    """
    Plays audio sound alerts utilizing Termux capabilities:
    1. termux-media-player or command line audio players with generated PCM WAV tone
    2. Terminal acoustic bell (\a) fallback
    """
    results = {"media": False, "bell": False}

    # 1. Termux Media Player / CLI Audio Player with generated WAV tone
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

    # 2. Terminal acoustic bell (\a)
    try:
        for _ in range(5):
            sys.stdout.write("\a")
            sys.stdout.flush()
            time.sleep(0.1)
        results["bell"] = True
    except Exception:
        pass

    return results


def trigger_alert_notification(mode: str, title: str, content: str) -> dict:
    """
    Triggers alert based on the configured mode:
    - mode='sound': plays sound + sends system notification
    - mode='vibration': vibrates device + sends system notification
    - mode='both': plays sound + vibrates device + sends system notification
    """
    results = {"sound": False, "vibration": False, "notification": False}
    mode = (mode or "both").lower()
    if mode not in ALERT_MODES:
        mode = "both"

    if mode in ("sound", "both"):
        sound_res = play_termux_sound()
        results["sound"] = any(sound_res.values())

    if mode in ("vibration", "both"):
        results["vibration"] = vibrate_termux(1000)

    vibrate_pattern = "1000,500,1000" if mode in ("vibration", "both") else None
    results["notification"] = send_termux_notification(title, content, vibrate_pattern=vibrate_pattern)
    return results


# ----------------------------------------------------------------------------- Terminal UI Commands
def display_rules_and_prices(rules: list, prices: dict = None):
    """Prints current alert rules and live prices to the terminal."""
    print("\n" + "=" * 60)
    print("           ALERT RULES & LATEST PRICES")
    print("=" * 60)
    if not rules:
        print(" No alert rules configured.")
        print("=" * 60 + "\n")
        return

    prices = prices or {}
    print(f"{'#':<3} {'Symbol':<10} {'Condition':<12} {'Target':<12} {'Current Price':<15} {'Status':<8}")
    print("-" * 60)
    for idx, r in enumerate(rules, start=1):
        sym = r["symbol"]
        cond_str = "Bullish (>=)" if r["condition"] == ">=" else "Bearish (<=)"
        target_str = f"{r['target']:.2f}"
        cur_p = prices.get(sym)
        p_str = f"{cur_p:.2f}" if cur_p is not None else "N/A"
        active_str = "ACTIVE" if r.get("active", True) else "DISABLED"
        print(f"{idx:<3} {sym:<10} {cond_str:<12} {target_str:<12} {p_str:<15} {active_str:<8}")
    print("=" * 60 + "\n")


def prompt_select_alert_mode(current_mode: str) -> str:
    """Prompt user to set or change alert mode (sound, vibration, or both)."""
    print("\n--- SELECT ALERT NOTIFICATION MODE ---")
    print(f" Current Mode: {current_mode.upper()}")
    print("  1) Sound Only (Suono)")
    print("  2) Vibration Only (Vibrazione)")
    print("  3) Both Sound & Vibration (Suono e Vibrazione)")
    try:
        choice = input("Select option [1-3] (Press Enter to keep current): ").strip()
        if choice == "1":
            return "sound"
        elif choice == "2":
            return "vibration"
        elif choice == "3":
            return "both"
        elif choice == "":
            return current_mode
        else:
            print("Invalid choice. Keeping current mode.")
            return current_mode
    except (KeyboardInterrupt, EOFError):
        print("\nOperation cancelled. Keeping current mode.")
        return current_mode


def prompt_add_rule(rules: list) -> list:
    """Prompt user to add a new alert rule."""
    print("\n--- ADD NEW ALERT RULE ---")
    try:
        symbol = input("Enter Ticker Symbol (e.g. AAPL, BTC-USD, MSTR): ").strip().upper()
        if not symbol:
            print("Invalid symbol. Operation cancelled.")
            return rules

        print("Direction to track:")
        print("  1) Bullish (Price >= Target)")
        print("  2) Bearish (Price <= Target)")
        choice = input("Select option (1 or 2): ").strip()
        if choice == "1":
            condition = ">="
        elif choice == "2":
            condition = "<="
        else:
            print("Invalid option. Operation cancelled.")
            return rules

        target_raw = input("Enter Target Price: ").strip()
        try:
            target = float(target_raw)
            if target <= 0:
                raise ValueError
        except ValueError:
            print("Invalid target price. Operation cancelled.")
            return rules
    except (KeyboardInterrupt, EOFError):
        print("\nOperation cancelled.")
        return rules

    new_rule = {"symbol": symbol, "condition": condition, "target": target, "active": True}
    rules.append(new_rule)
    save_rules(rules)
    print(f"Rule for {symbol} ({condition} {target:.2f}) added successfully!")
    return rules


def prompt_toggle_rule(rules: list) -> list:
    """Prompt user to toggle rule active state."""
    if not rules:
        print("\nNo rules to modify.")
        return rules
    display_rules_and_prices(rules)
    try:
        idx_raw = input("Enter the rule number to enable/disable: ").strip()
        idx = int(idx_raw) - 1
        if 0 <= idx < len(rules):
            rules[idx]["active"] = not rules[idx].get("active", True)
            save_rules(rules)
            status = "ACTIVE" if rules[idx]["active"] else "DISABLED"
            print(f"Rule for {rules[idx]['symbol']} set to: {status}")
        else:
            print("Invalid rule number.")
    except (KeyboardInterrupt, EOFError):
        print("\nOperation cancelled.")
    except ValueError:
        print("Invalid input.")
    return rules


def prompt_remove_rule(rules: list) -> list:
    """Prompt user to remove a rule."""
    if not rules:
        print("\nNo rules to remove.")
        return rules
    display_rules_and_prices(rules)
    try:
        idx_raw = input("Enter the rule number to remove: ").strip()
        idx = int(idx_raw) - 1
        if 0 <= idx < len(rules):
            removed = rules.pop(idx)
            save_rules(rules)
            print(f"Rule for {removed['symbol']} removed successfully!")
        else:
            print("Invalid rule number.")
    except (KeyboardInterrupt, EOFError):
        print("\nOperation cancelled.")
    except ValueError:
        print("Invalid input.")
    return rules


def start_live_monitoring(rules: list, alert_mode: str = "both"):
    """Runs continuous real-time market polling loop for active alert rules."""
    print("\n" + "=" * 60)
    print(f"      STARTING REAL-TIME ALERT MONITORING (TERMUX)")
    print(f"      Alert Mode: {alert_mode.upper()}")
    print("      Press Ctrl+C to stop and return to main menu")
    print("=" * 60 + "\n")

    wake_acquired = acquire_wake_lock()
    if wake_acquired:
        print("[+] Termux WakeLock acquired (CPU will remain active when screen is locked).")
    else:
        print("[!] Note: termux-wake-lock not available. For background monitoring with screen locked,")
        print("    ensure termux-wake-lock is accessible.")

    fired = set()
    poll_count = 0

    try:
        while True:
            current_rules = load_rules()
            active_rules = [r for r in current_rules if r.get("active", True)]
            symbols = [r["symbol"] for r in active_rules]

            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            poll_count += 1
            print(f"[{now_str}] Poll #{poll_count} - Fetching prices for {len(symbols)} symbols...")

            if symbols:
                prices = fetch_last_prices(symbols)
                for sym in symbols:
                    p = prices.get(sym)
                    p_str = f"{p:.2f}" if p is not None else "N/A"
                    print(f"  -> {sym}: {p_str}")

                events, fired = evaluate_rules(active_rules, prices, fired)
                for ev in events:
                    direction = "BULLISH" if ev["condition"] == ">=" else "BEARISH"
                    msg = (
                        f"ALERT {ev['symbol']}! Price {ev['price']:.2f} "
                        f"reached {direction} target {ev['condition']} {ev['target']:.2f}"
                    )
                    print("\n" + "!" * 60)
                    print(f" *** TRIGGER FIRED: {msg} ***")
                    print("!" * 60 + "\n")
                    trigger_alert_notification(alert_mode, f"Price Alert: {ev['symbol']}", msg)
            else:
                print(" No active rules configured.")

            print(f"Next check in {POLL_INTERVAL_SECONDS} seconds...\n")
            time.sleep(POLL_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user. Returning to main menu.\n")
    finally:
        if wake_acquired:
            release_wake_lock()
            print("[+] Termux WakeLock released.")


def run_main_menu():
    """Main interactive terminal loop for Termux."""
    enforce_termux_environment()

    rules, alert_mode = load_config()

    print("\n" + "=" * 50)
    print("     TERMUX PRICE ALERT MONITOR (TERMINAL MODE)")
    print("=" * 50)

    # Launch option prompt to select alert mode (sound / vibration / both)
    alert_mode = prompt_select_alert_mode(alert_mode)
    save_config(rules, alert_mode)

    while True:
        rules, alert_mode = load_config()
        print("\n" + "=" * 50)
        print(f"     TERMUX PRICE ALERT MONITOR - MODE: {alert_mode.upper()}")
        print("=" * 50)
        print(" 1) View Rules & Prices")
        print(" 2) Add New Alert Rule")
        print(" 3) Enable / Disable Rule")
        print(" 4) Remove Rule")
        print(" 5) Start Continuous Monitoring")
        print(" 6) Test Alert Notification (Sound / Vibration)")
        print(" 7) Change Alert Mode (Sound / Vibration / Both)")
        print(" 0) Exit")
        print("=" * 50)

        try:
            choice = input("Select an option [0-7]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting Termux Price Alert Monitor. Goodbye!")
            break

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
            start_live_monitoring(rules, alert_mode)
        elif choice == "6":
            print(f"\nExecuting alert test for mode [{alert_mode.upper()}]...")
            res = trigger_alert_notification(alert_mode, "Termux Alert Test", f"Alert test in progress (Mode: {alert_mode.upper()})")
            print(f"Test results: {res}\n")
        elif choice == "7":
            alert_mode = prompt_select_alert_mode(alert_mode)
            save_config(rules, alert_mode)
            print(f"Alert mode updated to: {alert_mode.upper()}")
        elif choice == "0":
            print("\nExiting Termux Price Alert Monitor. Goodbye!")
            break
        else:
            print("\nInvalid option. Please try again.")


if __name__ == "__main__":
    try:
        run_main_menu()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting Termux Price Alert Monitor. Goodbye!")
