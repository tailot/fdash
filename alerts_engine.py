"""
Price Alerts Engine.

Pure logic (no Streamlit): rule normalization, price download, rule evaluation,
persistence and alarm-sound generation. The UI lives in `alerts_ui.py`.

Configuration (via code or environment variables):
  POLL_INTERVAL_SECONDS -> seconds between two calls to the finance service (default 60)
  CONFIG_PATH           -> JSON file where the alert rules are persisted

Background notifications (Web Push, see `alerts_worker.py` and `static/sw.js`):
  FDASH_DATA_DIR        -> folder shared by the dashboard and the worker (default ".")
  FDASH_ALERTS_FILE     -> alert rules             (default <data dir>/alerts_config.json)
  FDASH_PUSH_SUBS_FILE  -> browser subscriptions   (default <data dir>/alerts_push_subs.json)
  FDASH_VAPID_FILE      -> VAPID key pair          (default <data dir>/alerts_vapid.json)
  FDASH_ALERTS_STATE_FILE -> worker state/heartbeat(default <data dir>/alerts_state.json)
  FDASH_VAPID_CONTACT   -> "mailto:..." contact sent to the push services
"""

import base64
import io
import json
import logging
import os
import tempfile
import time
import wave
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

# ----------------------------------------------------------------------------- configuration
POLL_INTERVAL_SECONDS = int(os.environ.get("FDASH_ALERT_POLL_SECONDS", "60"))
DATA_DIR = os.environ.get("FDASH_DATA_DIR", ".")
CONFIG_PATH = os.environ.get("FDASH_ALERTS_FILE", os.path.join(DATA_DIR, "alerts_config.json"))
PUSH_SUBS_PATH = os.environ.get("FDASH_PUSH_SUBS_FILE", os.path.join(DATA_DIR, "alerts_push_subs.json"))
VAPID_PATH = os.environ.get("FDASH_VAPID_FILE", os.path.join(DATA_DIR, "alerts_vapid.json"))
STATE_PATH = os.environ.get("FDASH_ALERTS_STATE_FILE", os.path.join(DATA_DIR, "alerts_state.json"))
VAPID_CONTACT = os.environ.get("FDASH_VAPID_CONTACT", "mailto:admin@example.com")
log = logging.getLogger("fdash.alerts")
MAX_LOG_ENTRIES = 200
PUSH_TTL_SECONDS = 3600  # the push service keeps an undelivered alert for 1 hour (device offline)

CONDITIONS = (">=", "<=")  # price rises to/above target | falls to/below target
RULE_COLUMNS = ["symbol", "condition", "target", "active"]


# ----------------------------------------------------------------------------- rules
def empty_rules() -> pd.DataFrame:
    return pd.DataFrame({
        "symbol": pd.Series(dtype="object"),
        "condition": pd.Series(dtype="object"),
        "target": pd.Series(dtype="float64"),
        "active": pd.Series(dtype="bool"),
    })


def normalize_rules(df: pd.DataFrame) -> list:
    """Cleans the table coming from the editor: drops incomplete/invalid rows."""
    rules = []
    if df is None or len(df) == 0:
        return rules
    for _, row in df.iterrows():
        symbol = str(row.get("symbol") or "").strip().upper()
        condition = row.get("condition")
        target = pd.to_numeric(row.get("target"), errors="coerce")
        if not symbol or condition not in CONDITIONS or pd.isna(target) or target <= 0:
            continue
        active = row.get("active")
        active = True if pd.isna(active) else bool(active)
        rules.append({"symbol": symbol, "condition": condition, "target": float(target), "active": active})
    return rules


def rule_key(rule: dict) -> str:
    return f"{rule['symbol']}|{rule['condition']}|{rule['target']:.6f}"


def is_triggered(rule: dict, price: float) -> bool:
    if rule["condition"] == ">=":
        return price >= rule["target"]
    return price <= rule["target"]


def evaluate_rules(rules: list, prices: dict, fired: set):
    """
    Returns (events, new_fired).
    A rule fires once when its condition becomes true; it re-arms as soon as the
    condition becomes false again (so the alarm is not repeated every minute).
    """
    events, new_fired = [], set()
    now = datetime.now().strftime("%H:%M:%S")
    for rule in rules:
        if not rule["active"]:
            continue
        price = prices.get(rule["symbol"])
        if price is None:
            if rule_key(rule) in fired:  # no data: keep current state
                new_fired.add(rule_key(rule))
            continue
        if is_triggered(rule, price):
            new_fired.add(rule_key(rule))
            if rule_key(rule) not in fired:
                events.append({"time": now, "symbol": rule["symbol"], "condition": rule["condition"],
                               "target": rule["target"], "price": float(price)})
    return events, new_fired


# ----------------------------------------------------------------------------- finance service
def fetch_last_prices(symbols) -> dict:
    """ONE batched call to the finance service for all the symbols -> {symbol: last 1-minute close}."""
    symbols = list(dict.fromkeys(s.strip().upper() for s in symbols if s and s.strip()))
    if not symbols:
        return {}
    raw = yf.download(symbols, period="1d", interval="1m", prepost=True, auto_adjust=True,
                      progress=False, threads=False)
    if raw is None or raw.empty:
        return {}
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(symbols[0])
    prices = {}
    for s in symbols:
        if s in close.columns:
            serie = close[s].dropna()
            if not serie.empty:
                prices[s] = float(serie.iloc[-1])
    return prices


# ----------------------------------------------------------------------------- persistence
def load_rules(path: str = None) -> pd.DataFrame:
    path = path or CONFIG_PATH
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data.get("rules", []), columns=RULE_COLUMNS)
        df["target"] = pd.to_numeric(df["target"], errors="coerce")
        df["active"] = df["active"].fillna(True).astype(bool)
        return df
    except (OSError, ValueError):
        return empty_rules()


def save_rules(df: pd.DataFrame, path: str = None) -> None:
    path = path or CONFIG_PATH
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"rules": normalize_rules(df)}, f, indent=2)
    except OSError:
        pass  # read-only filesystem: rules simply stay in the session


# ----------------------------------------------------------------------------- alarm sound
def make_beep_wav(freq: float = 880.0, beep_s: float = 0.22, gap_s: float = 0.12,
                  repeats: int = 3, rate: int = 22050) -> bytes:
    """Short 'beep-beep-beep' as WAV bytes (no audio asset needed)."""
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


# ----------------------------------------------------------------------------- shared files (dashboard <-> worker)
def _read_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _write_json_atomic(path: str, data) -> bool:
    """Write-then-rename, so the other process never reads a half-written file."""
    try:
        folder = os.path.dirname(os.path.abspath(path))
        os.makedirs(folder, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=folder, prefix=".tmp_", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
        return True
    except OSError:
        return False


# ----------------------------------------------------------------------------- VAPID keys (Web Push identity)
def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def ensure_vapid_keys(path: str = None) -> dict:
    """
    Returns {"private_pem": str, "public_key": str}; creates the key pair the first time.
    `public_key` is the base64url "applicationServerKey" the browser needs to subscribe.
    Dashboard and worker may both call this: creation is exclusive (os.link fails if the file exists).
    """
    path = path or VAPID_PATH
    data = _read_json(path, {})
    if data.get("private_pem") and data.get("public_key"):
        return data

    from cryptography.hazmat.primitives import serialization
    from py_vapid import Vapid

    vapid = Vapid()
    vapid.generate_keys()
    public_raw = vapid.public_key.public_bytes(serialization.Encoding.X962,
                                               serialization.PublicFormat.UncompressedPoint)
    data = {"private_pem": vapid.private_pem().decode("ascii"), "public_key": _b64url(public_raw)}

    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=folder, prefix=".tmp_vapid_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f)
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    try:
        os.link(tmp, path)  # atomic + exclusive: the first process wins
    except FileExistsError:
        data = _read_json(path, data)
    except OSError:  # filesystem without hard links
        os.replace(tmp, path)
        return data
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return data


# ----------------------------------------------------------------------------- push subscriptions
def valid_subscription(sub) -> bool:
    if not isinstance(sub, dict):
        return False
    keys = sub.get("keys") or {}
    endpoint = str(sub.get("endpoint") or "")
    return endpoint.startswith("https://") and bool(keys.get("p256dh")) and bool(keys.get("auth"))


def load_subscriptions(path: str = None) -> list:
    data = _read_json(path or PUSH_SUBS_PATH, {})
    return [s for s in data.get("subscriptions", []) if valid_subscription(s)]


def add_subscription(sub: dict, lang: str = "en", path: str = None) -> bool:
    """Adds (or refreshes) a browser subscription. Returns True when something changed."""
    if not valid_subscription(sub):
        return False
    path = path or PUSH_SUBS_PATH
    subs = load_subscriptions(path)
    record = {"endpoint": sub["endpoint"], "keys": {"p256dh": sub["keys"]["p256dh"], "auth": sub["keys"]["auth"]},
              "lang": lang or "en"}
    for i, existing in enumerate(subs):
        if existing["endpoint"] == record["endpoint"]:
            if existing == record:
                return False
            subs[i] = record
            break
    else:
        subs.append(record)
    return _write_json_atomic(path, {"subscriptions": subs})


def remove_subscription(endpoint: str, path: str = None) -> bool:
    path = path or PUSH_SUBS_PATH
    subs = load_subscriptions(path)
    kept = [s for s in subs if s["endpoint"] != endpoint]
    if len(kept) == len(subs):
        return False
    return _write_json_atomic(path, {"subscriptions": kept})


# ----------------------------------------------------------------------------- notification payload
def build_push_payload(event: dict, lang: str = "en", url: str = "/") -> dict:
    """
    The "pretty" notification: direction emoji, localized title/body, distance from the target,
    icon + badge, per-rule tag (a new alert for the same rule replaces the old one), action buttons.
    The same payload is used by the worker (Web Push) and by the in-page fallback.
    """
    from i18n import t  # local import: keeps this module importable without the UI layer

    up = event["condition"] == ">="
    delta = (event["price"] / event["target"] - 1) * 100 if event["target"] else 0.0
    return {
        "title": t("push_title_up" if up else "push_title_down", lang).format(event["symbol"]),
        "body": t("push_body", lang).format(event["price"], event["condition"], event["target"], delta),
        "icon": "icon-192.png",
        "badge": "badge-72.png",
        "tag": f"fdash-{event['symbol']}-{event['condition']}-{event['target']:.6f}",
        "renotify": True,
        "timestamp": int(event.get("ts") or time.time()) * 1000,
        "vibrate": [120, 60, 120],
        "url": url,
        "actions": [{"action": "open", "title": t("push_action_open", lang)},
                    {"action": "dismiss", "title": t("push_action_dismiss", lang)}],
    }


def send_push(sub: dict, payload: dict, vapid: dict = None, contact: str = None) -> str:
    """Sends one Web Push. Returns "ok" | "gone" (subscription expired: delete it) | "error"."""
    from py_vapid import Vapid
    from pywebpush import WebPushException, webpush

    vapid = vapid or ensure_vapid_keys()
    try:
        webpush(subscription_info={"endpoint": sub["endpoint"], "keys": sub["keys"]},
                data=json.dumps(payload),
                vapid_private_key=Vapid.from_pem(vapid["private_pem"].encode("ascii")),
                vapid_claims={"sub": contact or VAPID_CONTACT},
                ttl=PUSH_TTL_SECONDS,
                headers={"Urgency": "high"},  # deliver right away even if the phone is dozing
                timeout=15)
        return "ok"
    except WebPushException as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status in (404, 410):
            log.info("push subscription expired (HTTP %s): removing it", status)
            return "gone"
        log.warning("push failed (HTTP %s): %s", status, str(e)[:200])
        return "error"
    except Exception as e:
        log.warning("push failed: %s", e)
        return "error"


def notify_subscribers(event: dict, subs_path: str = None, sender=None, vapid: dict = None) -> dict:
    """Sends the event to every subscribed browser (each in its own language); prunes dead subscriptions."""
    subs_path = subs_path or PUSH_SUBS_PATH
    subs = load_subscriptions(subs_path)
    result = {"ok": 0, "gone": 0, "error": 0}
    if not subs:
        return result
    if sender is None:
        vapid = vapid or ensure_vapid_keys()
        sender = lambda sub, payload: send_push(sub, payload, vapid)  # noqa: E731
    for sub in subs:
        status = sender(sub, build_push_payload(event, sub.get("lang", "en")))
        result[status] = result.get(status, 0) + 1
        if status == "gone":
            remove_subscription(sub["endpoint"], subs_path)
    return result


# ----------------------------------------------------------------------------- worker cycle + shared state
def load_state(path: str = None) -> dict:
    state = _read_json(path or STATE_PATH, {})
    state.setdefault("heartbeat", 0.0)
    state.setdefault("last_poll", 0.0)
    state.setdefault("interval", POLL_INTERVAL_SECONDS)
    state.setdefault("prices", {})
    state.setdefault("fired", [])
    state.setdefault("events", [])
    state.setdefault("seq", 0)
    state.setdefault("last_error", None)
    return state


def worker_alive(state: dict, now: float = None) -> bool:
    """True when the worker service wrote its heartbeat recently (so the dashboard must not poll too)."""
    now = now or time.time()
    interval = float(state.get("interval") or POLL_INTERVAL_SECONDS)
    return now - float(state.get("heartbeat") or 0) < max(2.5 * interval, 30.0)


def worker_cycle(state: dict, rules_path: str = None, fetch=None, notify=None, now: float = None) -> dict:
    """
    ONE iteration of the worker: read the rules, 1 batched price call, evaluate, push the new alerts.
    Returns the updated state (the caller persists it).
    """
    fetch = fetch or fetch_last_prices
    notify = notify or notify_subscribers
    now = now or time.time()

    rules = [r for r in normalize_rules(load_rules(rules_path)) if r["active"]]
    prices = {}
    state["last_error"] = None
    if rules:
        try:
            prices = fetch([r["symbol"] for r in rules])
            if not prices:
                state["last_error"] = "no_data"
        except Exception as e:  # network / service errors must not kill the worker
            state["last_error"] = str(e)[:200]
        state["last_poll"] = now
    state["prices"].update(prices)

    events, fired = evaluate_rules(rules, prices, set(state["fired"]))
    state["fired"] = sorted(fired)
    for e in events:
        state["seq"] += 1
        e["id"], e["ts"] = state["seq"], int(now)
        e["push"] = notify(e)
    state["events"] = (events[::-1] + state["events"])[:MAX_LOG_ENTRIES]
    state["heartbeat"] = now
    state["interval"] = POLL_INTERVAL_SECONDS
    return state


def save_state(state: dict, path: str = None) -> bool:
    return _write_json_atomic(path or STATE_PATH, state)
