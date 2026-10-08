"""
Streamlit UI for the "Alerts" section.

Two parts:
  * `render_alert_monitor`  -> small fragment in the sidebar, rendered in EVERY section.
                               - Worker service running (`alerts_worker.py`): it only mirrors the worker's
                                 state (prices, log) and plays the in-app sound/toast; the system
                                 notification is sent by the worker through Web Push and the browser's
                                 service worker (`static/sw.js`), so it arrives even with the tab closed.
                               - No worker: fallback to the old behaviour, it polls the finance service
                                 itself (1 call every POLL_INTERVAL_SECONDS) while the tab is open.
  * `render_alerts_section` -> the section itself: rule editor, controls, push subscription panel,
                               live prices, log.
"""

import json
import time

import pandas as pd
import streamlit as st

from alerts_engine import (CONDITIONS, MAX_LOG_ENTRIES, POLL_INTERVAL_SECONDS, add_subscription,
                           build_push_payload, clear_state_events, clear_state_prices, empty_rules, ensure_vapid_keys,
                           evaluate_rules, fetch_last_prices, is_triggered, load_rules, load_state, load_subscriptions,
                           make_beep_wav, normalize_rules, notify_subscribers, remove_subscription, save_rules, worker_alive)
from i18n import t

UI_REFRESH_SECONDS = 5   # redraw of the live table only (no network calls)
WORKER_SYNC_SECONDS = 10  # how often the sidebar monitor reads the worker's state file (no network calls)


def _js_str(value: str) -> str:
    """JSON string literal that is safe to embed inside an inline <script>."""
    return json.dumps(value).replace("</", "<\\/")


def _html(html: str, height: int):
    """Renders a small HTML/JS snippet (st.iframe on recent Streamlit, components.html on older ones)."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        import streamlit.components.v1 as components
        components.html(html, height=height)


@st.cache_data(show_spinner=False)
def _beep() -> bytes:
    return make_beep_wav()


def init_alerts_state():
    ss = st.session_state
    if "alerts_rules" not in ss:
        ss["alerts_rules"] = load_rules()
    ss.setdefault("alerts_editor_base", None)
    ss.setdefault("alerts_on", False)
    ss.setdefault("alerts_prices", {})
    ss.setdefault("alerts_fired", set())
    ss.setdefault("alerts_log", [])
    ss.setdefault("alerts_last_poll", 0.0)
    ss.setdefault("alerts_last_error", None)
    ss.setdefault("alerts_sound", True)
    ss.setdefault("alerts_notify", True)
    ss.setdefault("alerts_seen_seq", None)  # last worker event already signalled in this session


# ----------------------------------------------------------------------------- polling + signalling
def _browser_notify(event: dict, lang: str):
    """
    In-page fallback (used only when the worker service is NOT running): shows the same "pretty"
    notification through the service worker registration (the only way that works on Android too);
    falls back to a plain Notification if the service worker is not registered yet.
    """
    payload = _js_str(json.dumps(build_push_payload(event, lang)))
    _html(f"""<script>
    (async function() {{
      try {{
        var P = window.parent || window;
        if (!P.Notification || P.Notification.permission !== "granted") return;
        var d = JSON.parse({payload});
        var scope = P.location.origin + P.location.pathname.replace(/\\/+$/, "") + "/app/static/";
        var reg = P.navigator.serviceWorker ? await P.navigator.serviceWorker.getRegistration(scope) : null;
        if (reg && reg.active) {{
          var title = d.title; delete d.title;
          d.icon = scope + d.icon; d.badge = scope + d.badge; d.data = {{ url: d.url }};
          await reg.showNotification(title, d);
        }} else {{
          new P.Notification(d.title, {{ body: d.body }});
        }}
      }} catch (e) {{}}
    }})();
    </script>""", height=1)


def _signal(events: list, lang: str, browser: bool = True):
    """In-app signalling: toast + sound (+ system notification only in the no-worker fallback)."""
    ss = st.session_state
    for e in events:
        title = t("alert_fired_title", lang).format(e["symbol"])
        body = t("alert_fired_body", lang).format(e["symbol"], e["price"], e["condition"], e["target"])
        if ss["alerts_notify"]:
            st.toast(f"**{title}**  \n{body}", icon="🔔")
            if browser:
                _browser_notify(e, lang)
    if ss["alerts_sound"] and events:
        st.audio(_beep(), format="audio/wav", autoplay=True)


def _poll_if_due(lang: str):
    """Calls the finance service at most once every POLL_INTERVAL_SECONDS (even if the app reruns)."""
    ss = st.session_state
    rules = [r for r in normalize_rules(ss["alerts_rules"]) if r["active"]]
    if not rules:
        return
    if time.time() - ss["alerts_last_poll"] < POLL_INTERVAL_SECONDS * 0.9:
        return
    ss["alerts_last_poll"] = time.time()
    try:
        prices = fetch_last_prices([r["symbol"] for r in rules])
        ss["alerts_last_error"] = None if prices else t("alerts_no_data", lang)
    except Exception as e:  # network / service errors must not kill the monitor
        prices = {}
        ss["alerts_last_error"] = str(e)
    ss["alerts_prices"].update(prices)
    events, ss["alerts_fired"] = evaluate_rules(rules, prices, ss["alerts_fired"])
    if events:
        ss["alerts_log"] = (events[::-1] + ss["alerts_log"])[:MAX_LOG_ENTRIES]
        _signal(events, lang)


def _adopt_worker_state(state: dict, lang: str):
    """Mirrors the worker's state (prices, log, last error) into the session. No signalling here."""
    ss = st.session_state
    ss["alerts_prices"].update(state["prices"])
    ss["alerts_last_poll"] = state["last_poll"]
    ss["alerts_fired"] = set(state["fired"])  # smooth hand-over if the worker stops and the fallback resumes
    ss["alerts_last_error"] = t("alerts_no_data", lang) if state["last_error"] == "no_data" else state["last_error"]
    ss["alerts_log"] = [{k: e[k] for k in ("time", "symbol", "condition", "target", "price")}
                        for e in state["events"]][:MAX_LOG_ENTRIES]


def _sync_from_worker(state: dict, lang: str):
    """The worker already fired the alerts (and sent the Web Push): here only toast + sound for open tabs."""
    ss = st.session_state
    seen = ss["alerts_seen_seq"]
    new = [] if seen is None else [e for e in reversed(state["events"]) if e["id"] > seen]
    ss["alerts_seen_seq"] = state["seq"]  # first sync: remember the history, don't replay old alerts
    _adopt_worker_state(state, lang)
    if new:
        _signal(new, lang, browser=False)


def _monitor_body(lang: str):
    ss = st.session_state
    state = load_state()
    if worker_alive(state):
        _sync_from_worker(state, lang)
        last = time.strftime("%H:%M:%S", time.localtime(state["last_poll"])) if state["last_poll"] else "—"
        st.success(t("push_worker_on", lang).format(last, int(state["interval"])))
        if ss["alerts_last_error"]:
            st.warning(ss["alerts_last_error"], icon="⚠️")
        return
    if ss["alerts_on"]:
        _poll_if_due(lang)
        last = time.strftime("%H:%M:%S", time.localtime(ss["alerts_last_poll"])) if ss["alerts_last_poll"] else "—"
        st.success(t("alerts_monitor_on", lang).format(last, POLL_INTERVAL_SECONDS), icon="🟢")
        if ss["alerts_last_error"]:
            st.warning(ss["alerts_last_error"], icon="⚠️")
    else:
        st.caption(t("alerts_monitor_off", lang))


def render_alert_monitor(lang: str):
    """Always rendered (sidebar). Auto-reruns every POLL_INTERVAL_SECONDS only while monitoring is ON."""
    if worker_alive(load_state()):
        every = WORKER_SYNC_SECONDS
    else:
        every = POLL_INTERVAL_SECONDS if st.session_state.get("alerts_on") else None
    st.fragment(_monitor_body, run_every=every)(lang)



# ----------------------------------------------------------------------------- push subscription panel
# Runs in the app page (components v2, no iframe), so it can reach navigator.serviceWorker / PushManager.
_PUSH_HTML = """<div class="row">
  <button id="enable" hidden></button>
  <button id="disable" hidden></button>
  <button id="test" hidden></button>
</div>"""

_PUSH_CSS = """
.row { display: flex; gap: .5rem; flex-wrap: wrap; margin: .25rem 0 .5rem; }
button { font: inherit; padding: .4rem .8rem; border-radius: .5rem; cursor: pointer;
         border: 1px solid rgba(128,128,128,.45); background: transparent; color: inherit; }
button:hover { border-color: #ff4b4b; color: #ff4b4b; }
button[hidden] { display: none; }
"""

_PUSH_JS = """
export default function (component) {
  const { data, parentElement, setStateValue, setTriggerValue } = component;
  const $ = (id) => parentElement.querySelector("#" + id);
  const enableBtn = $("enable"), disableBtn = $("disable"), testBtn = $("test");
  enableBtn.textContent = data.labels.enable;
  disableBtn.textContent = data.labels.disable;
  testBtn.textContent = data.labels.test;

  const base = window.location.pathname.replace(/\\/+$/, "");
  const scope = window.location.origin + base + "/app/static/";
  const swUrl = scope + "sw.js";
  const supported = "serviceWorker" in navigator && "PushManager" in window && "Notification" in window;

  const last = {};  // send a value to Python only when it changed (avoids rerun loops)
  const send = (key, value) => {
    const s = JSON.stringify(value);
    if (last[key] === s) return;
    last[key] = s;
    setStateValue(key, value);
  };
  const setStatus = (state, message) => {
    send("status", { state, message: message || "" });
    enableBtn.hidden = !(state === "off" || state === "error");
    disableBtn.hidden = state !== "on";
    testBtn.hidden = state !== "on";
  };
  const toBytes = (b64) => {
    const pad = "=".repeat((4 - (b64.length % 4)) % 4);
    const raw = atob((b64 + pad).replace(/-/g, "+").replace(/_/g, "/"));
    return Uint8Array.from(raw, (c) => c.charCodeAt(0));
  };
  const sameKey = (sub, bytes) => {
    const cur = sub.options && sub.options.applicationServerKey;
    if (!cur) return true;
    const a = new Uint8Array(cur);
    return a.length === bytes.length && a.every((v, i) => v === bytes[i]);
  };
  // The service worker's scope (/app/static/) does not cover the page, so navigator.serviceWorker.ready
  // would never resolve: wait on the registration returned by register() instead.
  const activeRegistration = async () => {
    const reg = await navigator.serviceWorker.register(swUrl, { scope, updateViaCache: "none" });
    if (reg.active) return reg;
    const sw = reg.installing || reg.waiting;
    await new Promise((resolve) => {
      if (!sw || sw.state === "activated") return resolve();
      sw.addEventListener("statechange", () => sw.state === "activated" && resolve());
    });
    return reg;
  };

  async function refresh() {
    if (!supported) return setStatus("unsupported");
    if (!window.isSecureContext) return setStatus("insecure");
    if (Notification.permission === "denied") return setStatus("denied");
    try {
      const reg = await activeRegistration();
      const sub = await reg.pushManager.getSubscription();
      if (sub && Notification.permission === "granted") {
        send("subscription", sub.toJSON());  // re-register it server side (idempotent)
        setStatus("on");
      } else {
        setStatus("off");
      }
    } catch (e) {
      setStatus("error", String((e && e.message) || e));
    }
  }

  enableBtn.onclick = async () => {
    try {
      const perm = await Notification.requestPermission();
      if (perm !== "granted") return setStatus(perm === "denied" ? "denied" : "off");
      const key = toBytes(data.publicKey);
      const reg = await activeRegistration();
      let sub = await reg.pushManager.getSubscription();
      if (sub && !sameKey(sub, key)) { await sub.unsubscribe(); sub = null; }  // server keys were regenerated
      if (!sub) sub = await reg.pushManager.subscribe({ userVisibleOnly: true, applicationServerKey: key });
      send("subscription", sub.toJSON());
      setStatus("on");
    } catch (e) {
      setStatus("error", String((e && e.message) || e));
    }
  };

  disableBtn.onclick = async () => {
    try {
      const reg = await activeRegistration();
      const sub = await reg.pushManager.getSubscription();
      if (sub) {
        const endpoint = sub.endpoint;
        await sub.unsubscribe();
        send("subscription", null);
        setTriggerValue("unsubscribed", endpoint);
      }
      setStatus("off");
    } catch (e) {
      setStatus("error", String((e && e.message) || e));
    }
  };

  testBtn.onclick = () => setTriggerValue("test", Date.now());

  refresh();
}
"""

_v2 = getattr(getattr(st, "components", None), "v2", None)
_push_component = _v2.component("fdash_push", html=_PUSH_HTML, css=_PUSH_CSS, js=_PUSH_JS) if _v2 else None


def _noop():
    pass


def _render_push_panel(lang: str):
    if not _push_component:
        st.caption(t("push_status_unsupported", lang))
        return
    ss = st.session_state
    st.subheader(t("push_header", lang))
    state = load_state()
    if worker_alive(state):
        st.success(t("push_worker_on", lang).format(
            time.strftime("%H:%M:%S", time.localtime(state["last_poll"])) if state["last_poll"] else "—",
            int(state["interval"])))
    else:
        st.warning(t("push_worker_off", lang))

    try:
        vapid = ensure_vapid_keys()
    except Exception as e:  # pywebpush not installed / data folder not writable
        st.error(t("push_status_error", lang).format(e))
        return

    res = _push_component(
        key="alerts_push",
        data={"publicKey": vapid["public_key"],
              "labels": {"enable": t("push_enable_button", lang), "disable": t("push_disable_button", lang),
                         "test": t("push_test_button", lang)}},
        on_subscription_change=_noop, on_status_change=_noop,
        on_unsubscribed_change=_noop, on_test_change=_noop)

    sub = getattr(res, "subscription", None)
    if sub:
        add_subscription(dict(sub), lang)
    gone = getattr(res, "unsubscribed", None)
    if gone:
        remove_subscription(str(gone))

    if getattr(res, "test", None):
        if load_subscriptions():
            demo = {"symbol": "TEST", "condition": ">=", "target": 100.0, "price": 101.25, "ts": int(time.time())}
            sent = notify_subscribers(demo)
            st.info(t("push_test_sent", lang).format(sent.get("ok", 0)))
        else:
            st.warning(t("push_test_none", lang))

    status = getattr(res, "status", None) or {}
    kind = status.get("state") if hasattr(status, "get") else None
    if kind == "on":
        st.caption(t("push_status_on", lang))
    elif kind == "error":
        st.caption(t("push_status_error", lang).format(status.get("message", "")))
    elif kind in ("denied", "unsupported", "insecure"):
        st.caption(t(f"push_status_{kind}", lang))
    else:
        st.caption(t("push_status_off", lang))


# ----------------------------------------------------------------------------- section
def _set_monitoring(value: bool):
    st.session_state["alerts_on"] = value
    if value:
        st.session_state["alerts_last_poll"] = 0.0  # first check immediately
        st.session_state["alerts_fired"] = set()


def _clear_prices():
    st.session_state["alerts_prices"] = {}
    clear_state_prices()


def _clear_log():
    st.session_state["alerts_log"] = []
    st.session_state["alerts_fired"] = set()
    clear_state_events()


def _live_body(lang: str):
    ss = st.session_state
    state = load_state()
    if worker_alive(state):
        _adopt_worker_state(state, lang)
    rules = normalize_rules(ss["alerts_rules"])

    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(t("alerts_live_header", lang))
    with col2:
        if ss["alerts_prices"]:
            st.button(t("alerts_clear_prices", lang), key="btn_clear_prices", on_click=_clear_prices)

    if not rules:
        st.info(t("alerts_no_rules", lang))
    else:
        rows = []
        for r in rules:
            price = ss["alerts_prices"].get(r["symbol"])
            if not r["active"]:
                status = t("alerts_status_paused", lang)
            elif price is None:
                status = t("alerts_status_waiting", lang)
            elif is_triggered(r, price):
                status = t("alerts_status_triggered", lang)
            else:
                status = t("alerts_status_ok", lang)
            rows.append({
                t("alerts_col_symbol", lang): r["symbol"],
                t("alerts_col_condition", lang): r["condition"],
                t("alerts_col_target", lang): f"{r['target']:.2f}",
                t("alerts_col_last_price", lang): f"{price:.2f}" if price is not None else "—",
                t("alerts_col_distance", lang): f"{(price / r['target'] - 1) * 100:+.2f}%" if price is not None else "—",
                t("alerts_col_status", lang): status,
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    col3, col4 = st.columns([3, 1])
    with col3:
        st.subheader(t("alerts_log_header", lang))
    with col4:
        if ss["alerts_log"]:
            st.button(t("alerts_clear_log", lang), key="btn_clear_log", on_click=_clear_log)

    if ss["alerts_log"]:
        log = pd.DataFrame(ss["alerts_log"]).rename(columns={
            "time": t("alerts_col_time", lang), "symbol": t("alerts_col_symbol", lang),
            "condition": t("alerts_col_condition", lang), "target": t("alerts_col_target", lang),
            "price": t("alerts_col_last_price", lang)})
        st.dataframe(log, width="stretch", hide_index=True)
    else:
        st.caption(t("alerts_log_empty", lang))


def render_alerts_section(lang: str):
    ss = st.session_state
    st.header(t("alerts_title", lang))
    st.markdown(t("alerts_subtitle", lang).format(POLL_INTERVAL_SECONDS))

    # The editor widget loses its state when the user is in another section: rebuild it from the saved rules.
    if ss["alerts_editor_base"] is None:
        ss["alerts_editor_base"] = ss["alerts_rules"].copy()

    edited = st.data_editor(
        ss["alerts_editor_base"], key="alerts_editor", num_rows="dynamic", width="stretch", hide_index=True,
        column_config={
            "symbol": st.column_config.TextColumn(t("alerts_col_symbol", lang), help=t("alerts_symbol_help", lang)),
            "condition": st.column_config.SelectboxColumn(t("alerts_col_condition", lang), options=list(CONDITIONS),
                                                          default=CONDITIONS[0], help=t("alerts_condition_help", lang)),
            "target": st.column_config.NumberColumn(t("alerts_col_target", lang), min_value=0.0, format="%.2f"),
            "active": st.column_config.CheckboxColumn(t("alerts_col_active", lang), default=True),
        })
    if not edited.equals(ss["alerts_rules"]):
        ss["alerts_rules"] = edited.reset_index(drop=True)
        save_rules(ss["alerts_rules"])

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        ss["alerts_sound"] = st.checkbox(t("alerts_sound_label", lang), value=ss["alerts_sound"])
    with c2:
        ss["alerts_notify"] = st.checkbox(t("alerts_notify_label", lang), value=ss["alerts_notify"])
    with c3:
        if ss["alerts_on"]:
            st.button(t("alerts_stop_button", lang), on_click=_set_monitoring, args=(False,), width="stretch")
        else:
            st.button(t("alerts_start_button", lang), type="primary", on_click=_set_monitoring, args=(True,),
                      width="stretch")

    _render_push_panel(lang)
    st.caption(t("alerts_autoplay_note", lang))

    if not any(r["active"] for r in normalize_rules(ss["alerts_rules"])):
        st.warning(t("alerts_no_active_rules", lang))

    st.divider()
    st.fragment(_live_body, run_every=UI_REFRESH_SECONDS)(lang)
