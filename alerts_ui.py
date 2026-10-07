"""
Streamlit UI for the "Alerts" section.

Two parts:
  * `render_alert_monitor`  -> small fragment in the sidebar, rendered in EVERY section.
                               It is the one that polls the finance service (1 call every
                               POLL_INTERVAL_SECONDS) and raises sound / notification,
                               so monitoring keeps running while the user is in "Analysis".
  * `render_alerts_section` -> the section itself: rule editor, controls, live prices, log.
"""

import json
import time

import pandas as pd
import streamlit as st

from alerts_engine import (CONDITIONS, MAX_LOG_ENTRIES, POLL_INTERVAL_SECONDS, empty_rules, evaluate_rules,
                           fetch_last_prices, is_triggered, load_rules, make_beep_wav, normalize_rules,
                           rule_key, save_rules)
from i18n import t

UI_REFRESH_SECONDS = 5  # redraw of the live table only (no network calls)


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


# ----------------------------------------------------------------------------- polling + signalling
def _browser_notify(title: str, body: str):
    _html(f"""<script>
    (function() {{
      try {{
        var N = (window.parent && window.parent.Notification) || window.Notification;
        if (N && N.permission === "granted") new N({_js_str(title)}, {{ body: {_js_str(body)} }});
      }} catch (e) {{}}
    }})();
    </script>""", height=1)


def _signal(events: list, lang: str):
    ss = st.session_state
    for e in events:
        title = t("alert_fired_title", lang).format(e["symbol"])
        body = t("alert_fired_body", lang).format(e["symbol"], e["price"], e["condition"], e["target"])
        if ss["alerts_notify"]:
            st.toast(f"**{title}**  \n{body}", icon="🔔")
            _browser_notify(title, body)
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


def _monitor_body(lang: str):
    ss = st.session_state
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
    every = POLL_INTERVAL_SECONDS if st.session_state.get("alerts_on") else None
    st.fragment(_monitor_body, run_every=every)(lang)


# ----------------------------------------------------------------------------- section
def _set_monitoring(value: bool):
    st.session_state["alerts_on"] = value
    if value:
        st.session_state["alerts_last_poll"] = 0.0  # first check immediately
        st.session_state["alerts_fired"] = set()


def _live_body(lang: str):
    ss = st.session_state
    rules = normalize_rules(ss["alerts_rules"])
    st.subheader(t("alerts_live_header", lang))
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

    st.subheader(t("alerts_log_header", lang))
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

    _html(f"""
    <div style="font-family: sans-serif; font-size: 14px;">
      <button style="padding:4px 10px; cursor:pointer;" onclick="
        var N = (window.parent && window.parent.Notification) || window.Notification;
        if (!N) {{ document.getElementById('st').textContent = 'n/a'; return; }}
        N.requestPermission().then(function(p) {{ document.getElementById('st').textContent = p; }});">
        {t("alerts_browser_permission", lang)}</button>
      <span id="st" style="margin-left:8px; opacity:.7;"></span>
    </div>""", height=40)
    st.caption(t("alerts_autoplay_note", lang))

    if not any(r["active"] for r in normalize_rules(ss["alerts_rules"])):
        st.warning(t("alerts_no_active_rules", lang))

    st.divider()
    st.fragment(_live_body, run_every=UI_REFRESH_SECONDS)(lang)
