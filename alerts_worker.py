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
Alerts worker service.

A standalone process (no Streamlit, no browser tab needed) that:
  1. every POLL_INTERVAL_SECONDS makes ONE batched call to the finance service for all the active rules,
  2. evaluates the rules (an alert fires once, re-arms when the condition becomes false),
  3. sends a Web Push notification to every subscribed browser/phone, so the alert arrives
     even when the dashboard tab is closed or the browser is in the background.

Shared with the dashboard through small JSON files (see the FDASH_* variables in `alerts_engine.py`):
rules (read), subscriptions (read/prune), state + heartbeat (write).

Usage:
  python alerts_worker.py              # run forever
  python alerts_worker.py --once       # a single cycle (cron / debugging)
  python alerts_worker.py --test-push  # send a test notification to every subscription and exit
"""

import argparse
import logging
import signal
import time

import alerts_engine as ae

log = logging.getLogger("fdash.alerts_worker")


class _Stop:
    requested = False


def _request_stop(signum, _frame):
    log.info("signal %s received: stopping after the current step", signum)
    _Stop.requested = True


def _sleep(seconds: float):
    """Sleeps in short steps so SIGTERM (docker stop) is honoured quickly."""
    end = time.time() + seconds
    while not _Stop.requested and time.time() < end:
        time.sleep(min(1.0, max(0.0, end - time.time())))


def run_once(state: dict) -> dict:
    state = ae.worker_cycle(state)
    ae.save_state(state)
    for e in state["events"][:5]:
        if e.get("ts") and e["ts"] >= state["heartbeat"] - 1:
            log.info("ALERT %s %s %.2f (price %.2f) -> push %s", e["symbol"], e["condition"], e["target"],
                     e["price"], e.get("push"))
    if state["last_error"]:
        log.warning("finance service: %s", state["last_error"])
    return state


def send_test() -> int:
    from i18n import t  # noqa: F401  (ensures translations are importable)
    subs = ae.load_subscriptions()
    if not subs:
        log.error("no subscriptions: open the dashboard > Alerts and enable background notifications first")
        return 1
    demo = {"symbol": "TEST", "condition": ">=", "target": 100.0, "price": 101.25, "ts": int(time.time())}
    log.info("sending a test notification to %d subscription(s): %s", len(subs), ae.notify_subscribers(demo))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="fdash alerts worker service")
    parser.add_argument("--once", action="store_true", help="run a single cycle and exit")
    parser.add_argument("--test-push", action="store_true", help="send a test notification and exit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    ae.ensure_vapid_keys()  # create the key pair at first start (the dashboard reads the public half)
    if args.test_push:
        return send_test()

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    state = ae.load_state()
    log.info("worker started: interval=%ss rules=%s subscriptions=%s", ae.POLL_INTERVAL_SECONDS,
             ae.CONFIG_PATH, ae.PUSH_SUBS_PATH)
    while not _Stop.requested:
        started = time.time()
        try:
            state = run_once(state)
        except Exception:  # never die: log and retry at the next interval
            log.exception("cycle failed")
        if args.once:
            break
        _sleep(max(1.0, ae.POLL_INTERVAL_SECONDS - (time.time() - started)))
    log.info("worker stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
