import numpy as np
import pandas as pd


def _map_signal(signal):
    if pd.isna(signal):
        return 0
    mapping = {
        "BUY": 1,
        "SELL": -1,
        "BULLISH": 1,
        "BEARISH": -1,
        "EQUILIBRIO": 0,
        "HOLD": 0,
        "NEUTRAL": 0,
    }
    return mapping.get(str(signal).upper(), 0)


def walk_forward_backtest(df: pd.DataFrame, signal_col: str, target_col: str,
                         train_window: int = 60, test_window: int = 20) -> pd.DataFrame:
    """
    Walk-forward backtest: train on window, test on next window.
    Returns DataFrame with accuracy, win counts per fold.
    """
    if signal_col not in df.columns:
        raise ValueError(f"Missing signal column: {signal_col}")
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    ordered = df.copy().reset_index(drop=True)
    results = []
    n = len(ordered)

    if n < train_window + test_window:
        return pd.DataFrame(columns=[
            "train_start", "train_end", "test_start", "test_end",
            "accuracy", "signal_count_buy", "signal_count_sell", "signal_count_hold"
        ])

    step = max(1, test_window)
    for i in range(0, n - train_window - test_window + 1, step):
        train = ordered.iloc[i:i + train_window]
        test = ordered.iloc[i + train_window:i + train_window + test_window]
        if train.empty or test.empty:
            continue

        train_signals = train[signal_col].map(_map_signal)
        test_targets = pd.to_numeric(test[target_col], errors="coerce").fillna(0.0)
        if test_targets.empty:
            continue

        match = np.sign(train_signals.iloc[: len(test_targets)].to_numpy()) * np.sign(test_targets.to_numpy())
        accuracy = float((match > 0).mean()) if len(match) > 0 else 0.0

        results.append({
            "train_start": i,
            "train_end": i + train_window - 1,
            "test_start": i + train_window,
            "test_end": i + train_window + len(test_targets) - 1,
            "accuracy": accuracy,
            "signal_count_buy": int((train_signals > 0).sum()),
            "signal_count_sell": int((train_signals < 0).sum()),
            "signal_count_hold": int((train_signals == 0).sum()),
        })

    return pd.DataFrame(results)


def compute_strategy_metrics(df: pd.DataFrame, target_col: str = "future_return") -> dict:
    """Compute aggregate performance metrics."""
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    signal_col = "signal" if "signal" in df.columns else "Trend_TimesFM"
    if signal_col not in df.columns:
        raise ValueError(f"Missing signal column: {signal_col}")

    work = df.copy()
    work["signal_score"] = work[signal_col].map(_map_signal).fillna(0)
    returns = pd.to_numeric(work[target_col], errors="coerce").fillna(0.0)
    pnl = returns * work["signal_score"].astype(float)

    return {
        "mean_return": float(pnl.mean()),
        "median_return": float(np.median(pnl)),
        "win_rate": float((pnl > 0).mean()) if len(pnl) else 0.0,
        "avg_pnl": float(pnl.mean()),
        "max_drawdown": float(np.max(np.maximum.accumulate(-pnl))) if len(pnl) else 0.0,
    }
