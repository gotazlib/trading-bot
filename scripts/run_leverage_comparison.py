"""Testet das beste System (Top-3 Long+Short) mit verschiedenen Hebel-Stufen.

Empirische Antwort auf 'mit Hebel 1-2 % pro Tag rausholen': wir vergleichen
2x, 3x, 5x, 10x mit Liquidations-Logik.
"""
import importlib
import sys
import warnings

import numpy as np

import scripts.run_multi_asset_trend as base

warnings.filterwarnings("ignore", category=UserWarning)

LEVERAGES = [2.0, 3.0, 5.0, 10.0]


def run_with_leverage(lev):
    """Setze Hebel und fuehre das Multi-Asset-Backtest aus, sammle Ergebnis."""
    base.LEVERAGE = lev
    # reimport um sicher zu stellen dass module-Zustand stimmt
    importlib.reload(base)
    base.LEVERAGE = lev

    # Daten + ML pro Coin (nur einmal teuer)
    coins_prepared = {}
    for sym in base.SYMBOLS:
        result = base.prepare_coin(sym)
        if result is None:
            continue
        coins_prepared[sym] = result

    coins_oos = {}
    for sym, (data, features) in coins_prepared.items():
        oos, _ = base.run_walk_forward_ml(data, features)
        if oos.empty:
            continue
        oos["signal"] = (
            (oos["close"] >= oos["donchian_high"])
            & (oos["p_trend"] >= base.ML_PROB_THRESHOLD)
        )
        oos["short_signal"] = (
            (oos["close"] <= oos["donchian_low"])
            & (oos["p_short"] >= base.ML_PROB_THRESHOLD + base.SHORT_BIAS_EXTRA)
        )
        coins_oos[sym] = oos[["close", "high", "low", "atr",
                              "p_trend", "p_short", "signal", "short_signal"]]

    res = base.multi_asset_trend_backtest(
        coins_data=coins_oos,
        starting_capital=base.STARTING_CAPITAL_CHF,
        fee_rate=base.FEE_RATE, leverage=lev,
        atr_mult=base.ATR_TRAILING_MULT,
        max_position_per_coin=base.MAX_POSITION_PER_COIN,
        max_total_exposure=base.MAX_TOTAL_EXPOSURE,
        daily_loss_limit=base.DAILY_LOSS_LIMIT,
        allow_shorts=base.ALLOW_SHORTS,
    )
    trades = res["trades"]
    equity = res["equity"]
    day_log = res["day_log"]
    n_d = len(day_log) if len(day_log) else 1
    return {
        "lev": lev,
        "final": res["final_capital"],
        "ret": res["total_return"],
        "trades": len(trades),
        "wr": (trades["net_return"] > 0).mean() if len(trades) else 0,
        "liq": (trades["reason"] == "LIQUIDATED").sum() if len(trades) else 0,
        "max_dd": (equity / equity.cummax() - 1).min() if len(equity) else 0,
        "best_day": day_log["day_return"].max() if len(day_log) else 0,
        "worst_day": day_log["day_return"].min() if len(day_log) else 0,
        "active_days": int((day_log["day_return"] != 0).sum()) if len(day_log) else 0,
        "n_days": n_d,
        "win_days": int((day_log["day_return"] > 0).sum()) if len(day_log) else 0,
        "loss_days": int((day_log["day_return"] < 0).sum()) if len(day_log) else 0,
    }


def main():
    print(f"Symbols: {base.SYMBOLS}")
    print(f"Test ueber {len(LEVERAGES)} Hebel-Stufen …\n")

    results = []
    for lev in LEVERAGES:
        print(f"--- Hebel {lev}x ---")
        r = run_with_leverage(lev)
        results.append(r)

    print("\n" + "=" * 88)
    print(f"=== HEBEL-VERGLEICH (gleicher OOS-Zeitraum) ===")
    print("=" * 88)
    print(f"  {'Hebel':>6s} {'Endkap.':>10s} {'Return':>9s} {'MaxDD':>8s} "
          f"{'Trades':>7s} {'WR':>6s} {'Liq':>4s} "
          f"{'Best/Worst Day':>18s} {'Win/Loss Tage':>15s}")
    for r in results:
        print(f"  {r['lev']:>5.1f}x "
              f"{r['final']:>10.2f} "
              f"{r['ret']*100:>+8.2f}% "
              f"{r['max_dd']*100:>+7.2f}% "
              f"{r['trades']:>7d} "
              f"{r['wr']*100:>5.1f}% "
              f"{r['liq']:>4d} "
              f"{r['best_day']*100:>+6.2f}/{r['worst_day']*100:>+6.2f}% "
              f"{r['win_days']:>4d}/{r['loss_days']:>4d}")

    # Ziel-Check: 1 % pro Tag waere?
    print("\n--- Ziel-Check: 1 %/Tag Compound ueber den OOS-Zeitraum ---")
    n_d = results[0]["n_days"]
    target_1p = (1.01 ** n_d - 1) * 100
    target_2p = (1.02 ** n_d - 1) * 100
    print(f"  {n_d} Tage Test-Zeitraum")
    print(f"  +1 %/Tag compound = {target_1p:>14,.0f} % Return  ({1000 * (1.01**n_d):>16,.0f} CHF)")
    print(f"  +2 %/Tag compound = {target_2p:>14,.0f} % Return  ({1000 * (1.02**n_d):>16,.0f} CHF)")
    print(f"  Bester Realwert:      {max(r['ret'] for r in results)*100:>+8.2f} %")


if __name__ == "__main__":
    main()
