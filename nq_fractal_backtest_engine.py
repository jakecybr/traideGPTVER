#!/usr/bin/env python3
"""Continuous NQ fractal backtest + refinement engine (stdlib-first)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class StrategyParams:
    short_ma: int
    long_ma: int
    rsi_period: int
    rsi_buy: float
    rsi_sell: float
    atr_period: int
    atr_stop_mult: float


@dataclass
class BacktestStats:
    total_return: float
    sharpe: float
    max_drawdown: float
    trade_count: int
    win_rate: float


class NQFractalBacktestEngine:
    def __init__(self, lookback_bars: int, interval: str, cycle_sleep: int, out_dir: str) -> None:
        self.lookback_bars = lookback_bars
        self.interval = interval
        self.cycle_sleep = cycle_sleep
        self.out_dir = Path(out_dir)
        self.chart_dir = self.out_dir / "charts"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.chart_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.out_dir / "engine_state.json"
        self.trades_file = self.out_dir / "trades_log.csv"

    def run(self, max_cycles: Optional[int]) -> None:
        cycle = 0
        print("▶ NQ fractal engine started.")
        while True:
            cycle += 1
            print(f"\n{'='*72}\nCycle {cycle} - {datetime.now().isoformat(timespec='seconds')}")
            bars = self.get_market_bars()
            if len(bars) < 250:
                print("⚠ Not enough bars; waiting for next cycle.")
                self._sleep_or_break(cycle, max_cycles)
                continue

            self.add_features(bars)
            best_params, train_stats = self.optimize_parameters(bars)
            bt, full_stats, trades = self.backtest(bars, best_params)

            self.save_outputs(cycle, bt, trades, best_params, full_stats, train_stats)
            self.print_cycle_summary(cycle, bt, best_params, full_stats, trades)

            self._sleep_or_break(cycle, max_cycles)

    def _sleep_or_break(self, cycle: int, max_cycles: Optional[int]) -> None:
        if max_cycles is not None and cycle >= max_cycles:
            print("■ Max cycles reached.")
            raise SystemExit(0)
        time.sleep(self.cycle_sleep)

    def get_market_bars(self) -> List[Dict]:
        real = self.fetch_yahoo_chart_api()
        if real:
            print(f"✓ Loaded {len(real)} real bars from Yahoo (NQ=F, {self.interval}).")
            return real[-self.lookback_bars :]
        print("⚠ Real bar pull unavailable. Using synthetic fractal bars.")
        return self.generate_synthetic_fractal_bars(self.lookback_bars)

    def fetch_yahoo_chart_api(self) -> List[Dict]:
        q = urllib.parse.urlencode({"interval": self.interval, "range": "5d", "includePrePost": "false"})
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/NQ=F?{q}"
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            print(f"⚠ Yahoo API fetch failed: {exc}")
            return []

        try:
            result = payload["chart"]["result"][0]
            ts = result["timestamp"]
            quote = result["indicators"]["quote"][0]
            bars = []
            for i, t in enumerate(ts):
                o, h, l, c, v = quote["open"][i], quote["high"][i], quote["low"][i], quote["close"][i], quote["volume"][i]
                if None in (o, h, l, c):
                    continue
                bars.append(
                    {
                        "time": datetime.fromtimestamp(t, tz=timezone.utc).isoformat(),
                        "open": float(o),
                        "high": float(h),
                        "low": float(l),
                        "close": float(c),
                        "volume": int(v or 0),
                    }
                )
            return bars
        except Exception as exc:
            print(f"⚠ Yahoo payload parse failed: {exc}")
            return []

    def generate_synthetic_fractal_bars(self, n: int) -> List[Dict]:
        now = int(time.time())
        step_sec = 300 if self.interval == "5m" else 60
        base = 18000.0
        price = base
        bars = []

        wave_state = 0.0
        for i in range(n):
            t = now - (n - i) * step_sec
            noise = random.gauss(0, 1.0)
            wave_state = 0.92 * wave_state + random.gauss(0, 0.6)
            fractal_component = 0.6 * noise + 0.3 * wave_state + 0.1 * math.sin(i / 18.0)
            ret = fractal_component * 0.0009

            open_ = price
            close = open_ * math.exp(ret)
            spread = abs(random.gauss(0, 6.5)) + 0.8
            high = max(open_, close) + spread
            low = min(open_, close) - spread
            volume = int(random.randint(600, 9000))

            bars.append(
                {
                    "time": datetime.fromtimestamp(t, tz=timezone.utc).isoformat(),
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )
            price = close
        return bars

    def add_features(self, bars: List[Dict]) -> None:
        closes = [b["close"] for b in bars]
        highs = [b["high"] for b in bars]
        lows = [b["low"] for b in bars]

        rsi = self.compute_rsi(closes, 14)
        atr = self.compute_atr(highs, lows, closes, 14)
        hurst = self.compute_hurst(closes, 80)

        for i, bar in enumerate(bars):
            bar["rsi"] = rsi[i]
            bar["atr"] = atr[i]
            bar["hurst"] = hurst[i]

    def compute_rsi(self, closes: List[float], period: int) -> List[Optional[float]]:
        out = [None] * len(closes)
        gains, losses = [], []
        for i in range(1, len(closes)):
            d = closes[i] - closes[i - 1]
            gains.append(max(d, 0.0))
            losses.append(max(-d, 0.0))
            if i >= period:
                avg_gain = sum(gains[i - period : i]) / period
                avg_loss = sum(losses[i - period : i]) / period
                rs = avg_gain / (avg_loss + 1e-9)
                out[i] = 100 - (100 / (1 + rs))
        return out

    def compute_atr(self, highs: List[float], lows: List[float], closes: List[float], period: int) -> List[Optional[float]]:
        tr = [None]
        for i in range(1, len(closes)):
            trv = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
            tr.append(trv)
        out = [None] * len(closes)
        for i in range(period, len(closes)):
            segment = [x for x in tr[i - period + 1 : i + 1] if x is not None]
            out[i] = sum(segment) / len(segment)
        return out

    def compute_hurst(self, closes: List[float], window: int) -> List[Optional[float]]:
        out = [None] * len(closes)
        for i in range(window, len(closes)):
            vals = closes[i - window : i]
            lags = list(range(2, 12))
            tau = []
            for lag in lags:
                diffs = [vals[j + lag] - vals[j] for j in range(len(vals) - lag)]
                std = statistics.pstdev(diffs) if len(diffs) > 1 else 0.0
                tau.append(math.sqrt(max(std, 1e-9)))
            x = [math.log(l) for l in lags]
            y = [math.log(t) for t in tau]
            slope = self._linreg_slope(x, y)
            out[i] = max(0.0, min(1.0, slope * 2.0))
        return out

    def _linreg_slope(self, x: List[float], y: List[float]) -> float:
        xm = statistics.mean(x)
        ym = statistics.mean(y)
        num = sum((xi - xm) * (yi - ym) for xi, yi in zip(x, y))
        den = sum((xi - xm) ** 2 for xi in x) + 1e-12
        return num / den

    def optimize_parameters(self, bars: List[Dict]) -> Tuple[StrategyParams, BacktestStats]:
        split = int(len(bars) * 0.7)
        train = bars[:split]
        best_params = None
        best_stats = None
        best_score = -1e18

        for s, l, rb, rs, atm in product([8, 13, 21], [34, 55, 89], [35, 40, 45], [55, 60, 65], [1.2, 1.8, 2.4]):
            if s >= l or rb >= rs:
                continue
            params = StrategyParams(s, l, 14, rb, rs, 14, atm)
            _, stats, _ = self.backtest(train, params)
            score = (stats.total_return * 100) + (stats.sharpe * 4) + (stats.win_rate * 25) - (abs(stats.max_drawdown) * 35)
            if stats.trade_count < 4:
                score -= 30
            if score > best_score:
                best_score, best_params, best_stats = score, params, stats

        assert best_params and best_stats
        return best_params, best_stats

    def backtest(self, bars: List[Dict], params: StrategyParams) -> Tuple[List[Dict], BacktestStats, List[Dict]]:
        data = [dict(b) for b in bars]
        closes = [b["close"] for b in data]
        ma_fast = self.sma(closes, params.short_ma)
        ma_slow = self.sma(closes, params.long_ma)

        pos, entry, stop = 0, 0.0, None
        equity = 1.0
        trades: List[Dict] = []
        equity_curve: List[float] = []

        for i, bar in enumerate(data):
            bar["ma_fast"] = ma_fast[i]
            bar["ma_slow"] = ma_slow[i]
            ready = all(
                [
                    ma_fast[i] is not None,
                    ma_slow[i] is not None,
                    bar.get("rsi") is not None,
                    bar.get("atr") is not None,
                    bar.get("hurst") is not None,
                ]
            )
            price = bar["close"]
            if not ready:
                bar["position"] = pos
                bar["equity"] = equity
                equity_curve.append(equity)
                continue

            trend_mode = bar["hurst"] >= 0.55
            cross_up = ma_fast[i] > ma_slow[i]
            cross_down = ma_fast[i] < ma_slow[i]

            buy_signal = (cross_up and bar["rsi"] <= params.rsi_buy) if not trend_mode else (cross_up and bar["rsi"] <= 55)
            sell_signal = (cross_down and bar["rsi"] >= params.rsi_sell) if not trend_mode else (cross_down and bar["rsi"] >= 45)

            if pos == 1 and stop is not None and price <= stop:
                pnl = (price - entry) / entry
                equity *= (1 + pnl)
                trades.append({"time": bar["time"], "side": "EXIT_STOP", "price": price, "pnl": pnl, "equity": equity})
                pos = 0

            if pos == 0 and buy_signal:
                pos = 1
                entry = price
                stop = price - bar["atr"] * params.atr_stop_mult
                trades.append({"time": bar["time"], "side": "BUY", "price": price, "pnl": 0.0, "equity": equity})
            elif pos == 1 and sell_signal:
                pnl = (price - entry) / entry
                equity *= (1 + pnl)
                trades.append({"time": bar["time"], "side": "SELL", "price": price, "pnl": pnl, "equity": equity})
                pos = 0

            marked_eq = equity if pos == 0 else equity * (price / entry)
            bar["position"] = pos
            bar["equity"] = marked_eq
            equity_curve.append(marked_eq)

        rets = [0.0]
        for i in range(1, len(equity_curve)):
            prev = equity_curve[i - 1]
            rets.append((equity_curve[i] - prev) / prev if prev != 0 else 0.0)

        total_return = equity_curve[-1] - 1.0
        mean_ret = statistics.mean(rets)
        std_ret = statistics.pstdev(rets)
        annual = math.sqrt(252 * 78) if self.interval == "5m" else math.sqrt(252)
        sharpe = (mean_ret / (std_ret + 1e-9)) * annual

        peak, max_dd = 1.0, 0.0
        for e in equity_curve:
            peak = max(peak, e)
            dd = (e - peak) / peak
            max_dd = min(max_dd, dd)

        closed = [t for t in trades if t["side"] in {"SELL", "EXIT_STOP"}]
        wins = len([t for t in closed if t["pnl"] > 0])
        win_rate = wins / len(closed) if closed else 0.0

        stats = BacktestStats(total_return, sharpe, max_dd, len(closed), win_rate)
        return data, stats, trades

    def sma(self, vals: List[float], period: int) -> List[Optional[float]]:
        out = [None] * len(vals)
        for i in range(period - 1, len(vals)):
            out[i] = sum(vals[i - period + 1 : i + 1]) / period
        return out

    def save_outputs(
        self,
        cycle: int,
        bt: List[Dict],
        trades: List[Dict],
        params: StrategyParams,
        full_stats: BacktestStats,
        train_stats: BacktestStats,
    ) -> None:
        chart_path = self.chart_dir / f"nq_cycle_{cycle:04d}.html"
        self.write_html_chart(bt[-450:], trades, chart_path)

        latest = bt[-1]
        state = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "cycle": cycle,
            "latest_bar": {k: latest[k] for k in ["time", "open", "high", "low", "close", "volume"]},
            "best_params": asdict(params),
            "train_stats": asdict(train_stats),
            "full_stats": asdict(full_stats),
            "latest_chart": str(chart_path),
        }
        self.state_file.write_text(json.dumps(state, indent=2))

        if trades:
            write_header = not self.trades_file.exists()
            with self.trades_file.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["cycle", "time", "side", "price", "pnl", "equity"])
                if write_header:
                    writer.writeheader()
                for row in trades:
                    writer.writerow({"cycle": cycle, **row})

    def write_html_chart(self, bt: List[Dict], trades: List[Dict], out_path: Path) -> None:
        labels = [b["time"] for b in bt]
        closes = [b["close"] for b in bt]
        fast = [b.get("ma_fast") for b in bt]
        slow = [b.get("ma_slow") for b in bt]
        equity = [b.get("equity") for b in bt]

        buy_map = {t["time"]: t["price"] for t in trades if t["side"] == "BUY"}
        sell_map = {t["time"]: t["price"] for t in trades if t["side"] in {"SELL", "EXIT_STOP"}}
        buys = [buy_map.get(ts) for ts in labels]
        sells = [sell_map.get(ts) for ts in labels]

        html = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>NQ Fractal Cycle</title>
<script src='https://cdn.jsdelivr.net/npm/chart.js'></script></head>
<body style='font-family:Arial;padding:16px;'>
<h2>NQ Fractal Strategy View</h2>
<canvas id='price' height='110'></canvas>
<canvas id='eq' height='80'></canvas>
<script>
const labels={json.dumps(labels)};
const closeData={json.dumps(closes)};
const fastData={json.dumps(fast)};
const slowData={json.dumps(slow)};
const buyData={json.dumps(buys)};
const sellData={json.dumps(sells)};
const eqData={json.dumps(equity)};
new Chart(document.getElementById('price'),{{type:'line',data:{{labels,datasets:[
{{label:'Close',data:closeData,borderWidth:1}},
{{label:'Fast MA',data:fastData,borderWidth:1}},
{{label:'Slow MA',data:slowData,borderWidth:1}},
{{label:'Buy',data:buyData,showLine:false,pointRadius:4,pointStyle:'triangle'}},
{{label:'Sell',data:sellData,showLine:false,pointRadius:4,pointStyle:'triangle',rotation:180}},
]}}}});
new Chart(document.getElementById('eq'),{{type:'line',data:{{labels,datasets:[{{label:'Equity',data:eqData,borderWidth:1}}]}}}});
</script></body></html>"""
        out_path.write_text(html)

    def print_cycle_summary(self, cycle: int, bt: List[Dict], params: StrategyParams, stats: BacktestStats, trades: List[Dict]) -> None:
        b = bt[-1]
        print(f"Latest: O={b['open']:.2f} H={b['high']:.2f} L={b['low']:.2f} C={b['close']:.2f} V={b['volume']}")
        print(f"Params: {params}")
        print(
            f"Performance: return={stats.total_return*100:.2f}% sharpe={stats.sharpe:.2f} "
            f"maxDD={stats.max_drawdown*100:.2f}% trades={stats.trade_count} winRate={stats.win_rate*100:.2f}%"
        )
        recent = [t for t in trades if t["side"] in {"SELL", "EXIT_STOP"}][-5:]
        if recent:
            print("Recent closed trades:")
            for r in recent:
                print(f"  {r['time']} {r['side']:>9} price={r['price']:.2f} pnl={r['pnl']*100:.2f}% equity={r['equity']:.4f}")
        print(f"Chart: {self.chart_dir / f'nq_cycle_{cycle:04d}.html'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Nonstop NQ backtest + refinement")
    p.add_argument("--interval", default="5m")
    p.add_argument("--lookback", type=int, default=1200)
    p.add_argument("--sleep", type=int, default=20)
    p.add_argument("--cycles", type=int, default=None)
    p.add_argument("--out", default="artifacts")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    engine = NQFractalBacktestEngine(args.lookback, args.interval, args.sleep, args.out)
    engine.run(args.cycles)


if __name__ == "__main__":
    main()
