# NQ Fractal Backtest Engine

A nonstop, cycle-based backtest/refinement script for **Nasdaq futures (NQ)** that:

- Pulls real market bars from Yahoo's chart API when available.
- Falls back to realistic synthetic fractal bars when data is unavailable.
- Re-optimizes strategy parameters every cycle.
- Logs trades and performance metrics.
- Produces HTML chart views so you can inspect entries/exits and equity.

## Run

```bash
python3 nq_fractal_backtest_engine.py --interval 5m --lookback 1200 --sleep 20
```

### Useful flags

- `--cycles 3` run a finite number of cycles for testing.
- `--out artifacts` set output directory.

## Output artifacts

- `artifacts/engine_state.json` latest cycle state and selected parameters.
- `artifacts/trades_log.csv` append-only trade log.
- `artifacts/charts/nq_cycle_####.html` chart view per cycle.

## Notes

- Script is stdlib-first and runs without external Python packages.
- Chart views use `Chart.js` CDN in browser for rendering.
