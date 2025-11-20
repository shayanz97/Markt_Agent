# Meme-Trader (Transformer + PnL-aware training + Backtest)

## Quickstart
1. python -m venv venv && source venv/bin/activate
2. pip install -r requirements.txt
3. Edit config in `train.py` (coins, thresholds, paths)
4. ./run_local.sh

## Files
- utils.py: fetch + features + labeling
- train.py: training pipeline (Transformer + PnL-aware loss)
- backtest.py: threshold tuning + final backtest
- Dockerfile + docker-compose: containerized run

## Next steps
- Add walk-forward retraining
- Add real deployment (FastAPI) using model serving
- Add Optuna hyperparameter tuning for pos_threshold, stop_loss_pct, R_SCALE
