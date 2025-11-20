#!/usr/bin/env bash
# Simple wrapper to run training then backtest
python train.py
python -u backtest.py
