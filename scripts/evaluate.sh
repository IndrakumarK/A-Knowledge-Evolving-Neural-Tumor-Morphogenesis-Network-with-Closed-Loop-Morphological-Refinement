#!/usr/bin/env bash
set -e

python evaluate.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/best.pt \
    --split test