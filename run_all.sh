#!/usr/bin/env bash
# Full pipeline. Expect ~25-40 min per seed on a 6GB consumer GPU, longer on CPU.
set -e
python -m src.data_prep
for SEED in 13 29 47 71 97; do
  echo "=== seed $SEED ==="
  python -m src.train_turn_encoder --seed $SEED
  python -m src.score_conversations --seed $SEED
  python -m src.evaluate --seed $SEED
done
python -m src.attribution_service --seed 13 --n 300
python -m src.aggregate
