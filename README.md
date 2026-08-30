# Setup and run

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

CUDA is optional. On CPU, cut `EPOCHS` to 2 and `SEEDS` to one seed in `src/config.py`, or the encoder stage will take hours.

## Verify the harness first (1 minute, no data needed)

```bash
python -m src.smoke_test
```

This builds synthetic score sequences where trajectory carries signal that the last turn does not. A correct build shows `A_last_turn` near chance and `C_features` / `D1_gru_scores` clearly above it. If that holds, any null result on real data is a finding rather than a wiring bug.

## Full pipeline

```bash
bash run_all.sh
```

Or stage by stage:

```bash
python -m src.data_prep                          # 1. build corpora
python -m src.train_turn_encoder --seed 13       # 2. fine-tune + calibrate encoder
python -m src.score_conversations --seed 13      # 3. score ESConv seeker turns
python -m src.evaluate --seed 13                 # 4. trajectory ablation + OOD + stats
python -m src.attribution_service --seed 13 --n 300   # 5. latency decomposition
python -m src.aggregate                          # 6. paper tables
```

Runtime per seed: roughly 25–40 min on a 6 GB consumer GPU, dominated by stage 2.

## If automatic download fails

The container running this may block the dataset hosts. Both corpora are small; fetch them by hand into `data/raw/` and rerun `python -m src.data_prep`.

- **Dreaddit** — `dreaddit-train.csv`, `dreaddit-test.csv`. From the Kaggle mirror (`turcan/dreaddit`) or the LOUHI 2019 release. Columns needed: `text`, `label`, `subreddit`.
- **ESConv** — `ESConv.json` from `thu-coai/Emotional-Support-Conversation` on GitHub, or `load_dataset("thu-coai/esconv")`.

Verify after stage 1: `data/esconv.jsonl` should hold roughly 1,000–1,300 conversations, and `data_prep` prints the label balance. If the positive rate is above 85% or below 15%, adjust `IMPROVEMENT_MARGIN` in `src/config.py` and note the choice in the paper — do not silently tune it to a convenient balance.

## Files

```
PROJECT.md                    intro, objectives, methodology, framework, results tables
src/config.py                 all hyperparameters and split definitions
src/data_prep.py              corpus construction and label derivation
src/train_turn_encoder.py     encoder fine-tuning + temperature scaling
src/score_conversations.py    transfer step: score every seeker turn
src/trajectory_models.py      the five ablation arms
src/evaluate.py               triage metrics, repeated CV, OOD, paired bootstrap
src/attribution_service.py    async attribution worker + latency measurement
src/aggregate.py              cross-seed tables (CSV + LaTeX)
src/smoke_test.py             synthetic verification of the harness
```

## Reporting discipline

`src/config.py` is the single source of truth for every threshold. Anything tuned after seeing test results must be reported as such. The three metric knobs most likely to be quietly optimised are `IMPROVEMENT_MARGIN`, `MIN_SEEKER_TURNS` and `FLAG_THRESHOLD` — fix them before the first full run and record the values in the paper.
