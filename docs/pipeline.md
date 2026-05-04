# Pipeline

The repository contains a compact version of the HSE ML homework pipeline for
Dota 2 match outcome prediction.

## Current baseline

The current reproducible baseline uses:

- match metadata: `region`, `avg_mmr`;
- fold-safe target encoding for region;
- missing-value indicator for MMR;
- square-root transformed MMR;
- sparse hero draft encoding:
  - `+1` for Radiant heroes;
  - `-1` for Dire heroes;
  - `0` otherwise;
- logistic regression over the resulting sparse matrix.

## Data cleaning

The `player_df` cleaning step:

- keeps only train/test matches;
- removes missing `match_id`, `account_id`, `hero_id`, `player_slot`;
- removes matches with `hero_id = 0`;
- keeps valid 5v5 Radiant/Dire slots;
- filters suspicious duplicated player-side pairs;
- keeps strict 5v5 matches after cleaning.

## Metric

The competition metric is Gini:

```text
Gini = 2 * ROC-AUC - 1
```

Best local Optuna CV Gini recorded in this repo: `0.4089`.

