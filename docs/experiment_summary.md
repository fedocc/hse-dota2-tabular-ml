# Experiment Summary

This file preserves the main experimental conclusions from the original notebooks
without committing raw data, generated submissions or large notebook outputs.

## Data Snapshot

| Object | Shape / Size |
|---|---:|
| Train matches | 641,090 rows |
| Test matches | 59,748 rows |
| Raw player records | 7,617,237 rows |
| Cleaned player records | 6,946,980 rows |

The test split corresponds to December 2024, so validation should be interpreted as an out-of-time problem rather than a fully random i.i.d. split.

## Baseline Feature Groups

| Feature group | CV Gini, mean |
|---|---:|
| Date features | 0.002 |
| Region target encoding | 0.075 |
| Region + MMR | 0.148 |
| Sparse hero draft only | 0.274 |
| Region + MMR + sparse hero draft | 0.308 |

Key observation: date features were almost useless in isolation, while hero draft features gave the largest jump among the baseline feature groups.

## Player Cleaning

The original player table contained invalid and noisy records. The cleaning pipeline:

- removed records outside train/test matches;
- removed invalid `hero_id = 0` matches;
- kept only valid Radiant/Dire player slots;
- removed suspicious duplicated player-side pairs;
- kept strict 5v5 matches after cleaning.

This reduced the player table from `7,617,237` to `6,946,980` rows and made sparse hero encoding reliable.

## Advanced Experiments

Additional notebook experiments explored:

- team chat preprocessing and TF-IDF features;
- gold/experience advantage time-series aggregations;
- scaling for more stable logistic regression optimization;
- richer feature matrices with thousands of sparse columns.

Recorded advanced matrix size:

| Matrix | Shape |
|---|---:|
| Advanced train matrix | 641,090 x 7,795 |
| Advanced test matrix | 59,748 x 7,795 |

## Final Packaged Baseline

The public repository keeps a compact reproducible baseline:

- region target encoding;
- MMR missing indicator and square-root MMR;
- sparse hero draft encoding;
- logistic regression;
- Optuna-selected model parameters.

Best saved Optuna result for the packaged baseline:

```json
{
  "best_cv_gini": 0.4088605023748033,
  "best_params": {
    "solver": "lbfgs",
    "C": 0.5868738470888988,
    "max_iter": 1500,
    "random_state": 42
  }
}
```

## Main Conclusions

1. The strongest simple signal comes from the hero draft, especially when encoded as a signed sparse vector.
2. Region is useful but must be encoded carefully to avoid target leakage.
3. MMR contributes a stable but moderate signal; missingness itself is informative.
4. Chat and advantage features are promising, but they make the pipeline heavier and require more careful preprocessing.
5. A compact linear sparse model is a good baseline because most useful signals are high-dimensional and sparse.

