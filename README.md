# HSE Dota 2 Tabular ML

Cleaned project version of my HSE ML homework on Dota 2 match outcome prediction.

The goal is to predict whether the Radiant side wins a match. I kept the repository focused on one reproducible baseline: careful player-table cleaning, sparse hero draft features, region/MMR preprocessing and logistic regression.

## Project Focus

This repo is not a full competition solution dump. It is a compact, readable version of the work:

- reusable code is moved from notebooks to `src/dota_ml`;
- raw competition data is excluded;
- generated submissions are excluded;
- notebooks are kept only as experiment traces;
- the main pipeline can be launched from `scripts/train.py`.

## Approach

### 1. Player data cleaning

The raw `player_df` table is filtered before building hero features:

- keep only matches from train/test;
- remove missing match, account, hero and slot values;
- remove invalid `hero_id = 0` matches;
- keep valid Radiant/Dire player slots;
- remove suspicious duplicated player-side pairs;
- keep only strict 5v5 matches.

### 2. Match-level features

The baseline uses a small set of robust match features:

- region target encoding;
- MMR missing indicator;
- square-root transformed average MMR.

### 3. Hero draft encoding

Heroes are encoded as a sparse matrix:

- `+1` if a hero is picked by Radiant;
- `-1` if a hero is picked by Dire;
- `0` otherwise.

This representation lets a linear model use draft information without expanding the data into dense categorical columns.

### 4. Model

The final baseline is logistic regression over sparse features.

Best Optuna configuration saved from experiments:

```json
{
  "solver": "lbfgs",
  "C": 0.5868738470888988,
  "max_iter": 1500,
  "random_state": 42
}
```

Recorded best CV result:

| Setup | CV Gini |
|---|---:|
| Region + MMR + sparse hero draft | 0.4089 |

The competition metric is:

```text
Gini = 2 * ROC-AUC - 1
```

## Structure

```text
.
├── configs/
│   └── default.yaml
├── data/
│   └── README.md
├── docs/
│   └── pipeline.md
├── notebooks/
│   ├── 01_baseline_pipeline.ipynb
│   └── 02_advanced_features.ipynb
├── results/
│   ├── metrics/
│   └── submissions/
├── scripts/
│   └── train.py
└── src/
    └── dota_ml/
        ├── __init__.py
        └── pipeline.py
```

## Data

Raw data is not committed. Put the competition files here:

```text
data/raw/dota-2-hse-ml-1-course-competition-2026/
  matches_df_train.csv
  matches_df_test.csv
  player_df.csv
  dota_adv.csv
  game_chat.csv
  Constants.Heroes.csv
```

The current script uses `matches_df_train.csv`, `matches_df_test.csv` and `player_df.csv`.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

PYTHONPATH=src python scripts/train.py --config configs/default.yaml
```

The output submission is written to:

```text
results/submissions/submission_base_all_features.csv
```

## Notes

This repository is intentionally small. The heavier experimental artifacts, raw CSV files, generated submissions and local cache files stay outside Git.

