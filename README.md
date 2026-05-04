# HSE Dota 2 Tabular ML

Аккуратно оформленная версия моей домашней работы по курсу машинного обучения в НИУ ВШЭ: предсказание исхода матча Dota 2.

Задача — предсказать, победит ли сторона Radiant. Я оставил репозиторий сфокусированным на воспроизводимом baseline: очистка таблицы игроков, sparse-кодирование драфта героев, признаки по региону/MMR и логистическая регрессия.

## Что внутри

Это не полный dump соревнования и не папка с сырыми ноутбуками. Репозиторий приведён к читаемому проектному виду:

- переиспользуемый код вынесен из ноутбуков в `src/dota_ml`;
- сырые данные не коммитятся;
- сгенерированные submissions не коммитятся;
- ноутбуки оставлены как след экспериментов;
- основной pipeline запускается через `scripts/train.py`.

## Подход

### 1. Очистка player data

Перед построением признаков по героям таблица `player_df` фильтруется:

- оставляются только матчи из train/test;
- удаляются строки с пропусками в `match_id`, `account_id`, `hero_id`, `player_slot`;
- удаляются матчи с некорректным `hero_id = 0`;
- остаются только валидные слоты Radiant/Dire;
- удаляются подозрительные дубли игрок-сторона;
- после очистки остаются только строгие 5v5 матчи.

### 2. Match-level признаки

В baseline используются простые и устойчивые признаки матча:

- target encoding региона;
- индикатор пропуска MMR;
- `sqrt`-преобразование среднего MMR.

### 3. Sparse-кодирование драфта

Герои кодируются sparse-матрицей:

- `+1`, если герой выбран Radiant;
- `-1`, если герой выбран Dire;
- `0`, если герой не участвовал в матче.

Так линейная модель видит составы команд без превращения данных в огромную dense-таблицу.

### 4. Модель

Финальная baseline-модель — логистическая регрессия на sparse-признаках.

Лучшая сохранённая Optuna-конфигурация:

```json
{
  "solver": "lbfgs",
  "C": 0.5868738470888988,
  "max_iter": 1500,
  "random_state": 42
}
```

Основные результаты из экспериментов:

| Группа признаков / setup | CV Gini |
|---|---:|
| Date features | 0.002 |
| Region target encoding | 0.075 |
| Region + MMR | 0.148 |
| Sparse hero draft only | 0.274 |
| Region + MMR + sparse hero draft | 0.308 |
| Packaged baseline, лучший сохранённый Optuna run | 0.4089 |

Метрика соревнования:

```text
Gini = 2 * ROC-AUC - 1
```

## Структура

```text
.
├── configs/
│   └── default.yaml
├── data/
│   └── README.md
├── docs/
│   ├── pipeline.md
│   └── experiment_summary.md
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

## Данные

Сырые данные не лежат в репозитории. Ожидаемая локальная структура:

```text
data/raw/dota-2-hse-ml-1-course-competition-2026/
  matches_df_train.csv
  matches_df_test.csv
  player_df.csv
  dota_adv.csv
  game_chat.csv
  Constants.Heroes.csv
```

Текущий скрипт использует `matches_df_train.csv`, `matches_df_test.csv` и `player_df.csv`.

## Запуск

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

PYTHONPATH=src python scripts/train.py --config configs/default.yaml
```

Submission сохраняется в:

```text
results/submissions/submission_base_all_features.csv
```

## Заметки

Репозиторий намеренно маленький. Тяжёлые артефакты экспериментов, сырые CSV, submissions и локальные кэши остаются вне Git.

Outputs в ноутбуках очищены, чтобы репозиторий было удобно читать. Основные выводы сохранены в [`docs/experiment_summary.md`](docs/experiment_summary.md).

