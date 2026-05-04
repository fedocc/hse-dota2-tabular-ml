# Pipeline

В репозитории лежит компактная версия HSE ML homework pipeline для задачи
предсказания исхода матча Dota 2.

## Текущий baseline

Воспроизводимый baseline использует:

- match metadata: `region`, `avg_mmr`;
- target encoding региона;
- индикатор пропущенного MMR;
- `sqrt`-преобразование MMR;
- sparse-кодирование драфта героев:
  - `+1` для героев Radiant;
  - `-1` для героев Dire;
  - `0`, если герой не участвовал в матче;
- логистическую регрессию на итоговой sparse-матрице.

## Очистка данных

Шаг очистки `player_df`:

- оставляет только train/test матчи;
- удаляет пропуски в `match_id`, `account_id`, `hero_id`, `player_slot`;
- удаляет матчи с `hero_id = 0`;
- оставляет валидные Radiant/Dire слоты;
- фильтрует подозрительные duplicated player-side pairs;
- сохраняет только strict 5v5 матчи.

## Метрика

Метрика соревнования — Gini:

```text
Gini = 2 * ROC-AUC - 1
```

Лучший локальный Optuna CV Gini, зафиксированный в этом репозитории: `0.4089`.

