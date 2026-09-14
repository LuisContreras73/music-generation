# Mejor modelo: estilo_llama_ctx2048_24ep

Promovido el 2026-09-13T18:55:29-05:00 usando **val_bpt** (menor es mejor).

## Metricas del ganador

- modelo `modern`, familia `token`, 25.78 M parametros
- gen_score: **79.28** / 100
- val_bpt: **1.5172** bits/paso (val_ppl 4.497)
- frame_f1: -
- pasos: 23800, horas de entrenamiento: 1.95
- ultima actualizacion del experimento: 2026-09-13T15:45:26-05:00

## Por que gano

Frente a `estilo_llama_24ep` (best_val_bpt = 1.5815) el ganador alcanza best_val_bpt = 1.5172; diferencia = -0.0643.

## Comparativa (top 5 del leaderboard)

| name | model | family | params_M | best_val_bpt | best_gen_score | steps |
|---|---|---|---|---|---|---|
| lstm | lstm | token | 23.7 | 2.0424 | 82.83 | 4900 |
| estilo_llama_ctx2048_24ep | modern | token | 25.78 | 1.5172 | 79.28 | 23800 |
| music_transformer | music_transformer | token | 25.56 | 1.8276 | 70.63 | 7300 |
| estilo_llama_24ep | modern | token | 25.78 | 1.5815 | 67.18 | 23800 |
| perceiver_ar | perceiver_ar | token | 25.78 | 1.7046 | 63.57 | 9800 |

## Ficheros de esta carpeta

- `best_checkpoint.pt`
- `config.json`
- `distributions.png`
- `generation_01_generation_best.png`
- `generation_02_generation_grid.png`
- `metrics.csv`
- `summary.json`
- `training_curves.png`

---

Regenerar con `python src/registry.py --refresh` (rebuild del leaderboard + figuras + promocion).
