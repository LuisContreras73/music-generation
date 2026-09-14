# Audio generado

Organizado **por escenario**, porque es de donde sale el prefijo lo que
determina si dos resultados son comparables entre si. Dentro de cada
escenario, una carpeta por modelo.

Cada `.wav` tiene su `.mid` (suena mejor con un piano real) y su `.png`
con el piano-roll, donde la zona sombreada es el prefijo.

## `0_referencia_corpus/` — 3 archivos

musica real del dataset, sin modelo: el objetivo

## `1_continuaciones_corpus/` — 42 archivos

prefijo de una pieza del corpus (caso de entrenamiento)

| modelo | archivos | bits/paso | gen_score |
|---|---|---|---|
| `ablacion_melle_sin_flux` | 3 | 3.2299 | 2.1 |
| `ablacion_sin_atencion_relativa` | 3 | 2.8015 | 15.7 |
| `deep_lstm` | 3 | 2.3592 | 63.5 |
| `estilo_llama_10ep` | 3 | 1.6683 | 62.6 |
| `estilo_llama_24ep` | 3 | 1.5815 | 67.2 |
| `estilo_llama_ctx2048_10ep` | 3 | 1.6077 | 58.9 |
| `estilo_llama_ctx2048_24ep` | 3 | 1.5172 | 79.3 |
| `lstm` | 3 | 2.0424 | 82.8 |
| `melle` | 3 | 3.3319 | 3.2 |
| `music_transformer` | 3 | 1.8276 | 70.6 |
| `music_transformer_con_augmentacion` | 3 | 1.9744 | 51.3 |
| `music_transformer_ctx2048` | 3 | 1.8378 | 51.2 |
| `perceiver_ar` | 3 | 1.7046 | 63.6 |
| `tft` | 3 | 1.8669 | 35.3 |

## `2_prefijos_de_prueba/` — 48 archivos

prefijo de los .npz de evaluacion (caso de la prueba)

| modelo | archivos | bits/paso | gen_score |
|---|---|---|---|
| `melodias_5s` | 30 | — | — |
| `muestras_densas` | 18 | — | — |

## `3_generacion_libre/` — 3 archivos

sin prefijo: el modelo compone desde cero

| modelo | archivos | bits/paso | gen_score |
|---|---|---|---|
| `estilo_llama_ctx2048_24ep` | 3 | 1.5172 | 79.3 |

## Por donde empezar

1. `0_referencia_corpus/` — musica real, para calibrar el oido.
2. `1_continuaciones_corpus/` — el caso para el que se entreno el modelo.
3. `2_prefijos_de_prueba/melodias_5s/` — compara `estilo_llama_10ep` (gen_score 17.4)
   con `estilo_llama_ctx2048_24ep` (gen_score 6.6 pero MEJOR bits/paso). El de mejor
   metrica de prediccion genera peor: es el resultado central del laboratorio.
4. `3_generacion_libre/` — lo que el modelo compone sin ninguna pista.

## Que esperar

El corpus son 714 h de piano interpretado polifonico. Las cinco melodias
conocidas son monofonicas y estan FUERA de esa distribucion: el modelo las
continua en su propio estilo, no reconoce la cancion. El techo medido para
ese escenario es 21.56, no 87.6.

Al escuchar, juzga: mantiene la tonalidad y el registro del prefijo?
respeta su densidad y textura? evita quedarse en bucle?
