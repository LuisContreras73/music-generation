# Resultados

Generado por `scripts/05_report.py` a partir de `reports/leaderboard.csv` y de los
`summary.json` de cada experimento. Todos comparten tokenización, particiones y
protocolo de evaluación. El presupuesto de entrenamiento **no** es el mismo para
todos: hay tres tandas (120 M tokens ≈ 3.7 épocas equivalentes, 321 M ≈ 10 y
780 M ≈ 24 épocas equivalentes), y solo son comparables entre sí los experimentos
que comparten tanda. Las columnas de pasos y horas permiten distinguirlas.

## 1. Comparativa

| # | experimento | modelo | familia | par. (M) | bits/paso val | bits/paso test | gen_score | horas |
|---|---|---|---|---|---|---|---|---|
| 1 | `lstm` | LSTM (linea base recurrente) | eventos | 23.7 | **2.042** | 2.022 | **82.8** | 0.50 |
| 2 | `estilo_llama_ctx2048_24ep` | Decoder estilo LLaMA (RoPE + RMSNorm + SwiGLU) | eventos | 25.8 | **1.517** | 1.536 | **79.3** | 1.95 |
| 3 | `music_transformer` | Music Transformer (atencion relativa) | eventos | 25.6 | **1.828** | 1.810 | **70.6** | 0.42 |
| 4 | `estilo_llama_24ep` | Decoder estilo LLaMA (RoPE + RMSNorm + SwiGLU) | eventos | 25.8 | **1.582** | 1.568 | **67.2** | 1.78 |
| 5 | `perceiver_ar` | Perceiver AR (cuello latente + cross-attend) | eventos | 25.8 | **1.705** | 1.688 | **63.6** | 0.68 |
| 6 | `deep_lstm` | LSTM profunda (4 capas x 1280) | eventos | 27.0 | **2.359** | n/d | **63.5** | 0.25 |
| 7 | `estilo_llama_10ep` | Decoder estilo LLaMA (RoPE + RMSNorm + SwiGLU) | eventos | 25.8 | **1.668** | 1.652 | **62.6** | 0.74 |
| 8 | `estilo_llama_ctx2048_10ep` | Decoder estilo LLaMA (RoPE + RMSNorm + SwiGLU) | eventos | 25.8 | **1.608** | 1.625 | **58.9** | 0.84 |
| 9 | `music_transformer_con_augmentacion` | Music Transformer (atencion relativa) | eventos | 25.6 | **1.974** | 1.952 | **51.3** | 2.21 |
| 10 | `music_transformer_ctx2048` | Music Transformer (atencion relativa) | eventos | 25.8 | **1.838** | 1.852 | **51.2** | 0.59 |
| 11 | `tft` | Temporal Fusion Transformer adaptado | eventos | 23.2 | **1.867** | 1.851 | **35.3** | 0.38 |
| 12 | `ablacion_sin_atencion_relativa` | Music Transformer (atencion relativa) | eventos | 25.3 | **2.802** | 2.734 | **15.7** | 0.24 |
| 13 | `melle` | MELLE adaptado a frames | frames | 20.4 | **3.332** | 3.215 | **3.2** | 0.21 |
| 14 | `ablacion_melle_sin_flux` | MELLE adaptado a frames | frames | 20.4 | **3.230** | n/d | **2.1** | 0.22 |

Anclas de referencia medidas empiricamente:

| ancla | bits/paso | gen_score |
|---|---|---|
| modelo i.i.d. con la densidad marginal | 4.0921 | - |
| ruido i.i.d. con la densidad correcta | - | 1.6 |
| fragmentos reales del corpus | - | 87.6 |
| silencio o bucle degenerado | - | 0.0 |

> **Aviso: 1 experimento(s) no completaron su presupuesto de tokens.**
> Su bits/paso es valido, pero **no** se comparan a igualdad de computo:
>
> - `deep_lstm`: 1900 de 4900 pasos (39 %)

## 2. Modelo ganador

**`estilo_llama_ctx2048_24ep`** (Decoder estilo LLaMA (RoPE + RMSNorm + SwiGLU)), por **bits/paso** = 1.517.

Se elige por bits/paso y no por `gen_score` porque bits/paso es **determinista**:
se mide por *teacher forcing* sobre el split de validación entero, sin muestreo y
sin sortear prefijos. El `gen_score` de la tabla es el **máximo** sobre las
evaluaciones del entrenamiento, y cada evaluación usaba prefijos distintos. Medido
sobre `estilo_llama_ctx2048_24ep`: **79.3** en el paso 22500 y **62.7** en el 23800,
con el `val_bpt` pasando de 1.5185 a 1.5172, es decir el mismo modelo. Con un ruido
de 10 a 22 puntos, una diferencia pequeña de `gen_score` no es interpretable: que la
tabla la encabece `lstm` **no** significa que genere mejor. La comparación
pareada de `scripts/18_comparacion_pareada.py` --mismos prefijos de test para todos
y varias semillas de muestreo-- sustituye a ese criterio.


- bits/paso de validacion: **1.517** (62.9% por debajo de la referencia trivial)
- bits/paso de test: 1.536
- perplejidad por token: 4.50
- probabilidad asignada a una nota imposible: 0.000296

Sus artefactos estan sincronizados en `reports/best/`: checkpoint, `metrics.csv`,
`training_curves.png`, piano-rolls y MIDI.

## 3. Ablacion: aporta algo la atencion relativa?

`music_transformer` y `ablacion_sin_atencion_relativa` son la MISMA red con el mismo numero de parametros y el
mismo presupuesto de tokens; solo cambia `rel_attn`.

| | bits/paso val | gen_score | par. (M) |
|---|---|---|---|
| atencion relativa (`music_transformer`) | 1.828 | 70.6 | 25.6 |
| atencion absoluta (`ablacion_sin_atencion_relativa`) | 2.802 | 15.7 | 25.3 |

Diferencia: **+0.9740 bits/paso** y **+54.9 puntos de gen_score** a favor de la
atencion relativa.

## 4. Contexto: 1024 frente a 2048 tokens

| contexto | bits/paso val | gen_score |
|---|---|---|
| 1024 tokens (~78 s) | 1.828 | 70.6 |
| 2048 tokens (~154 s) | 1.838 | 51.2 |

## 5. Eventos frente a frames

El mejor modelo de eventos (`estilo_llama_ctx2048_24ep`) logra 1.517 bits/paso; el mejor de frames (`ablacion_melle_sin_flux`),
3.230. La comparacion es legitima porque ambos miden -log2 p(piano_roll) / pasos
sobre el mismo objeto.

La familia de frames arrastra una limitacion estructural: predice las 88 notas de
un mismo paso de forma condicionalmente independiente, asi que no puede modelar la
correlacion de un acorde. La familia de eventos si, porque emite las notas
simultaneas en secuencia y cada una condiciona a la siguiente.

## 6. Por que la familia de frames genera demasiadas notas

`scripts/08_melle_drift.py` descompone el exceso de densidad en sus dos factores
multiplicativos, midiendo por separado lo que el modelo ya sobreestima con datos
reales (teacher forcing) y lo que anade la realimentacion al muestrear.

| experimento | flux_weight | calibracion | realimentacion | total | crecimiento |
|---|---|---|---|---|---|
| `ablacion_melle_sin_flux` | 0.00 | 1.99x | 4.78x | **9.53x** | 1.77x |

Con `ablacion_melle_sin_flux`: el modelo espera 0.92 notas por frame cuando la realidad es 0.46
(factor 1.99), y al muestrear llega a 4.40 notas por paso (otro factor 4.78).

## 7. Comparacion pareada: el mismo material para todos

`scripts/18_comparacion_pareada.py`. Los mismos **16 prefijos del split de
test** para todos los modelos, **3 semillas** de muestreo con generador
explicito, y el checkpoint `best.pt` (elegido por `val_bpt`, que es
determinista) en vez de `best_gen.pt` (elegido por el `gen_score` ruidoso).
La columna *leaderboard* es el maximo que reportaba la tabla 1.

### Muestreo del entrenamiento (t 1.0, top-p 0.95)

Decodificacion: el mismo con el que se produjo la tabla 1.

| modelo | bits/paso | gen_score pareado | desv. | rango | leaderboard |
|---|---|---|---|---|---|
| `lstm` | 2.0424 | **78.91** | ±2.99 | 75.55 - 81.29 | 82.83 |
| `estilo_llama_24ep` | 1.5815 | **74.37** | ±15.32 | 57.31 - 86.94 | 67.18 |
| `music_transformer` | 1.8276 | **64.3** | ±22.62 | 38.21 - 78.42 | 70.63 |
| `deep_lstm` | 2.3592 | **62.59** | ±2.15 | 60.3 - 64.56 | 63.49 |
| `perceiver_ar` | 1.7046 | **50.62** | ±2.59 | 47.95 - 53.12 | 63.57 |
| `estilo_llama_ctx2048_24ep` | 1.5172 | **49.27** | ±2.42 | 46.48 - 50.76 | 79.28 |

### Muestreo calibrado anti-bucle

Decodificacion: `data/processed/best_sampling.json`.

| modelo | bits/paso | gen_score pareado | desv. | rango | leaderboard |
|---|---|---|---|---|---|
| `estilo_llama_ctx2048_24ep` | 1.5172 | **51.31** | ±2.52 | 48.56 - 53.52 | 79.28 |
| `estilo_llama_24ep` | 1.5815 | **48.47** | ±1.94 | 46.24 - 49.78 | 67.18 |
| `perceiver_ar` | 1.7046 | **48.27** | ±3.06 | 45.99 - 51.75 | 63.57 |
| `deep_lstm` | 2.3592 | **46.26** | ±0.34 | 45.99 - 46.64 | 63.49 |
| `lstm` | 2.0424 | **41.61** | ±4.0 | 37.0 - 44.14 | 82.83 |
| `music_transformer` | 1.8276 | **37.59** | ±1.91 | 36.09 - 39.74 | 70.63 |

### Lectura: la decodificacion calibrada **sobrecorrige**

La referencia real del corpus es **densidad 0.451** y **repeat8 0.238**.
Las dos decodificaciones fallan por lados opuestos:

| modelo | densidad (entren.) | densidad (calibr.) | repeat8 (entren.) | repeat8 (calibr.) | Δ score |
|---|---|---|---|---|---|
| `lstm` | 0.558 | 0.306 | 0.162 | 0.130 | **-37.3** |
| `music_transformer` | 0.556 | 0.298 | 0.294 | 0.130 | **-26.7** |
| `estilo_llama_24ep` | 0.531 | 0.335 | 0.278 | 0.128 | **-25.9** |
| `deep_lstm` | 0.557 | 0.354 | 0.165 | 0.067 | **-16.3** |
| `perceiver_ar` | 0.627 | 0.340 | 0.274 | 0.111 | **-2.4** |
| `estilo_llama_ctx2048_24ep` | 0.593 | 0.353 | 0.318 | 0.086 | **+2.0** |

El muestreo del entrenamiento genera **de mas** (densidad por encima de la real en
todos los modelos) y el calibrado genera **de menos** (por debajo en todos), con
errores de magnitud parecida y signo contrario. Con `repeat8` pasa igual: el
calibrado deja a todos los modelos **menos repetitivos que la musica real**, lo
cual tambien es un error, solo que del otro lado.

El unico modelo que mejora con el calibrado es el que de verdad tenia el problema
de bucles (`estilo_llama_ctx2048_24ep`, repeat8 0.318 -> 0.086). El `lstm`, que ya
estaba por debajo de la repeticion real sin ninguna ayuda, **pierde 37 puntos**:
aplicarle una penalizacion por repeticion es quitarle lo que no le sobraba.

Conclusion, que es mas precisa que la que se tenia antes: **la decodificacion
domina la calidad de la generacion** --hasta 37 puntos sobre un modelo y unos
prefijos fijos, mas que cualquier diferencia de arquitectura medida aqui-- pero
**no existe una decodificacion buena en abstracto**. `best_sampling.json` se
calibro sobre melodias monofonicas fuera de distribucion, donde el fallo era el
bucle; llevado a prefijos polifonicos del corpus, corrige un problema que la
mayoria de los modelos no tenia.

## 8. En la metrica del paper de Music Transformer

Huang et al. (2018, arXiv:1809.04281) reportan **NLL por token**; nuestra lectura,
marcada como no confirmada en `docs/05_estado_del_arte.md`, es que son nats.
`scripts/21_metricas_paper.py` reexpresa nuestros modelos en esa unidad.

| modelo | NLL val (nats/token) | NLL test | bits/token | bits/paso |
|---|---|---|---|---|
| `estilo_llama_ctx2048_96ep` | **1.4861** | 1.5441 | 2.144 | 1.4996 |
| `estilo_llama_ctx2048_24ep` | **1.5035** | 1.5441 | 2.1691 | 1.5172 |
| `estilo_llama_24ep` | **1.5788** | 1.5971 | 2.2778 | 1.5815 |
| `estilo_llama_ctx2048_10ep` | **1.5932** | 1.6326 | 2.2984 | 1.6077 |
| `estilo_llama_10ep` | **1.6654** | 1.6824 | 2.4026 | 1.6683 |
| `perceiver_ar` | **1.7017** | 1.7184 | 2.455 | 1.7046 |
| `music_transformer_ctx2048` | **1.8212** | 1.8614 | 2.6275 | 1.8378 |
| `music_transformer` | **1.8244** | 1.843 | 2.6321 | 1.8276 |

> **No compares estas cifras con el 1.84 del paper.** Su vocabulario son 388
> simbolos CON velocity y NOTE_OFF sobre MAESTRO; el nuestro son 155 SIN velocity,
> duracion ni pedal, sobre otro corpus y otra rejilla temporal. Una NLL por token
> depende del vocabulario y del dato. Que a `music_transformer` le salga 1.8430 es
> una coincidencia aritmetica, no un empate.

Y lo mas importante del paper para este informe no es esa cifra: su evidencia de
**calidad** no es una metrica automatica, es un **test de escucha por pares** con
humanos. Es decir, el propio paper ya asume que la NLL no mide si suena bien.

## 9. Generacion libre: sin ningun prefijo

`scripts/20_generacion_libre.py`. La entrada es un unico token `BOS`: el modelo
compone desde cero, sin nada que lo ancle. Mismas semillas y misma decodificacion
para todos.

| modelo | gen_score | densidad | repeat8 | polifonia |
|---|---|---|---|---|
| `lstm` | **70.52** | 0.395 | 0.249 | 2.03 |
| `estilo_llama_ctx2048_24ep` | **45.63** | 0.486 | 0.391 | 2.27 |
| `estilo_llama_24ep` | **37.96** | 0.331 | 0.387 | 1.71 |
| `music_transformer` | **29.27** | 0.318 | 0.457 | 1.86 |
| `perceiver_ar` | **29.26** | 0.521 | 0.420 | 1.87 |
| `deep_lstm` | **21.53** | 0.300 | 0.406 | 1.17 |

Referencia del corpus: densidad **0.451**, repeat8 **0.238**, polifonia **1.97**.

Sin prefijo todos los modelos caen respecto al escenario con prefijo, que es lo
esperable: la mitad del trabajo lo hacia el contexto real. Es el escenario que
mejor separa a un modelo que **compone** de uno que solo **continua**.

Audio en `reports/audio/3_generacion_libre/<modelo>/desde_cero_sN.wav`.

## 10. Figuras

- [`reports/figures/prediccion_vs_generacion.png`](../reports/figures/prediccion_vs_generacion.png) - prediccion frente a generacion
- [`reports/comparacion_pareada/comparacion.png`](../reports/comparacion_pareada/comparacion.png) - comparacion pareada, mismo material
- [`reports/figures/leaderboard.png`](../reports/figures/leaderboard.png) - comparativa de todos los experimentos
- [`reports/figures/dataset_overview.png`](../reports/figures/dataset_overview.png) - resumen del dataset
- [`reports/figures/corpus_examples.png`](../reports/figures/corpus_examples.png) - fragmentos reales del corpus
- [`reports/best/training_curves.png`](../reports/best/training_curves.png) - curvas del modelo ganador
- [`reports/best/generation_best.png`](../reports/best/generation_best.png) - mejor continuacion generada  *(no generada)*
- [`reports/best/distributions.png`](../reports/best/distributions.png) - distribuciones generadas vs corpus
