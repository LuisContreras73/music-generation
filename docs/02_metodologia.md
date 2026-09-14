# Metodología: modelos, métricas y protocolo

## 1. Nota sobre la elección de arquitecturas

El encargo sugería entrenar «un Temporal Fusion Transformer, un MELLE o alguno para
generación de música que sea bueno». Conviene ser explícito sobre lo que cada uno es, porque
ninguno de los dos primeros fue diseñado para esta tarea:

- **TFT** (*Temporal Fusion Transformer*, Lim et al. 2021) es un **predictor multi-horizonte
  de series temporales** con covariables conocidas a futuro y salida por cuantiles. No es un
  modelo generativo autoregresivo: predice un horizonte completo de una vez y no define una
  distribución sobre secuencias que se pueda muestrear token a token.
- **MELLE** (Meng et al. 2024) sí es autoregresivo, pero opera sobre **valores continuos**
  (mel-espectrogramas de voz) y su aportación es evitar la cuantización vectorial. Aquí los
  datos ya son binarios y discretos, así que su ventaja principal no aplica.
- Lo que el estado del arte usa para esta tarea exacta —piano-roll simbólico, generación de
  continuaciones— es un **decoder-only Transformer con atención relativa**, es decir el
  **Music Transformer** (Huang et al. 2018).

En lugar de descartar la sugerencia, el laboratorio la convierte en el experimento: se
entrenan **las cuatro** arquitecturas sobre exactamente los mismos datos, la misma
tokenización y el mismo presupuesto, y se comparan con una métrica común. TFT y MELLE se
implementan como **adaptaciones documentadas** —se conservan sus componentes característicos
y se sustituye lo que no aplica—, y cada desviación respecto al paper original queda anotada
en el docstring de su módulo. Así la recomendación queda contrastada con evidencia en vez de
aceptada o rechazada de palabra.

## 2. Los catorce experimentos

| experimento | modelo | familia | qué aporta |
|---|---|---|---|
| `music_transformer` | Music Transformer | eventos | atención relativa (*skewing* de Huang et al.); el candidato fuerte |
| `ablacion_sin_atencion_relativa` | idéntico, `rel_attn=False` | eventos | **ablación**: aísla el efecto de la atención relativa |
| `music_transformer_ctx2048` | contexto 2048 tokens | eventos | estructura de más largo alcance (~154 s de música) |
| `lstm` | LSTM 3×1024 | eventos | línea base recurrente; muestreo O(1) por token |
| `tft` | TFT adaptado | eventos | VSN + GRN + LSTM local + atención interpretable |
| `melle` | MELLE adaptado | frames | muestreo latente variacional + regresión + *flux loss* |
| `ablacion_melle_sin_flux` | `flux_weight = 0` | frames | **ablación**: la *flux loss* infla la densidad en datos binarios |
| `music_transformer_con_augmentacion` | idéntico + transposición y adelgazado | eventos | **ablación de datos**: aísla el efecto de la augmentación |
| `deep_lstm` | LSTM 4×1280 con augmentación | eventos | ¿basta más profundidad recurrente? |
| `estilo_llama_10ep` | receta LLaMA (RoPE+RMSNorm+SwiGLU) | eventos | arquitectura de 2024-25 frente a la de 2018 |
| `estilo_llama_24ep` | la misma, presupuesto ×2.4 | eventos | cuánto queda por ganar solo entrenando más |
| `estilo_llama_ctx2048_10ep` | la misma con contexto 2048 | eventos | contexto largo a igualdad de tokens |
| `estilo_llama_ctx2048_24ep` | la misma, presupuesto ×2.4 | eventos | **mejor bits/paso del laboratorio** |
| `perceiver_ar` | Perceiver AR (ICML 2022) | eventos | cuello latente y *cross-attend* inicial |

Los pasos se derivan siempre de un **presupuesto de tokens**, no del número de iteraciones:
igualar pasos sería injusto cuando difieren el batch o la longitud de ventana. Hay tres
presupuestos, y solo son comparables entre sí los experimentos que comparten uno:

| presupuesto | tokens | épocas equivalentes | experimentos |
|---|---|---|---|
| comparativa de arquitecturas | 120 M | ≈3.7 | los siete primeros de la tabla, más `music_transformer_con_augmentacion` y `deep_lstm` |
| entrenamiento largo | 321 M | ≈10 | `estilo_llama_10ep`, `estilo_llama_ctx2048_10ep`, `perceiver_ar` |
| entrenamiento máximo | 780 M | ≈24 | `estilo_llama_24ep`, `estilo_llama_ctx2048_24ep` |

«Épocas equivalentes» es `tokens_procesados / tamaño_del_corpus`, **no** épocas reales: las
ventanas se muestrean al azar con reemplazo, así que a 1.0 épocas equivalentes solo se ha
visto el 63.7 % del corpus (medido). Los parámetros se igualan en todos los casos (20–27 M)
para que la comparación sea de arquitectura, no de capacidad.

**Familia «eventos»**: la secuencia son tokens `NOTE_ON`/`SHIFT`; la salida es una categórica
sobre 155 símbolos. Modela la polifonía **exactamente**, porque las notas simultáneas se
emiten en orden y cada una condiciona a la siguiente.

**Familia «frames»**: la secuencia son los 88 valores binarios de cada paso de 50 ms. Su
limitación intrínseca es que las 88 notas de un mismo frame se predicen de forma
condicionalmente independiente, así que **no puede modelar la correlación de un acorde**.
Se mide y se comenta, en lugar de ocultarse.

### Por qué el nombre es `estilo_llama` y no `llama`

El backbone de estos cuatro experimentos es la **receta de LLaMA** (Touvron et al., 2023,
arXiv:2302.13971), que es hoy la configuración por defecto compartida por PaLM, LLaMA, Gemma,
Qwen y OLMo. Se implementa en [`src/models/modern_transformer.py`](../src/models/modern_transformer.py)
y consiste en cambiar cuatro piezas del decoder de 2018, dejando intactas profundidad, anchura
y número de parámetros:

| pieza | Music Transformer (2018) | receta LLaMA | referencia |
|---|---|---|---|
| posición | sesgo relativo con *skewing* | **RoPE** (rotación en el plano Q·K) | Su et al., arXiv:2104.09864 |
| normalización | LayerNorm | **RMSNorm**, pre-norma | Zhang & Sennrich, arXiv:1910.07467 |
| feed-forward | MLP con GELU | **SwiGLU** | Shazeer, arXiv:2002.05202 |
| estabilidad | — | **QK-norm** (opcional) | Dehghani et al., arXiv:2302.05442 |

El prefijo «estilo» está para evitar la lectura que el nombre pelado invitaría a hacer:
aquí **no** hay ningún LLaMA de por medio. No se parte de pesos preentrenados ni se
afina un modelo de lenguaje: la red se entrena **desde cero**, tiene 25.8 M de parámetros
(no 7 B), su vocabulario son los 155 símbolos musicales de este laboratorio (no 32 K piezas
de texto) y su contexto es de 1024 o 2048 tokens. Lo único que se toma de LLaMA es esa
combinación de cuatro componentes, que es exactamente lo que el nombre debe transmitir.

Qué aporta, y qué **no** se puede afirmar todavía. `estilo_llama_10ep` logra **1.668
bits/paso** frente a los **1.828** de `music_transformer`, pero ese par **no está
controlado**: el primero vio 321 M tokens y usó augmentación, el segundo 120 M y ninguna.
La diferencia mezcla arquitectura, presupuesto y datos, así que atribuirla a la receta sería
un salto injustificado. El experimento que aislaría la variable (`estilo_llama_presupuesto_mt`
en [`scripts/run_experiments.py`](../scripts/run_experiments.py): la misma red con los pasos,
el batch y la ausencia de augmentación exactos de `music_transformer`) está definido pero
**no se llegó a ejecutar**, y es la primera pieza que faltaría para cerrar la comparación.

Lo que sí es atribuible es el **coste**: 124 276 frente a 55 378 tokens por segundo de mediana
(2.24×, leído de `logs/metrics.csv`), porque RoPE no materializa la matriz relativa que el
*skewing* de 2018 sí construye y deja memoria libre para duplicar el batch. En esa cifra van
juntos los dos efectos, que es justo por lo que permitió pagar 10 y 24 épocas equivalentes.

### Adaptaciones respecto a los papers originales

**TFT → decoder causal.** Se conservan: *Gated Residual Network* (Dense→ELU→Dense→GLU→Add&Norm),
*Variable Selection Network* con pesos softmax por variable, encoder LSTM local con puerta
residual, codificador de covariables estáticas y *Interpretable Multi-Head Attention* (cabezas
que comparten la matriz de valores). Se elimina: el decoder de horizonte futuro, la salida por
cuantiles y las covariables conocidas a futuro. Las «variables» de entrada son features
derivadas del token actual (identidad, tipo, pitch, clase de pitch, duración del *shift*), y
el contexto estático se obtiene por **media acumulada estrictamente causal**. Se gana
interpretabilidad: `interpret()` devuelve los pesos de selección de variables y de atención,
que se grafican.

**MELLE → piano-roll.** Se conservan: el módulo de muestreo latente variacional
(reparametrización + término KL), la pérdida de regresión, la *flux loss* que premia la
variación frame a frame, y la cabeza de parada. Se sustituye el mel continuo por un *pre-net*
que proyecta el frame binario de 88 dimensiones. Se **añade** una cabeza Bernoulli
obligatoria, sin la cual el modelo no tendría una verosimilitud discreta comparable con los
demás.

> **Cuánta música ve el modelo.** Medido sobre 60 ventanas reales de validación:
> 1024 tokens cubren **78 s** de música de media (mediana 68 s, rango 31–171) y 2048
> tokens cubren **154 s** (mediana 141). El corpus tiene 0.697 tokens por paso de
> 50 ms, así que un token equivale a ~0.072 s. La varianza es alta porque un pasaje
> denso gasta muchos tokens en pocos segundos y uno disperso al revés.

## 3. La métrica común: bits por paso temporal

Comparar un modelo de eventos con uno de frames es el problema metodológico central: sus
pérdidas de entrenamiento no son comparables (una es entropía cruzada sobre 155 clases, la
otra una BCE sobre 88 dimensiones). La solución es medir ambos sobre **el mismo objeto**, el
piano-roll:

```
bits/paso = −log₂ p(piano_roll) / número de pasos de 50 ms
```

- **Eventos**: la tokenización es biyectiva ⇒ `log p(tokens) = log p(roll)`. Se acumula la NLL
  de todos los tokens y se divide entre los pasos que representan (la suma de las duraciones
  de los tokens `SHIFT`).
- **Frames**: NLL Bernoulli de las 88 notas, sumada sobre las 88 dimensiones y dividida entre
  el número de frames.

Dos modelos que asignen la misma probabilidad al mismo roll obtienen el mismo valor, sea cual
sea su parametrización.

**Referencia trivial:** un modelo i.i.d. con la densidad marginal del corpus (p = 0.005143) da
`88 · H₂(p)` = **4.0921 bits/paso**. Es el umbral mínimo que cualquier modelo debe batir.

> Consecuencia de diseño: en la familia de frames se optimiza la **BCE pura, sin `pos_weight`**.
> Reponderar las clases mejoraría el F1 con umbral 0.5 pero descalibraría el modelo y empeoraría
> sus bits/paso, que es la magnitud que de verdad se compara. La densidad correcta se recupera
> igualmente al generar, porque se muestrea de la Bernoulli en lugar de umbralizar.

Esa decisión tiene un efecto lateral en el F1 que hay que corregir al medir: como el modelo
queda **calibrado a la densidad real (0.0051)**, casi ninguna nota supera una probabilidad de
0.5, y el F1 a ese umbral salió `0.000` con precisión 0.144 y recall 0.0008. Ese número mide la
calibración, no la capacidad de discriminar. Por eso `eval_frame_model` **barre el umbral** y
reporta tanto `val_frame_f1` (umbral 0.5, comparable con la literatura que lo usa) como
`val_frame_f1_best` con su umbral óptimo, que es el informativo aquí.

## 4. Calidad de la generación: `gen_score` calibrado

Los bits/paso miden ajuste estadístico, no si la música generada *suena* plausible. Un modelo
puede tener buena verosimilitud y aun así generar bucles degenerados. Por eso se puntúa además
la **distribución** de lo generado frente a la del corpus, mediante el área de solapamiento
(`OA = Σ min(p, q)`, 1 = distribuciones idénticas) de seis histogramas.

### El primer diseño no funcionaba

La versión inicial promediaba los OA con pesos elegidos a mano. Los controles la tumbaron:

| entrada | `gen_score` sin calibrar |
|---|---|
| fragmentos **reales** del corpus | 79.7 |
| modelo con 150 pasos de entrenamiento (ruido) | 69.6 |
| **ruido i.i.d.** con la densidad correcta | 63.7 |

Un rango de 16 puntos entre música real y ruido puro es inservible. Dos causas:

1. **El techo no es 1.** El OA entre dos muestras *finitas* de la misma distribución baja por
   error de muestreo: con 16 fragmentos de 800 pasos el máximo alcanzable es ≈ 0.88.
2. **Algunos componentes no discriminan.** El histograma de IOI del ruido i.i.d. solapa 0.80
   con el corpus, porque la densidad marginal ya determina la distribución geométrica de
   huecos. Y el de clase de pitch solapa **0.94**: con 12 celdas, muestrear al azar sobre las
   88 teclas reproduce casi exactamente las proporciones reales.

### Diseño corregido: dos anclas empíricas

`scripts/02_ref_stats.py` mide por bootstrap (40 repeticiones, con el mismo N y la misma
longitud que se usarán al evaluar) dos anclas por componente:

- `suelo_k` = OA de ruido i.i.d. con la densidad del corpus → «no ha aprendido nada»;
- `techo_k` = OA de una muestra real del split de *train* → «indistinguible del corpus».

Cada componente se reescala a `z_k = (OA_k − suelo_k) / (techo_k − suelo_k)`, y **los pesos se
derivan del poder discriminativo medido** en vez de fijarse a mano: `w_k ∝ techo_k − suelo_k`.
Los componentes con margen inferior a 0.02 se descartan.

| histograma | suelo | techo | peso derivado |
|---|---|---|---|
| **intervalo armónico** (notas simultáneas) | 0.431 | 0.914 | **33.6 %** |
| intervalo melódico | 0.505 | 0.904 | 27.7 % |
| pitch (88) | 0.582 | 0.870 | 19.9 % |
| polifonía | 0.744 | 0.936 | 13.3 % |
| IOI (ritmo) | 0.798 | 0.877 | 5.5 % |
| clase de pitch | 0.943 | 0.918 | **descartado** |

```
gen_score = 100 · Σ_k w_k · z_k · penal_densidad · penal_bucle
```

- `penal_densidad = exp(−|log(densidad_gen / densidad_real)|)`: castiga generar demasiadas o
  demasiadas pocas notas, aunque las proporciones relativas sean correctas.
- `penal_bucle = clip(1 − 2·max(0, repetición_gen − repetición_real))`: castiga la
  degeneración en bucles, medida como fracción de 8-gramas de frames repetidos.

El componente que resultó decisivo es el **intervalo armónico** —las distancias entre notas
que suenan en el mismo paso—, y es el único que mira *dentro* de un paso temporal. El corpus
concentra octavas (17.6 %), terceras menores (14.0 %), terceras mayores (12.6 %), cuartas
(10.2 %) y quintas (6.2 %): consonancias tonales. Un modelo que muestrea notas
independientemente produce intervalos casi uniformes y queda delatado.

### Validación de la escala

| entrada | `gen_score` calibrado |
|---|---|
| fragmentos reales del split de **test** | **87.6** |
| modelo con 150 pasos de entrenamiento | 52.8 |
| ruido i.i.d. con la densidad correcta | **1.6** |
| silencio total | 0.0 |
| bucle de 8 frames repetido | 0.0 |

La referencia se calcula sobre **fragmentos reales del mismo largo** que las generaciones
(800 pasos = 40 s) tomados de validación, no sobre piezas completas: comparar 40 s generados
contra piezas de 4 minutos sesgaría los histogramas.

> La calibración depende de `N = 16` fragmentos y `800` pasos. Si se cambia
> `cfg.n_gen_samples` o `cfg.gen_steps` hay que volver a ejecutar
> `scripts/02_ref_stats.py`, porque el techo de muestreo se mueve con N.

### Métrica auxiliar: probabilidad de nota imposible

En la representación real es **imposible** que una nota aparezca dos veces en el mismo paso
temporal (la tokenización emite un conjunto ordenado por paso). Medir esto da una lectura de
cuánto ha aprendido el modelo de la estructura de la representación, independiente de la
verosimilitud.

El primer intento —contar las repeticiones realmente muestreadas— resultó inútil: como el
muestreo prohíbe esos tokens, el contador siempre daba 0. Lo que se mide es la **masa de
probabilidad** que el modelo asigna a las notas ya emitidas, *antes* de enmascararlas
(`gen_grammar_prob_mass`). Ambas medidas concuerdan cuando se desactiva el enmascarado, lo que
las valida cruzadamente: con el modelo de 150 pasos, 1.35 % de masa de probabilidad frente a
1.37 % de violaciones reales.

## 5. Protocolo

- **Particiones por pieza** (9544 / 530 / 530), nunca por ventana: dos ventanas de la misma
  pieza en train y val serían fuga de información.
- **Prefijos de generación** tomados de la partición de **validación**, con longitud fija en
  tokens y sin *padding*: meter `PAD` a la izquierda contaminaría el contexto causal, porque
  durante el entrenamiento el `PAD` solo aparece a la derecha.
- **Generación por duración musical, no por número de tokens**: se muestrea hasta acumular 800
  pasos (40 s) en todas las muestras, de modo que las métricas sean comparables entre modelos
  que gastan tokens a ritmos distintos.
- **Selección de checkpoint**: `best.pt` = mínimo `val_bpt`; `best_gen.pt` = máximo
  `gen_score`. La evaluación final sobre test se hace recargando `best.pt`.
- **Causalidad verificada por gradiente** en todos los modelos (`tests/test_causality.py`) más
  una auditoría adversarial independiente por módulo. Un modelo con fuga temporal daría
  métricas excelentes y generaciones inservibles, así que es la comprobación previa
  obligatoria antes de gastar GPU.
