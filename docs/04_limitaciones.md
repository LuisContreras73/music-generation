# Limitaciones y qué no se ha comprobado

Registro honesto de lo que este laboratorio **no** establece, para que nadie extraiga de él
conclusiones que no soporta.

## 1. Limitaciones de los datos

- **No hay duración de nota.** La representación es de ataques (`representation='onset'`), así
  que el corpus no dice cuánto suena cada tecla. El exportador MIDI tiene que inventarla
  (cada nota hasta el siguiente ataque de la misma nota, tope 1.5 s). Lo que se oye no es la
  interpretación original.
- **No hay dinámica.** Tampoco hay velocidad de pulsación, así que todo suena a volumen
  constante. Parte de lo que hace humana una interpretación de piano está ausente del
  problema.
- **No hay pedal.** El sustain que añade el exportador es una decisión estética, no un dato.
- **Resolución de 50 ms.** Dos ataques separados menos de 50 ms colapsan en el mismo paso. En
  piano interpretado eso ocurre en los acordes rápidos y en los trémolos.
- **Corpus truncado a ~10 minutos por pieza** (máximo observado: 11 977 pasos = 599 s), así
  que la estructura de obras largas está cortada.
- **El dataset no viene identificado.** Por el tamaño (10 604 piezas, 714 h) y sus
  características parece GiantMIDI-Piano o similar, pero no se ha verificado; no se conoce el
  reparto de compositores ni de estilos, así que no se puede decir nada sobre sesgo del
  repertorio.

## 2. Limitaciones de la métrica `gen_score`

- **Mide distribuciones marginales, no estructura musical.** Detecta bien el ruido y los
  bucles degenerados, pero un modelo podría acertar los seis histogramas y aun así generar
  música sin dirección, sin frases ni forma. El intervalo armónico es el único componente que
  mira *dentro* de un paso; nada mira a escala de frase.
- **Su techo depende del tamaño de muestra.** La calibración está hecha para `N = 16`
  fragmentos de 800 pasos. Cambiar `cfg.n_gen_samples` o `cfg.gen_steps` invalida el techo y
  obliga a reejecutar `scripts/02_ref_stats.py`.
- **El techo empírico (≈ 87.6) tiene varianza.** Se estimó con 40 repeticiones de bootstrap;
  las diferencias de un par de puntos entre modelos no son significativas. Solo las
  diferencias grandes deben interpretarse.
- **No sustituye la escucha.** Por eso el laboratorio exporta MIDI junto con fragmentos reales
  del corpus: la comparación auditiva es la validación que ninguna métrica reemplaza.

## 3. Limitaciones del protocolo experimental

- **Una sola semilla por configuración.** No hay barras de error sobre la variabilidad de
  inicialización, así que las diferencias pequeñas entre experimentos podrían ser ruido.
- **Sin búsqueda de hiperparámetros.** El learning rate, el dropout y el tamaño se fijaron por
  criterio y por lo que cabía en 12.9 GB, no por barrido. Es perfectamente posible que alguna
  arquitectura quede por debajo de su potencial: en particular el LSTM y el TFT usan el
  learning rate que les correspondería a su familia, pero sin ajustar.
- **Presupuesto de 120 M tokens (~3.7 épocas).** Es un presupuesto modesto. La comparación es
  válida *a ese presupuesto*; un ranking distinto a 10× más cómputo no está descartado, y de
  hecho las curvas de validación no están todas saturadas.
- **Los parámetros se igualaron aproximadamente** (23–26 M), no exactamente.
- **La temperatura de muestreo es 1.0 con `top_p = 0.95` para todos.** No se ha barrido; un
  modelo puede parecer peor solo porque su distribución óptima de muestreo es otra.

## 4. Sobre las adaptaciones de TFT y MELLE

Ambas son **portes**, no reimplementaciones fieles, y por tanto sus resultados **no son
evidencia sobre los modelos originales**:

- El **TFT** original predice un horizonte multi-paso con covariables conocidas a futuro y
  salida por cuantiles. Aquí se le quitó el decoder de horizonte, los cuantiles y las
  covariables futuras, y se le añadió una salida categórica sobre el vocabulario. Que gane o
  pierda en este laboratorio no dice nada sobre su desempeño en previsión de series
  temporales, que es su tarea.
- **MELLE** opera sobre mel-espectrogramas continuos y su aportación central es evitar la
  cuantización vectorial. Aquí los datos ya son discretos, así que esa ventaja no aplica; y se
  le añadió una cabeza Bernoulli que el paper no tiene, sin la cual no habría verosimilitud
  comparable. Un resultado pobre aquí es esperable por el dominio, no un juicio sobre MELLE
  en síntesis de voz.
- **La adaptación de MELLE genera ~9× más notas de las reales**, y la ablación demostró que
  *no* es por el hiperparámetro que yo suponía. La hipótesis inicial era que la *flux loss*
  —que premia la variación entre frames para evitar mel-espectrogramas planos— equivalía a
  premiar encender notas en un roll binario disperso. Medido con `scripts/08_melle_drift.py`:

  | | `flux_weight` | calibración | realimentación | total | deriva |
  |---|---|---|---|---|---|
  | `melle` | 0.1 | 2.36× | 3.57× | **8.45×** | 0.98× (estable) |
  | `ablacion_melle_sin_flux` | 0.0 | 1.99× | 4.78× | **9.53×** | 1.77× (acumulativa) |

  Quitar la *flux loss* mejora algo la calibración (2.36× → 1.99×) pero **empeora** la
  realimentación (3.57× → 4.78×), y el exceso total no baja. Es decir, la *flux loss* actuaba
  como freno parcial de la deriva a costa de descalibrar, y el factor dominante es en ambos
  casos la **realimentación al muestrear**, no el término de pérdida. Ver el punto 5.
- Las «variables» de la Variable Selection Network del TFT son features derivadas del mismo
  token, no covariables independientes como en el paper. Su interpretación es por tanto más
  débil que la original.

## 5. Limitación estructural de la familia de frames

MELLE, como cualquier modelo frame-level con salida factorizada, predice las 88 notas de un
paso de forma **condicionalmente independiente**. No puede representar la correlación de un
acorde: sabe que Do y Mi son probables, pero no que aparecen *juntos*. Al muestrear produce
combinaciones que el corpus nunca contiene. Esto no es un defecto de implementación sino de la
familia, y es la razón por la que se esperaba que los modelos de eventos —que emiten las notas
simultáneas en secuencia, cada una condicionando a la siguiente— tuvieran ventaja.

La medición lo confirma y explica el mecanismo. Con teacher forcing el modelo espera ~0.9–1.1
notas por frame frente a 0.46 reales: descalibrado, pero solo por un factor 2. El salto grande
(otro factor 3.6–4.8) aparece **al muestrear**, y el motivo es que el número de notas de un
frame es la suma de 88 Bernoullis independientes: su varianza hace que algunos frames salgan
muy poblados, un frame así queda fuera de la distribución de entrenamiento, y el modelo
condicionado a él predice aún más notas. No hay nada en la factorización que corrija esa
realimentación.

Los modelos de eventos son inmunes a este mecanismo concreto porque cada nota de un acorde se
emite condicionada a las anteriores del mismo paso: el modelo *ve* cuántas notas lleva puestas.

Arreglarlo en la familia de frames requeriría una salida autoregresiva dentro del frame
(estilo RNN-NADE), que no se ha implementado.

## 6. Lo que sí está verificado

Para contraste, esto sí se comprobó ejecutando código:

- La tokenización es **exactamente reversible** (400 casos sintéticos incluyendo rolls vacíos,
  silencios y huecos de 1250 pasos; 40 piezas reales del corpus).
- **Ningún modelo tiene fuga causal**: perturbar `x[:,t]` deja los logits de todas las
  posiciones anteriores idénticos bit a bit, comprobado posición a posición en los tres
  modelos de eventos y por gradiente exacto en el de frames.
- El *skewing* de la atención relativa coincide con una implementación explícita O(L²)
  derivada de la definición del paper (error 1.8e-15).
- MELLE es determinista en `eval()` y estocástico en `train()`, es decir su muestreo latente
  variacional funciona y el gradiente fluye por la reparametrización.
- Las particiones son **por pieza**, así que no hay ventanas de la misma obra en train y val.
- El `gen_score` calibrado separa música real (87.6) de ruido con la densidad correcta (1.6),
  de silencio (0.0) y de bucles (0.0).
