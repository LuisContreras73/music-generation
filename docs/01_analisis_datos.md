# Análisis del dataset (`train.npz`)

Todas las cifras salen de `scripts/00_explore.py` → `data/processed/data_stats.json`.

## 1. Qué contiene el archivo

`train.npz` no guarda una matriz por pieza, sino todas las piezas **concatenadas** con un
vector de desplazamientos:

| clave | forma | significado |
|---|---|---|
| `rolls_flat` | `(51 456 090, 88)` `uint8` | todas las piezas apiladas |
| `offsets` | `(10 605,)` | límites de cada pieza: pieza *i* = `rolls_flat[offsets[i]:offsets[i+1]]` |
| `ids` | `(10 604,)` | `seq_000000` … |
| `step_sec` | `0.05` | resolución temporal: **20 Hz** |
| `note_min` / `note_max` | `21` / `108` | A0 … C8, el piano completo de 88 teclas |
| `representation` | `'onset'` | **cada 1 es un ataque de nota, no una nota sostenida** |

## 2. Los cinco hechos que determinan el diseño

**(a) `representation = 'onset'`.** La matriz no es un piano-roll de notas mantenidas: un `1`
marca el instante en que se pulsa la tecla, y nada indica cuánto dura. Esto simplifica el
problema (no hay que modelar `note_off`) y explica la escasez extrema.

**(b) La matriz está casi vacía: densidad 0.51 %.** El 75.6 % de los pasos temporales no
tiene ningún ataque. Un modelo que opere frame a frame gastaría tres cuartas partes de su
capacidad aprendiendo a emitir silencio.

**(c) La polifonía es baja.** Entre los pasos que sí tienen ataques, la media es de **1.86
notas simultáneas**:

| notas simultáneas | 1 | 2 | 3 | 4 | 5 | 6 | ≥7 |
|---|---|---|---|---|---|---|---|
| % de pasos activos | 55.0 | 23.4 | 10.8 | 6.0 | 2.9 | 1.3 | 0.7 |

**(d) Los ataques se agrupan en el tiempo.** El hueco entre pasos activos consecutivos tiene
mediana 3 pasos (0.15 s), p90 = 8 y p99 = 21; el 99.6 % de los huecos es ≤ 32 pasos, pero el
máximo llega a 1200 (60 s de silencio). La cola es larga y hay que codificarla sin desperdiciar
vocabulario.

**(e) No hay compás.** Este es el hallazgo decisivo. Si se acumulan los ataques según su
posición módulo 16 pasos, la distribución es **plana**:

```
0.0646 0.0617 0.0629 0.0617 0.0633 0.0618 0.0624 0.0618
0.0635 0.0620 0.0626 0.0622 0.0633 0.0617 0.0626 0.0621
```

frente al 0.0625 que corresponde al azar. En música cuantizada a partitura, los tiempos
fuertes concentrarían muchos más ataques. Aquí no: la rejilla de 50 ms es **tiempo de reloj**,
no subdivisiones de compás. Se trata de interpretaciones de piano transcritas
(estilo MAESTRO / GiantMIDI-Piano), con *rubato* y tempo variable.

**Consecuencia práctica:** las tokenizaciones que dependen de la métrica —REMI y sus tokens
`Bar` / `Position`, Compound Word, etc.— **no aportan nada en este dataset**, porque no existe
la rejilla métrica a la que anclarse. Lo que corresponde es un esquema de tiempo relativo
tipo *Performance-RNN* / *Music Transformer*: `NOTE_ON` + `TIME_SHIFT`.

## 3. Contenido musical

Las clases de pitch no están distribuidas al azar, lo que confirma que es música tonal real:

| G | D | C | A | E | F | A♯ | B | D♯ | G♯ | C♯ | F♯ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 9.8 % | 9.8 % | 9.7 % | 9.6 % | 9.0 % | 8.4 % | 7.9 % | 7.7 % | 7.4 % | 7.1 % | 6.8 % | 6.6 % |

Las cinco primeras son las notas naturales del centro del círculo de quintas; F♯ y C♯ son las
menos frecuentes. Las notas individuales más tocadas son C4 (MIDI 60, 3.3 %), D4, G4 y A4:
el registro central del piano.

## 4. Tamaño y particiones

- **10 604 piezas**, 51.46 M pasos = **714.7 horas** de música, 23.29 M ataques.
- Duración por pieza: mediana 4406 pasos (**3 min 40 s**), p1 = 729, p99 = 11 541. El máximo
  (11 977 ≈ 599 s) delata un truncado en 10 minutos.
- Particiones **por pieza** (no por ventana, para que no haya fuga entre train y val):
  9544 / 530 / 530 piezas → 32.2 M / 1.83 M / 1.80 M tokens.

## 5. Tokenización elegida

`src/data/tokenizer.py`, vocabulario de **155** símbolos:

```
0        PAD
1        BOS
2        EOS
3 .. 90  NOTE_ON(p)   p en [0,88)     88 símbolos
91..154  SHIFT(d)     d en [1,64]     64 símbolos
```

Reglas:

- Dentro de un mismo paso temporal las notas se emiten **en orden ascendente de pitch**, lo
  que hace la correspondencia biyectiva (no hay ambigüedad de orden).
- Un hueco `g > 64` se codifica como `⌊g/64⌋` × `SHIFT(64)` seguido del resto. Cubre el
  hueco máximo de 1200 pasos y solo afecta al 0.4 % de los huecos.
- El silencio final se codifica explícitamente, de modo que `T` se conserva.

**Round-trip exacto verificado** en 400 casos sintéticos (incluidos rolls vacíos, silencios
iniciales y finales, huecos de 1250 pasos) y en 40 piezas reales del corpus.

Resultado: **35 870 033 tokens**, es decir **0.697 tokens por paso temporal**. La
representación de eventos es 1.43× más compacta que la frame-level y, sobre todo, no gasta
capacidad en los pasos vacíos.

## 6. Por qué esto importa para la métrica

Como la tokenización es biyectiva, `log p(tokens) = log p(roll)`. Eso permite comparar en la
**misma unidad** un modelo de eventos y un modelo de frames:

```
bits/paso = −log₂ p(piano_roll) / número de pasos
```

Referencia trivial: un modelo que predijera cada nota de forma independiente con la densidad
marginal del corpus (p = 0.005143) obtendría **4.0921 bits/paso**. Cualquier modelo del
laboratorio debe quedar claramente por debajo.
