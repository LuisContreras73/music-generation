# 05. Estado del arte (2018-2026) y que aplica a NUESTROS datos

Documento de sintesis. Espanol sin tildes, ASCII puro, por convencion del proyecto.

## 0. Aviso previo sobre la procedencia de este documento (leer antes que nada)

El encargo decia que **cuatro agentes** habian verificado papers con fuentes primarias.
**A este agente le llego UN solo informe: el de Perceiver AR (arXiv:2202.07765).**
Los otros tres informes **no estan en el material recibido**, y el unico que llego venia
**truncado** (se corta en mitad del punto 5 de su seccion de cifras no confirmadas).

Consecuencia directa, y es la parte mas importante de esta seccion:

- Las filas de la tabla marcadas **[V1]** estan verificadas contra fuente primaria por el
  agente de Perceiver AR: son Perceiver AR, la Seccion A.2 de Music Transformer
  (arXiv:1809.04281) y la Tabla 7 de MAESTRO (arXiv:1810.12247). Nada mas.
- **TODAS las demas filas estan marcadas [V0]: proceden del conocimiento del modelo que
  escribe este documento, SIN acceso a la fuente primaria en esta sesion.** Los
  identificadores arXiv, los anos y las descripciones de metodo son razonablemente
  fiables; **las cifras concretas NO, y por eso en las filas [V0] no se escriben cifras
  de NLL ni de perplejidad.** Si alguien necesita citar una cifra de una fila [V0], tiene
  que abrir el paper. No se ha hecho aqui, y decirlo es mas util que inventarlo.
- Esto no invalida el documento: las conclusiones de las secciones 3, 4, 5 y 6 se apoyan
  en (a) lo verificado de Perceiver AR y (b) **nuestras propias mediciones**, que si son
  primarias y reproducibles en este repositorio.

---

## 1. Tabla cronologica 2018-2026

Se parte en dos mitades con el mismo orden de filas (una sola tabla de 12 columnas es
ilegible). La columna **ID** enlaza las dos mitades.

### 1A. Identidad, datos y representacion

| ID | Ano | Trabajo | Arquitectura | Dataset REAL | Tokenizer / representacion | Contexto | V |
|---|---|---|---|---|---|---|---|
| A1 | 2018 | MusicVAE | VAE recurrente jerarquico | Lakh MIDI (fragmentos de 2 y 16 compases) | rejilla metrica de 16 pasos por compas, monofonico o 3 pistas | 2-16 compases | V0 |
| A2 | 2018 | Music Transformer (ICLR 2019) | decoder-only + atencion relativa (skewing) | Piano-e-Competition / MAESTRO; JSB Chorales | MIDI-like de 388: 128 NOTE_ON + 128 NOTE_OFF + 100 TIME_SHIFT (10 ms..1 s) + 32 SET_VELOCITY | 2048 tokens (crops aleatorios) | **V1** (solo tokenizacion y baseline) |
| A3 | 2018 | MAESTRO / Wave2Midi2Wave (ICLR 2019) | transcripcion + LM simbolico + WaveNet | MAESTRO v1 (~200 h de piano clasico interpretado) | igual que A2 | 2048 | **V1** (solo Tabla 7) |
| A4 | 2019 | Sparse Transformer | atencion factorizada dispersa | texto, imagenes y audio crudo | varios | decenas de miles | V0 |
| A5 | 2019 | Nucleus sampling ("The Curious Case of Neural Text Degeneration") | **no es arquitectura: es decodificacion** | texto (WebText) | BPE | n/a | V0 |
| A6 | 2019 | Unlikelihood training | **no es arquitectura: es funcion de perdida** | texto (Wikitext-103) | BPE | n/a | V0 |
| A7 | 2019 | LakhNES | Transformer-XL + transfer desde Lakh | NES-MDB (chiptune multipista) | eventos con DELTA de tiempo | memoria XL | V0 |
| A8 | 2020 | Pop Music Transformer (REMI) | Transformer-XL | pop piano transcrito (corpus propio del grupo de Yang) | **REMI: BAR, POSITION(1/16), TEMPO, CHORD, NOTE_ON, DURATION, VELOCITY** | memoria XL | V0 |
| A9 | 2020 | MMM (Multi-Track Music Machine) | GPT-2 | Lakh MIDI multipista | eventos por pista concatenados (bar-fill) | 2048 | V0 |
| A10 | 2021 | Compound Word Transformer (AAAI 2021) | Transformer lineal con cabezas por campo | pop piano (AILabs) | **compound word**: un paso = tupla (tipo, compas/posicion, acorde, tempo, pitch, duracion, velocity) | secuencias mucho mas cortas que REMI | V0 |
| A11 | 2021 | MuseMorphose | Transformer-VAE condicionado por atributos | pop piano | REMI + atributos por compas (intensidad ritmica, polifonia) | compases | V0 |
| A12 | **2022** | **Perceiver AR (ICML 2022)** | **una sola cross-attention M->N + L capas de self-attention causal sobre N latentes** | **MAESTRO v1 y v3 (simbolico); MAESTRO SoundStream (audio); corpus privado de 10000+ h de piano transcrito** | **la misma de A2 (388 tokens, con velocity y NOTE_OFF)** | **4096 tokens con 2048 latentes (simbolico); 32768 con 1024 latentes (privado); 65536 (audio)** | **V1** |
| A13 | 2022 | Locally typical sampling (TACL) | decodificacion | texto | BPE | n/a | V0 |
| A14 | 2022 | SimCTG / contrastive search (NeurIPS 2022) | perdida contrastiva + decodificacion | texto | BPE | n/a | V0 |
| A15 | 2022 | DITTO ("Learning to Break the Loop", NeurIPS 2022) | analisis del auto-refuerzo de repeticiones + fine-tune | texto | BPE | n/a | V0 |
| A16 | 2022 | FIGARO | descripcion por compas + seq2seq controlable | Lakh MIDI | REMI+ multipista | compases | V0 |
| A17 | 2022 | Museformer (NeurIPS 2022) | atencion fina/gruesa segun la estructura de compases | LMD pop | eventos con compas explicito | miles de tokens | V0 |
| A18 | 2022 | SymphonyNet | Transformer con BPE de eventos multipista | corpus sinfonico MIDI | eventos multipista + BPE musical | largo | V0 |
| A19 | 2022/23 | Multitrack Music Transformer (ICASSP 2023) | decoder-only multicampo | SOD / Lakh multipista | tuplas (tipo, beat, posicion, pitch, duracion, instrumento) | miles | V0 |
| A20 | 2023 | Difusion discreta para musica simbolica (IJCAI 2023) | difusion discreta tipo D3PM sobre piano-roll | corales / pop en rejilla | **piano-roll en rejilla metrica, longitud fija** | ventana fija | V0 |
| A21 | 2023 | Polyffusion | difusion latente tratando el piano-roll como imagen | POP909 y similares | **piano-roll en rejilla, longitud fija** | 8-32 compases | V0 |
| A22 | 2023 | Anticipatory Music Transformer | decoder-only con eventos de llegada (anticipacion) para control e inpainting | Lakh MIDI | eventos con tiempo de llegada + eventos de control | miles | V0 |
| A23 | 2023 | Mamba (SSM selectivo) | espacio de estados selectivo, sin atencion | texto, ADN, audio; **no musica simbolica** | varios | muy largo, coste lineal | V0 |
| A24 | 2024 | Whole-Song Hierarchical Generation (ICLR 2024) | difusion jerarquica por niveles (forma -> acordes -> notas) | POP909 y similares | rejilla metrica con niveles de forma | cancion completa | V0 |
| A25 | 2024 | MuPT | LLM sobre notacion ABC | corpus de partituras ABC | **texto ABC (compases, tonalidad, figuras)** | contexto de LLM | V0 |
| A26 | 2024 | PerceiverS | variante Perceiver para musica simbolica de largo plazo | no verificado | no verificado | largo | V0 (el agente lo cita como existente y declara **no** haber comprobado sus cifras) |
| A27 | 2025 | NotaGen y familia de LLM de partitura (pretrain + finetune + RL) | LLM | corpus de partituras | notacion simbolica tipo ABC / MusicXML | contexto de LLM | V0, **identificador arXiv NO verificado: comprobar antes de citar** |
| A28 | 2025-26 | -- | -- | -- | -- | -- | **No hay ninguna fila verificada de 2025-2026 en el material recibido. Este hueco es real, no una omision de estilo.** |

### 1B. Cifras, recursos y fuentes

| ID | Cifra reportada (metrica + dataset + split) | Parametros | Hardware | Repo | arXiv / DOI | V |
|---|---|---|---|---|---|---|
| A1 | metricas de reconstruccion y test de escucha; sin NLL comparable | no verificado | no verificado | github.com/magenta/magenta (music_vae) | arXiv:1803.05428 | V0 |
| A2 | **NLL 1.84 en validacion de MAESTRO v1** (base del logaritmo NO declarada); 6 capas, crops de 2048, **CON aumentacion (transposicion +-3 semitonos y time-stretch)** | no citado en el material verificado | no citado en el material verificado | github.com/magenta/magenta; tensor2tensor | arXiv:1809.04281 (ICLR 2019) | **V1** |
| A3 | su Tabla 7 es la fuente real del 1.84 que luego cita Perceiver AR | -- | -- | github.com/magenta/magenta (maestro) | arXiv:1810.12247 (ICLR 2019) | **V1** |
| A4 | bits/dim y perplejidad en dominios no musicales | no verificado | no verificado | github.com/openai/sparse_attention | arXiv:1904.10509 | V0 |
| A5 | reduce la repeticion degenerada frente a beam/greedy | n/a | n/a | -- | arXiv:1904.09751 (ICLR 2020) | V0 |
| A6 | reduce repeticion a nivel de token y de secuencia | n/a | n/a | github.com/facebookresearch/unlikelihood_training | arXiv:1908.04319 (ICLR 2020) | V0 |
| A7 | mejora por transfer desde Lakh a NES-MDB | no verificado | no verificado | github.com/chrisdonahue/LakhNES | arXiv:1907.04868 | V0 |
| A8 | su evidencia central es **test de escucha** sobre ritmo y compas, no NLL | no verificado | no verificado | github.com/YatingMusic/remi | arXiv:2002.00212 (ACM MM 2020) | V0 |
| A9 | test de escucha y control por pista | no verificado | no verificado | repo oficial no verificado | arXiv:2008.06048 | V0 |
| A10 | su argumento principal es **el coste**: secuencias mas cortas, entrenamiento viable en **una sola GPU** | no verificado | 1 GPU (modelo y horas NO verificados) | github.com/YatingMusic/compound-word-transformer | arXiv:2101.02402 (AAAI 2021) | V0 |
| A11 | control de atributos + test de escucha | no verificado | no verificado | github.com/YatingMusic/MuseMorphose | arXiv:2105.04090 | V0 |
| A12 | **NLL MAESTRO v1: test 1.82, val 1.82. MAESTRO v3: test 1.91, val 1.90. Corpus privado de piano (10000+ h): 1.24 en test.** Audio SoundStream, contexto 65536: 2.49/2.34 (12 kbps), 2.60/2.57 (18 kbps), 2.65/2.62 (22 kbps). **Base del logaritmo NO declarada.** 1M pasos, **SIN aumentacion** | **NO SE REPORTA para musica simbolica.** Estimacion del agente, no del paper: ~164 M (12 capas, d=1024, V=388) y ~315 M (24 capas). Si reportados: ImageNet 770.1 M, PG-19 974.6 M, SoundStream 215-408 M | **"either TPUv2 or TPUv3 clusters" y nada mas. Ni numero de chips, ni horas, ni coste, para ningun experimento** | github.com/google-research/perceiver-ar (JAX + Haiku). **Verificado por la API de GitHub: NO hay loader de MIDI, NO hay config de musica, NO hay checkpoint de musica. El unico checkpoint publicado es el de la tarea sintetica de copia.** Alternativa oficial: google/flaxformer (architectures/perceiver_ar) | arXiv:2202.07765; PMLR v162, pp. 8535-8558 (ICML 2022) | **V1** |
| A13 | mejora la calidad percibida frente a nucleus en texto | n/a | n/a | -- | arXiv:2202.00666 | V0 |
| A14 | reduce la degeneracion asociada a la anisotropia del espacio de representacion | n/a | n/a | github.com/yxuansu/SimCTG | arXiv:2202.06417 | V0 |
| A15 | documenta el **auto-refuerzo**: cuanto mas se ha repetido algo, mas probable lo hace el modelo | n/a | n/a | -- | arXiv:2206.02369 | V0 |
| A16 | control por descripcion + test de escucha | no verificado | no verificado | github.com/dvruette/figaro | arXiv:2201.10936 | V0 |
| A17 | reduce el coste de la atencion aprovechando la estructura de compases | no verificado | no verificado | github.com/microsoft/muzic (museformer) | arXiv:2210.10349 | V0 |
| A18 | sinfonias largas + test de escucha | no verificado | no verificado | github.com/symphonynet/SymphonyNet | arXiv:2205.05448 | V0 |
| A19 | metricas simbolicas + escucha | no verificado | no verificado | github.com/salu133445/mmt | arXiv:2207.06983 | V0 |
| A20 | metricas de rejilla + escucha | no verificado | no verificado | repo no verificado | arXiv:2305.09489 | V0 |
| A21 | inpainting de piano-roll + escucha | no verificado | no verificado | github.com/aik2mlj/polyffusion | arXiv:2307.10304 | V0 |
| A22 | test de escucha sobre acompanamiento generado (cifra exacta NO verificada) | no verificado | no verificado | github.com/jthickstun/anticipation | arXiv:2306.08620 | V0 |
| A23 | perplejidad en texto, ADN y audio; **ningun resultado de musica simbolica** | no verificado | no verificado | github.com/state-spaces/mamba | arXiv:2312.00752 | V0 |
| A24 | escucha + metricas de forma | no verificado | no verificado | no verificado | no verificado | V0 |
| A25 | perplejidad sobre ABC y leyes de escala | no verificado | no verificado | github.com/multimodal-art-projection/MuPT | arXiv:2404.06393 | V0 |
| A26 | no verificado | no verificado | no verificado | no verificado | arXiv:2411.08307 | V0 |
| A27 | no verificado | no verificado | no verificado | no verificado | **no verificado** | V0 |

### 1C. QUE COMPARACIONES **NO** SON VALIDAS Y POR QUE (lo mas importante de la tabla)

Siete clases de invalidez. Cada una esta respaldada por algo verificado.

**(1) A2 frente a A12 (Music Transformer 1.84 frente a Perceiver AR 1.82) NO es una
comparacion limpia.** El propio paper de Perceiver AR reconoce que el baseline **usaba
aumentacion de datos (transposicion +-3 semitonos y time-stretch) y Perceiver AR no**, y
que el baseline tenia 6 capas y crops de 2048 frente a 12 capas y contexto 4096. Hay al
menos tres variables cambiadas a la vez. La diferencia es de 0.02 (1.1% relativo).

**(2) MAESTRO v1 frente a MAESTRO v3 NO son el mismo problema.** Verificado: el MISMO
modelo Perceiver AR da 1.82 en v1 y 1.90-1.91 en v3. **Cambiar la version del dataset
mueve la cifra 0.09, mas de cuatro veces la "mejora" de 0.02 sobre Music Transformer.**
Cualquier tabla que mezcle numeros de v1 y de v3 en una misma columna esta mintiendo.

**(3) Nuestro 1.83 bits/paso NO es comparable ni con el 1.82 ni con el 1.84.** Verificado
por el agente: el paper no declara la base del logaritmo para musica; su lectura, **no
confirmada**, es nats por token sobre V=388. Si es asi, 1.82 nats/token = 2.63 bits/token,
mientras que nuestro 1.83 bits/paso con 0.697 tokens/paso equivale a 2.62 bits/token sobre
V=155. **La coincidencia 1.82 / 1.83 es puro azar de unidades distintas, sobre datasets
distintos y vocabularios distintos. Prohibido usar ese 1.82 como referencia nuestra.**

**(4) Ninguna cifra de NLL es comparable entre tokenizadores.** La NLL por token depende de
V y de cuantos tokens cuesta un segundo de musica. REMI (A8), compound word (A10), el
MIDI-like de 388 (A2/A12) y el nuestro (155) producen numeros que no se pueden poner en la
misma columna. La unica normalizacion honesta seria **bits por segundo de musica**, y casi
ningun paper la reporta.

**(5) Ninguna cifra de NLL es comparable entre datasets.** MAESTRO son ~200 h de piano
clasico interpretado, con velocity y pedal. El corpus privado de A12 son 10000+ h
transcritas automaticamente. El nuestro son 714 h de ataques binarios cuantizados a 50 ms.
La prueba esta en el propio paper: 1.24 en el corpus privado frente a 1.82-1.91 en
MAESTRO. **Ese salto de ~0.6 es cambio de dataset, no mejora de modelo** -- y por eso
tampoco es, en rigor, una comparacion valida entre si: son test sets distintos.

**(6) NLL, test de escucha y metricas de dominio no son la misma escala.** A8, A11, A16,
A18, A21, A22 y A24 justifican su aportacion principalmente con escucha humana. Eso **no**
se puede ordenar junto a un NLL, y menos junto a nuestro `gen_score`, que es una metrica
local del proyecto con techo empirico 87.6 calibrado para N=16 fragmentos de 800 pasos (y
21.6 sobre los prefijos monofonicos).

**(7) Bonus interno, y es el que mas nos afecta: en nuestro propio leaderboard la
verosimilitud y el `gen_score` estan ANTICORRELADOS.** `music_transformer` tiene mejor bpt (1.83) y
peor gen_score (70.6) que `lstm` (2.04 y 82.8). En `tft` esta documentado que al
bajar val_bpt de 2.07 a 1.87 el gen_score **cayo** de 35.3 a 23.6. Por tanto **optimizar
la cifra que optimizan casi todos estos papers no optimiza lo que a nosotros nos evaluan.**

---

## 2. CIFRAS QUE NO SE PUDIERON VERIFICAR

### 2.1. Lo que falto en el propio proceso

- **Tres de los cuatro informes de agentes no llegaron.** No se sabe siquiera que papers
  cubrian. Todo lo que aparece aqui fuera de A2, A3 y A12 esta sin verificar (marca V0).
- **El informe de Perceiver AR llego truncado**, cortado dentro del punto 5 de su seccion
  de cifras no confirmadas. Puede faltar material verificado que no se ha podido usar.
- **Ningun paper mencionado resulto NO existir**, con dos salvedades: el identificador
  arXiv de NotaGen (A27) **no esta verificado** y podria ser incorrecto tal como se
  escriba; y PerceiverS (A26) existe como arXiv:2411.08307 segun el agente, pero el propio
  agente declara **no** haber comprobado ninguna de sus cifras.

### 2.2. Huecos dentro de Perceiver AR (la unica fuente verificada a fondo)

1. **No se reporta el numero de parametros del modelo de musica simbolica.** El Apendice F
   da parametros de ImageNet, Wikitext-103, PG-19, Books y MAESTRO-SoundStream, pero el
   apartado de musica (F.8) solo da hiperparametros de optimizacion (lr 1e-4, 1 cabeza de
   cross-attend, 4 cabezas en el stack, cross-attend dropout 0.7, rotacion del 25% de las
   dimensiones). Las cifras ~164 M y ~315 M son **estimacion del agente**, no del paper.
2. **No se reporta cuanta TPU ni cuanto tiempo.** La unica frase es *"Training and
   evaluation were done on either TPUv2 or TPUv3 clusters"*. Ni chips, ni horas-chip, ni
   dias, ni coste, para ningun experimento. Cualquier cifra que se lea por ahi sobre "N
   TPUs para Perceiver AR" **no sale de este paper**.
3. **No se declara la base del logaritmo de los NLL musicales.** Tampoco la declaran Music
   Transformer ni MAESTRO. La lectura "son nats" es inferencia del agente (Tensor2Tensor
   reporta en nats), **no confirmada por la fuente primaria**.
4. **El titular de "100k tokens" NO es un resultado de musica.** Corresponde a una tarea
   sintetica de copia de 131072 posiciones, con 6 capas y perdida calculada solo sobre la
   segunda mitad. El contexto musical simbolico del paper es **4096**.
5. **La ganancia musical simbolica reportada es de 0.02**, con el confundido de aumentacion
   ya descrito.
6. **Aviso de generacion documentado por el propio paper** (Apendice E.3 y Figura 11): el
   cacheo de activaciones en inferencia introduce dependencias de latentes que ya no estan
   activos y degrada el resultado si se generan demasiados pasos seguidos; su solucion es
   resetear periodicamente la memoria con un forward completo. Es un coste real de
   generacion larga.

### 2.3. Huecos en todo lo demas

Para **todas** las filas V0: parametros, hardware, horas de entrenamiento, splits exactos y
cifras numericas **no se han verificado en esta sesion**. Los repos de la columna
correspondiente son los que el modelo recuerda como oficiales; **no se han comprobado por
la API de GitHub** salvo el de Perceiver AR.

---

## 3. Las TRES dimensiones: arquitectura, representacion, escala

La literatura mezcla las tres y despues atribuye la mejora a la primera. Separandolas:

### 3.1. Arquitectura

**Evidencia a favor de que importa (nuestra, medida):** la ablacion `music_transformer` frente a
`ablacion_sin_atencion_relativa` es una comparacion limpia -- mismo dato, mismo tokenizador, mismo presupuesto,
parametros casi iguales, y la rama sin atencion relativa usa codificacion posicional
sinusoidal absoluta (verificado en `src/models/music_transformer.py`, funcion
`sinusoidal_pe`, cuyo docstring la declara como control de la ablacion; no es un modelo
"sin posiciones"). Resultado: **1.83 frente a 2.80 bits/paso, y gen_score 70.6 frente a
15.7.** Es el efecto mas grande que hemos medido cambiando una sola variable.

**Evidencia en contra de que siga importando despues de ese primer escalon:** Perceiver AR,
con ~6x mas parametros estimados, el doble de capas, el doble de contexto y 1M pasos, gana
**0.02** sobre Music Transformer en MAESTRO v1, y ese 0.02 esta contaminado por la
aumentacion. Es decir: **el salto de posicion absoluta a posicion relativa vale mucho; lo
que viene despues, dentro de la familia "decoder-only con posicion relativa o rotatoria",
vale poco.**

### 3.2. Representacion

Es la dimension que mas mueve los numeros y la que menos se controla. REMI (A8) anade
BAR/POSITION/TEMPO/CHORD y su aportacion declarada es exactamente esa: **meter la rejilla
metrica dentro del vocabulario**. Compound word (A10) acorta la secuencia agrupando campos.
La difusion sobre piano-roll (A20, A21) abandona los eventos. ABC (A25, A27) sube al nivel
de partitura. **Ninguna de estas NLL es comparable con ninguna otra** (invalidez 4). Lo
unico solido que se puede afirmar: la representacion decide que estructura es *facil* de
aprender, y por eso una representacion sin compas no puede beneficiarse de un modelo que
explota el compas.

### 3.3. Escala (datos y parametros)

**El dato mas informativo de todo el material verificado:** en Perceiver AR, la mejor cifra
musical simbolica no viene de la arquitectura, sino del dataset. Misma tokenizacion, misma
familia de modelo: **1.82-1.91 en MAESTRO (~200 h) frente a 1.24 en un corpus privado de
10000+ h.** Aun aceptando que no son test sets comparables, el orden de magnitud del efecto
de los datos aplasta al 0.02 atribuible a la arquitectura.

### 3.4. Veredicto

Por orden de efecto observado: **datos > representacion > arquitectura**, con la excepcion
del escalon inicial posicion-absoluta -> posicion-relativa, que si es un efecto
arquitectonico grande **y que nosotros ya hemos cruzado**. Y hay una cuarta dimension que
la literatura de arquitecturas casi no toca y que es la que nos esta matando: **la
decodificacion y el objetivo de entrenamiento** (A5, A6, A13, A14, A15).

---

## 4. QUE APLICA A NUESTROS DATOS (brutalmente honesto)

Recordatorio de lo que somos: **onsets binarios [T,88] a 20 Hz, sin velocity, sin duracion,
sin pedal, sin tempo, sin compas**, con histograma de onsets sobre rejilla de 16 pasos
**plano (0.0625 +- 0.001)**, densidad 0.51%, polifonia media 1.86, vocabulario 155, 0.697
tokens/paso.

### 4.1. Tecnicas que pierden su razon de ser aqui (no "son peores": no tienen objeto)

| Tecnica | Por que queda sin objeto en nuestros datos |
|---|---|
| `SET_VELOCITY` / 32 bins de velocity (A2, A12) | **No tenemos velocity.** Una parte del vocabulario de 388 modela algo que en nuestro dato no existe. |
| `NOTE_OFF` y tokens de duracion (A2, A8, A10, A12, A19) | **No tenemos duracion.** El exportador MIDI la inventa. Modelarla seria modelar nuestro propio postproceso. |
| Pedal (implicito via NOTE_OFF largo en A2/A12, con ejemplo explicito en la Figura 7 de Music Transformer) | **No tenemos pedal.** |
| `BAR` / `POSITION` / `TEMPO` / `CHORD` de REMI (A8) y todo lo derivado (A11, A16) | **No tenemos compas, y esta verificado por nosotros que no hay estructura metrica que explotar** (histograma plano a 0.0625 +- 0.001). Anadir tokens de posicion de compas sobre una rejilla sin compas es inyectar ruido y vocabulario extra. |
| Atencion fina/gruesa por compases (A17, Museformer) | Su unidad de agrupacion, el compas, **no esta definida en nuestro dato**. |
| Jerarquia forma -> acordes -> notas (A24) | Requiere segmentacion formal y funciones armonicas. No disponibles. |
| Compound word (A10) | Su ganancia viene de comprimir **6-7 campos** por paso. Nosotros solo tenemos pitch y shift: la tupla colapsa a 2 campos y la compresion posible es marginal frente a nuestro 0.697 tokens/paso, que ya es muy compacto. |
| Timing expresivo a 10 ms (A2, A12) | Nuestra rejilla es de **50 ms**. Todo el modelado de microtiming es inobservable aqui. |
| Modelos de partitura ABC / MusicXML (A25, A27) | Necesitan tonalidad, compas y figuras ritmicas. **No tenemos ninguna de las tres.** |
| Tokenizadores de audio (SoundStream en A12) | Otro dominio. |
| MELLE (valores continuos) | Ya medido y descartado: 3.23 bits/paso, gen_score 3.2. Su aportacion (evitar la cuantizacion vectorial) no aplica a un dato que ya es binario. |
| TFT | Ya medido: predictor multihorizonte con covariables futuras que no tenemos. 1.87 bits/paso pero gen_score 35.3. |

### 4.2. Lo que no aplica por otra razon: el contexto largo es un problema que NO tenemos

Esto merece parrafo propio, porque es el argumento que descarta media tabla.

Nuestra evaluacion real es: **prefijo de 100 pasos + continuacion**. Nuestros scripts
generan 600-800 pasos. Total maximo ~900 pasos, que a 0.697 tokens/paso son **~630
tokens**. Nuestro contexto actual es de **1024 tokens**, es decir **~1470 pasos = 73
segundos**. **La evaluacion entera cabe holgadamente en el contexto que ya tenemos, con
margen de sobra.**

Por tanto **Perceiver AR (A12), Transformer-XL (A7, A8), la atencion dispersa (A4), Mamba
(A23), PerceiverS (A26) y Museformer (A17) resuelven un problema que nosotros no tenemos.**
Y en el caso concreto de Perceiver AR hay tres razones adicionales, todas verificadas:

- Su configuracion musical simbolica usa **N/M = 2048/4096 = 0.5**, un factor 2 de
  reduccion. A nuestra escala eso no ahorra nada relevante.
- **Solo calcula perdida sobre N de M posiciones**: con N<M se tira (1 - N/M) de la senal
  de gradiente por forward. Su Tabla 2 documenta que por eso un modelo deja de converger
  con batch 64 y vuelve a converger al subir los latentes. **Con un presupuesto de horas,
  tirar la mitad del gradiente es exactamente lo contrario de lo que nos conviene.**
- **El codigo musical no se libero.** El repo es JAX + Haiku, sin loaders de MIDI, sin
  configs de musica y sin checkpoints de musica (verificado por la API de GitHub); el unico
  checkpoint publicado es el de la tarea sintetica de copia. Portarlo a PyTorch es trabajo
  de dias, no de horas.

### 4.3. Lo que SI aplica

| Tecnica | Por que aplica |
|---|---|
| **Atencion relativa / RoPE** (A2; RoPE tambien en A12) | Ya validado por nuestra propia ablacion: 1.83 frente a 2.80. Es el unico componente arquitectonico con efecto grande demostrado **en nuestros datos**. |
| **Decodificacion anti-repeticion** (A5, A13, A14, A15) | Nuestro fallo medido es exactamente la degeneracion que estos trabajos estudian. Ver seccion 5. |
| **Objetivos que penalizan la repeticion** (A6 unlikelihood, A14 contrastivo) | La version de entrenamiento de lo anterior, para cuando la decodificacion no baste. |
| **Aumentacion dirigida al eje que esta fuera de distribucion** | La leccion de escala (3.3) traducida a nuestro caso: no podemos conseguir 10000 h, pero si podemos cambiar la **distribucion** de las 714 h que tenemos. |
| **Inpainting / condicionamiento por prefijo** (A21, A22) | Conceptualmente correcto para "dame una continuacion", aunque con coste de implementacion alto. |

---

## 5. RECOMENDACION FINAL, priorizada para UNA RTX 4070 SUPER y pocas horas

### 5.1. Primero, el diagnostico correcto del fallo (con nuestras propias mediciones)

Cifras reales leidas de `reports/external_eval/external_eval_prefix_5s/*/summary.json`,
sobre los cinco prefijos monofonicos (temperatura 1.0, top_p 0.95, 600 pasos generados).
El corpus real esta en **repeat8 = 0.217**:

| pieza | music_transformer: dens / poly / alturas / repeat8 | music_transformer_con_augmentacion | lstm |
|---|---|---|---|
| ode_to_joy | 0.237 / 1.92 / 15 / **0.440** | 0.100 / 1.00 / **5** / **0.931** | 0.190 / 1.63 / 12 / 0.533 |
| twinkle_twinkle | 0.230 / 1.73 / 17 / 0.455 | 0.103 / 1.07 / 10 / 0.823 | 0.183 / 1.43 / 14 / 0.496 |
| mary_had_a_little_lamb | 0.125 / 1.21 / 16 / 0.696 | 0.112 / 1.02 / 14 / 0.754 | 0.110 / 1.05 / 11 / 0.793 |
| frere_jacques | 0.220 / 2.16 / 10 / 0.722 | 0.107 / 1.03 / 12 / 0.868 | 0.182 / 1.65 / 14 / 0.514 |
| greensleeves | 0.108 / 1.02 / 9 / 0.841 | 0.113 / 1.03 / 15 / 0.782 | 0.195 / 1.12 / 11 / 0.734 |

**Esto cambia el diagnostico, y por tanto la recomendacion. Dos hallazgos:**

1. **El problema de "fuera de distribucion por densidad" YA ESTA RESUELTO.** `music_transformer_con_augmentacion`
   (aumentacion de texturas: transposicion, time-stretch hasta 2x y adelgazado de voces)
   clava la densidad del prefijo (0.100-0.113 frente al 0.10 objetivo) y la polifonia
   (1.00-1.07 frente a 1.00). **La aumentacion hizo su trabajo.**
2. **Y aun asi empeoro.** El repeat8 de `music_transformer_con_augmentacion` es **0.754-0.931**, peor que el de
   `music_transformer` (0.440-0.841), y en `ode_to_joy` colapsa a **5 alturas distintas** en 600
   pasos. La aumentacion corrigio las marginales y **agravo el bucle**.

Conclusion, y es el eje de toda la recomendacion: **no tenemos un problema de dominio, ni
de verosimilitud, ni de contexto. Tenemos un problema de DEGENERACION AUTORREFORZADA, y lo
tenemos incluso con nucleus 0.95 ya activo.** Ese ultimo detalle hay que decirlo sin
adornos: **el remedio estandar (A5) ya esta puesto y no basta.**

### 5.2. Prioridad 0 -- Barrido de decodificacion anti-repeticion (coste: MINUTOS, cero entrenamiento)

**Que:** barrer los checkpoints que ya existen (`music_transformer`, `music_transformer_con_augmentacion`, `lstm`) en el eje
anti-repeticion, evaluando **sobre los cinco prefijos monofonicos**, no sobre prompts del
corpus.

**Por que puede funcionar aunque nucleus ya este puesto:** nucleus trunca la cola, pero no
prohibe repetir. Lo que ataca directamente un bucle de frames es (a) prohibir n-gramas ya
emitidos y (b) penalizar tokens recientes en una ventana. **Ambas cosas ya estan
implementadas en `src/sampling.py`** y **ninguna de las dos se ha barrido todavia**:
`repetition_penalty` (con `penalty_window` y `exclude_shift`), `no_repeat_ngram` /
`NGramBlocker`, `typical_sampling`, `min_p`, `top_a`. El script
`scripts/09_sampling_sweep.py` existe, pero **solo barre temperatura x top_p y evalua sobre
prompts del corpus**; ademas no hay ningun `sampling_sweep.json` en el arbol, o sea que
**no se ha ejecutado**.

**Como, concretamente:**
- Extender `09_sampling_sweep.py` con dos ejes nuevos: `penalty` en {1.0, 1.05, 1.15, 1.3}
  con `penalty_window` en {32, 128} y `exclude_shift=True`; y `no_repeat_ngram` aplicado a
  **identificadores de frame** (usar `metrics_prefix._frame_ids`), no a tokens sueltos: el
  bucle es de frames, no de tokens.
- Evaluar con `metrics_prefix.conditional_score` y `self_similarity` (`loop_strength`,
  `longest_repeat`, `frame_entropy_norm`), no solo con `repeat8`.
- Objetivo cuantitativo: **repeat8 -> 0.217 +- 0.05** manteniendo densidad y polifonia
  cerca de las del prefijo.

**Riesgo, dicho por adelantado:** la anti-repeticion se puede "hacer trampa" -- subiendo
temperatura o prohibiendo n-gramas se baja repeat8 a costa de generar ruido. Hay que mirar
**a la vez** repeat8 y los componentes distribucionales del score, y descartar cualquier
configuracion que baje repeat8 subiendo la entropia por encima de la del corpus.

**Fuentes:** Holtzman et al. (arXiv:1904.09751), Meister et al. (arXiv:2202.00666), Keskar
et al. (arXiv:1909.05858, de donde viene la penalizacion por repeticion). **Todas V0: no
verificadas en esta sesion.**

### 5.3. Prioridad 1 -- Fine-tune con perdida anti-repeticion sobre `music_transformer_con_augmentacion` (coste: 2-4 h)

**Que:** partir del checkpoint `music_transformer_con_augmentacion` (que ya acierta densidad y polifonia) y hacerle un
fine-tune corto con **unlikelihood** (A6) o con la perdida contrastiva de SimCTG (A14):
penalizar explicitamente la probabilidad de los frames ya emitidos en el contexto.

**Por que aqui y no antes:** porque 5.1 demuestra que el problema esta **en la distribucion
del modelo**, no solo en el muestreo. Si nucleus 0.95 no lo arregla, lo probable es que la
masa de probabilidad este efectivamente concentrada en repetirse -- el auto-refuerzo que
documenta A15.

**Por que es barato:** es fine-tune, no entrenamiento desde cero. A 7.34 it/s, unos pocos
miles de pasos son menos de una hora de GPU por configuracion.

**Riesgo:** la variante a nivel de secuencia exige generar durante el entrenamiento, lo que
la hace 3-5x mas lenta por paso. Empezar por la variante barata: unlikelihood a nivel de
token sobre el contexto previo, sin generacion.

### 5.4. Prioridad 2 -- Aumentacion monofonica de verdad, no adelgazado suave (coste: 1-3 h)

**Que:** anadir al pipeline un modo **skyline** (quedarse con la altura mas aguda de cada
frame) con probabilidad ~0.15-0.25, ademas de un adelgazado mucho mas agresivo.

**Por que:** la aumentacion actual usa `aug_thin_lo = 0.85`, es decir **conserva entre el
85% y el 100% de las notas**. Eso lleva la polifonia de 1.86 a ~1.58 como mucho: **no
genera material monofonico real**. Que `music_transformer_con_augmentacion` acierte la densidad se debe mas al
time-stretch (hasta 2x) que al adelgazado. Con skyline se generan ejemplos de **polifonia
exactamente 1.00**, que es la condicion del prefijo de evaluacion.

**Honestidad obligatoria:** esto **no** arreglara por si solo el bucle -- ya sabemos que
`music_transformer_con_augmentacion` acerto las marginales y empeoro la repeticion. Es complementario a 5.2 y 5.3, no
un sustituto. Con presupuesto de horas, va **despues**.

### 5.5. Prioridad 3 -- Terminar el entrenamiento del transformer moderno con RoPE (coste: 3-6 h)

**Que:** `estilo_llama_10ep` esta en **val_bpt 2.226 en el paso 800** (26.2 M tokens vistos). Esta
crudo. A 10.85 it/s y 2.2 GB frente a 7.34 it/s del Music Transformer, es **1.48x mas
rapido** y deja memoria libre para subir el batch.

**Por que solo prioridad 3:** porque lo que compra es **verosimilitud**, y 1C(7) muestra que
en este proyecto la verosimilitud **no** predice el gen_score (LSTM: peor bpt, mejor score;
TFT: mejora bpt, empeora score). Es la inversion correcta para el informe final y para
tener un mejor modelo en general, pero **no es la que arregla el fallo que nos evaluan**.
Lanzarlo en segundo plano mientras se hacen 5.2 y 5.3.

### 5.6. Prioridad 4 -- Mezcla de logits Music Transformer + LSTM (coste: <1 h, sin entrenar)

`lstm` tiene el mejor gen_score del laboratorio (82.8 sobre techo 87.6) y su repeat8
en prefijos monofonicos es el mas bajo de los tres en 3 de las 5 piezas. Mezclar los logits
de `music_transformer` y `lstm` (media geometrica con un peso barrido) es media hora de trabajo.
**Sin respaldo bibliografico especifico en este documento** -- es un ensemble estandar, no
una tecnica de ningun paper de la tabla -- pero es tan barato que cuesta menos probarlo que
argumentarlo.

### 5.7. QUE DESCARTAR, y por que

| Descartado | Razon (no "es viejo" ni "esta poco citado") |
|---|---|
| **Perceiver AR** | Resuelve el contexto largo; nuestra evaluacion entera cabe en ~630 tokens de 1024. Ademas: tira la mitad del gradiente por forward (verificado, su Tabla 2), su ganancia musical verificada es 0.02 y esta contaminada por la aumentacion, documenta degradacion al generar con cache, y **su codigo musical no existe publicamente** (verificado por la API de GitHub). Portar de JAX a PyTorch son dias. |
| **Mamba / SSM, atencion dispersa, Transformer-XL** | Mismo motivo: optimizan un cuello de botella que no tenemos. Ademas Mamba **no reporta ningun resultado de musica simbolica**. |
| **REMI, compound word, Museformer, jerarquias por compas** | Requieren compas. **Verificado por nosotros que no hay estructura metrica en el dato.** Anadirian vocabulario sin informacion. |
| **Difusion sobre piano-roll (Polyffusion, D3PM)** | Es lo mas cercano a nuestra representacion y merece el reconocimiento, pero: longitud fija, entrenamiento caro, y reescribe todo el pipeline de generacion y evaluacion. **Incompatible con "pocas horas".** Candidato para el proyecto siguiente, no para este. |
| **Modelos de partitura (ABC / MuPT / NotaGen)** | Necesitan tonalidad, compas y figuras. No tenemos ninguna. |
| **Escalar a 100-300 M parametros** | 12.9 GB y horas. Y la evidencia verificada dice que ~6x parametros dieron 0.02 en el unico caso musical simbolico medible. |
| **Seguir optimizando val_bpt como objetivo principal** | Anticorrelado con lo que nos evaluan (1C(7)). |
| **TFT y MELLE** | Ya medidos y perdedores en nuestros propios datos (gen_score 35.3 y 3.2). |

---

## 6. Ranking final justificado

Ordenado por **efecto esperado sobre el fallo real (degeneracion en bucles con prefijos
monofonicos) dividido por horas de GPU**, no por novedad ni por numero de citas.

| # | Accion | Coste | Ataca el fallo real | Evidencia que lo respalda | Confianza |
|---|---|---|---|---|---|
| 1 | **Barrido anti-repeticion en decodificacion** (`no_repeat_ngram` sobre frames + `repetition_penalty` + `typical` / `min_p`), evaluado sobre los 5 prefijos | minutos | **Si, directamente** | repeat8 medido 0.44-0.93 frente a 0.217 del corpus; el codigo ya existe en `src/sampling.py` y **no se ha barrido nunca**; A5, A13, A15 (V0) | **Alta** de que mejore algo; **media** de que baste sola, porque nucleus 0.95 ya fallo |
| 2 | **Fine-tune con unlikelihood o contrastivo sobre `music_transformer_con_augmentacion`** | 2-4 h | **Si, en la causa** | La distribucion del modelo esta concentrada en repetir (colapso a 5 alturas en `ode_to_joy`); A6, A14, A15 (V0) | Media-alta |
| 3 | **Aumentacion skyline monofonica real** (el `aug_thin_lo = 0.85` actual es demasiado suave) | 1-3 h | Parcialmente: arregla marginales, no el bucle | `music_transformer_con_augmentacion` ya clavo densidad 0.10 y polifonia 1.00 **y empeoro repeat8**: prueba de que este eje esta casi agotado | Media para el score global, **baja** para el bucle |
| 4 | **Mezcla de logits MT + LSTM** | <1 h | Indirectamente | `lstm` gen_score 82.8 y menor repeat8 en 3 de 5 prefijos | Baja-media, con coste casi nulo |
| 5 | **Terminar `estilo_llama_10ep` (RoPE) a presupuesto completo** | 3-6 h, en segundo plano | No | 1.48x mas rapido (10.85 frente a 7.34 it/s), 2.2 GB; pero bpt y gen_score estan anticorrelados aqui | Alta de que mejore bpt, **baja** de que mejore el score de prefijos |
| 6 | Difusion sobre piano-roll (Polyffusion, D3PM) | dias | Posiblemente si, via inpainting | Representacion casi identica a la nuestra, pero longitud fija y coste alto | No evaluable en este presupuesto |
| 7 | Perceiver AR / Mamba / Transformer-XL / atencion dispersa | dias | **No** | Resuelven contexto largo; nosotros cabemos en 630 de 1024 tokens | Descartado |
| 8 | REMI / compound word / Museformer / jerarquias por compas | dias | **No** | **Verificado que no hay metrica que explotar en nuestro dato** | Descartado |
| 9 | ABC / MuPT / NotaGen | dias | **No** | Requieren partitura, tonalidad y compas | Descartado |

### Resumen en una frase

**El estado del arte de arquitecturas 2018-2026 optimiza verosimilitud y contexto largo;
nosotros no tenemos problema de contexto, nuestra verosimilitud esta anticorrelada con la
metrica que nos evaluan, y nuestro fallo -- repeat8 de 0.44 a 0.93 frente al 0.217 del
corpus, con nucleus 0.95 ya activo y con la densidad ya corregida por la aumentacion -- es
un problema de decodificacion y de objetivo de entrenamiento. Por eso lo primero que hay
que implementar es el barrido anti-repeticion que ya esta codificado y sin ejecutar, y lo
segundo un fine-tune con unlikelihood; y por eso Perceiver AR, pese a ser el unico paper
verificado a fondo de esta lista, se descarta.**
