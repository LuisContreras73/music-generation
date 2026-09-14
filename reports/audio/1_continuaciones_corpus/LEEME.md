# Galeria de audio

Generado por `scripts/11_audio_gallery.py` desde las continuaciones que cada
experimento guardo durante el entrenamiento. Timbre de caja de musica
(`audio_play.py`). Cada `.wav` va acompanado del `.mid` equivalente.

Las carpetas estan numeradas **de mejor a peor** `gen_score`.

| carpeta | modelo | gen_score | que se oye |
|---|---|---|---|
| `00_CORPUS_real` | *(dataset)* | 87.6 (techo) | musica real, la referencia |
| `lstm` | lstm | 82.8 | linea base recurrente |
| `estilo_llama_ctx2048_24ep` | modern | 79.3 |  |
| `music_transformer` | music_transformer | 70.6 | Music Transformer con atencion relativa |
| `estilo_llama_24ep` | modern | 67.2 |  |
| `perceiver_ar` | perceiver_ar | 63.6 |  |
| `deep_lstm` | deep_lstm | 63.5 |  |
| `estilo_llama_10ep` | modern | 62.6 |  |
| `estilo_llama_ctx2048_10ep` | modern | 58.9 |  |
| `music_transformer_con_augmentacion` | music_transformer | 51.3 | Music Transformer con atencion relativa |
| `music_transformer_ctx2048` | music_transformer | 51.2 | Music Transformer con atencion relativa |
| `tft` | tft | 35.3 | TFT adaptado |
| `ablacion_sin_atencion_relativa` | music_transformer | 15.7 | sin atencion relativa (ablacion) |
| `melle` | melle | 3.2 | familia frames: genera ~9x mas notas de las reales, se oye saturado |
| `ablacion_melle_sin_flux` | melle | 2.1 | familia frames: genera ~9x mas notas de las reales, se oye saturado |

## Como escuchar

Los `.wav` se abren con cualquier reproductor. Los `.mid` suenan mejor con un
sintetizador de piano real (por ejemplo VLC con un SoundFont, o importandolos
en un DAW).

### Que escuchar

1. Empieza por `00_CORPUS_real`: es el objetivo.
2. Compara con la carpeta `01_...`, que es el modelo con mejor `gen_score`.
3. Baja por la lista: las ultimas carpetas suenan saturadas o repetitivas, y
   eso es exactamente lo que miden las metricas.

Nota sobre la duracion: el dataset guarda **ataques sin duracion**, asi que la
duracion de cada nota es una reconstruccion (hasta el siguiente ataque de la
misma nota, tope 1.5 s en MIDI; timbre percusivo con decaimiento en WAV).