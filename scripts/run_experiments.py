"""Orquestador de la matriz de experimentos del laboratorio.

    python scripts/run_experiments.py --list           # ver la matriz
    python scripts/run_experiments.py --smoke          # 150 pasos por modelo, valida el pipeline
    python scripts/run_experiments.py --only music_transformer   # un experimento
    python scripts/run_experiments.py                  # la matriz completa, en secuencia

Los entrenamientos van EN SECUENCIA porque comparten una sola GPU.
Tras cada experimento se refresca reports/leaderboard.csv y reports/best/.

Presupuesto igualado por TOKENS VISTOS
--------------------------------------
Comparar modelos por numero de pasos seria injusto si difieren el batch o la
longitud de ventana. Todos los experimentos procesan ~TOKEN_BUDGET tokens, y los
pasos se derivan de ahi.

Sobre el termino "epocas": en este laboratorio significa EPOCAS EQUIVALENTES,
es decir tokens procesados / tamano del corpus (32.2 M). NO son epocas en
sentido estricto, porque el muestreo es aleatorio CON REEMPLAZO y no una pasada
ordenada. Medido: a 1.02 "epocas" solo se habia visto el 63.7% del corpus y 1149
piezas de 9544 seguian sin tocarse; hacen falta ~5 para cubrir el 99.3%. Los parametros tambien se igualan en el rango 23-26 M, salvo el
LSTM (23.7 M), para que la comparacion sea de arquitectura y no de capacidad.
"""
from __future__ import annotations
import argparse, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import Config                     # noqa: E402

TOKEN_BUDGET = 120_000_000          # ~3.7 epocas: presupuesto de la comparativa
LONG_BUDGET = 320_000_000           # ~10 epocas: entrenamiento en serio del ganador
MAX_BUDGET = 780_000_000            # ~24 epocas: el run definitivo


def steps_for(batch_size: int, seq_len: int, budget: int = TOKEN_BUDGET) -> int:
    """Pasos necesarios para ver `budget` tokens con ese batch y ventana."""
    return int(round(budget / (batch_size * seq_len) / 100.0) * 100)


def matrix() -> list:
    """Matriz de experimentos. El orden importa: primero lo mas barato."""
    E = []

    # --- 1. MELLE adaptado a piano-roll continuo (familia frame) ---------------
    E.append(Config(name="melle", model="melle", family="frame",
                    d_model=512, n_layers=6, n_heads=8, d_ff=2048, latent_dim=64,
                    flux_weight=0.1, kl_weight=1e-3, dropout=0.1,
                    seq_len=1024, batch_size=16, lr=3e-4, warmup=500,
                    steps=steps_for(16, 1024), eval_every=700, gen_every=2100,
                    notes="MELLE (Meng et al. 2024) portado a frames binarios: muestreo "
                          "latente variacional + regresion + flux loss + cabeza Bernoulli."))

    # --- 1b. ABLACION de MELLE: sin flux loss ----------------------------------
    # Hipotesis inicial: la flux loss, que premia la variacion entre frames para
    # evitar mel-espectrogramas planos, en un roll binario disperso equivaldria a
    # premiar encender notas, explicando la densidad 4.35 vs 0.45 real.
    # RESULTADO: la ablacion REFUTA esa hipotesis. Con flux_weight=0 la
    # calibracion mejora (2.36x -> 1.99x) pero la realimentacion empeora
    # (3.57x -> 4.78x) y el exceso total no baja (8.45x -> 9.53x). El factor
    # dominante es la realimentacion del muestreo Bernoulli independiente.
    # Se conserva el experimento porque es justamente lo que permitio saberlo.
    E.append(Config(name="ablacion_melle_sin_flux", model="melle", family="frame",
                    d_model=512, n_layers=6, n_heads=8, d_ff=2048, latent_dim=64,
                    flux_weight=0.0, kl_weight=1e-3, dropout=0.1,
                    seq_len=1024, batch_size=16, lr=3e-4, warmup=500,
                    steps=steps_for(16, 1024), eval_every=700, gen_every=2100,
                    notes="ABLACION de melle con flux_weight=0. Refuta la hipotesis de "
                          "que la flux loss causa el exceso de densidad: mejora la calibracion "
                          "pero empeora la realimentacion y el total no baja."))

    # --- 2. Temporal Fusion Transformer adaptado a decoder causal --------------
    E.append(Config(name="tft", model="tft", family="token",
                    d_model=512, n_layers=4, n_heads=8, d_ff=1024, hidden=512,
                    dropout=0.1, seq_len=1024, batch_size=16, lr=5e-4, warmup=500,
                    steps=steps_for(16, 1024), eval_every=700, gen_every=2100,
                    notes="TFT (Lim et al. 2021) portado a decoder causal: VSN + GRN + "
                          "LSTM local + atencion interpretable. 23.2 M par."))

    # --- 3. linea base recurrente ---------------------------------------------
    E.append(Config(name="lstm", model="lstm", family="token",
                    d_model=512, hidden=1024, n_layers=3, dropout=0.1,
                    seq_len=1024, batch_size=24, lr=1e-3, warmup=400,
                    steps=steps_for(24, 1024), eval_every=500, gen_every=1500,
                    notes="Linea base recurrente: LSTM 3x1024 sobre tokens de evento. "
                          "Muestreo O(1) por token gracias al estado recurrente."))

    # --- 4. Music Transformer: el modelo principal -----------------------------
    E.append(Config(name="music_transformer", model="music_transformer", family="token",
                    d_model=512, n_layers=8, n_heads=8, d_ff=2048, dropout=0.1,
                    rel_attn=True, max_rel_dist=512, tie_weights=True,
                    seq_len=1024, batch_size=16, lr=3e-4, warmup=600,
                    steps=steps_for(16, 1024), eval_every=700, gen_every=2100,
                    notes="Music Transformer (Huang et al. 2018) con atencion relativa. "
                          "Modelo principal del laboratorio. 25.6 M par."))

    # --- 5. ablacion: misma red SIN atencion relativa --------------------------
    E.append(Config(name="ablacion_sin_atencion_relativa", model="music_transformer", family="token",
                    d_model=512, n_layers=8, n_heads=8, d_ff=2048, dropout=0.1,
                    rel_attn=False, tie_weights=True,
                    seq_len=1024, batch_size=16, lr=3e-4, warmup=600,
                    steps=steps_for(16, 1024), eval_every=700, gen_every=2100,
                    notes="ABLACION de music_transformer: atencion causal absoluta en vez de relativa. "
                          "Misma capacidad y mismo presupuesto: aisla el efecto de la "
                          "atencion relativa."))

    # --- 7. music_transformer CON AUGMENTACION DE TEXTURAS -------------------------------
    # El experimento decisivo. Medido: con prefijos del CORPUS de 70 tokens
    # music_transformer puntua 58.4 y no degenera (repeat8 0.223); con melodias
    # MONOFONICAS de la misma longitud cae a 2.4 con repeat8 0.631. No es la
    # longitud del contexto, es el DOMINIO: el 0.0% de las ventanas de
    # entrenamiento son monofonicas, asi que el modelo nunca vio esa textura.
    # Con esta receta el 47% lo son, conservando un 24% de piano denso.
    E.append(Config(name="music_transformer_con_augmentacion", model="music_transformer", family="token",
                    d_model=512, n_layers=8, n_heads=8, d_ff=2048, dropout=0.1,
                    rel_attn=True, max_rel_dist=512, tie_weights=True,
                    seq_len=1024, batch_size=16, lr=3e-4, warmup=600,
                    steps=steps_for(16, 1024), eval_every=700, gen_every=2100,
                    augment=True, aug_p_transpose=0.9, aug_p_stretch=0.5,
                    aug_p_thin=0.5, aug_thin_lo=0.85, aug_thin_hi=1.0,
                    aug_p_jitter=0.1,
                    notes="music_transformer + augmentacion de texturas (adelgazado de voces, "
                          "transposicion, time-stretch hasta 2x). Ataca el mismatch de "
                          "dominio con melodias monofonicas."))

    # --- 8. ARQUITECTURAS NUEVAS, priorizadas por evidencia --------------------
    # Medido sobre las melodias monofonicas con metrica condicional al prefijo:
    #   continuacion real del corpus 21.56 (techo) | lstm 13.32 |
    #   music_transformer 8.98 | music_transformer_con_augmentacion 0.00
    # La familia RECURRENTE gana en este regimen de contexto corto y senal
    # escasa, asi que deep_lstm va primero. prefix_enc es el unico disenado
    # explicitamente para el escenario prefijo-fijo + continuacion.
    # num_workers alto: la augmentacion cuesta 5x en CPU (music_transformer_con_augmentacion bajo a 0.65 it/s
    # frente a 3.20 de music_transformer) y el cuello es el DataLoader, no la GPU.
    E.append(Config(name="deep_lstm", model="deep_lstm", family="token",
                    d_model=512, hidden=1024, n_layers=8, dropout=0.1,
                    seq_len=1024, batch_size=24, lr=1e-3, warmup=400,
                    steps=steps_for(24, 1024), eval_every=500, gen_every=1500,
                    num_workers=12, augment=True, aug_p_thin=0.5, aug_thin_lo=0.85,
                    notes="Deep LSTM residual con locked dropout + augmentacion de "
                          "texturas. El recurrente lidera en prefijos cortos."))

    E.append(Config(name="prefix_enc_aug", model="prefix_enc", family="token",
                    d_model=512, n_layers=8, n_heads=8, d_ff=2048, hidden=512,
                    dropout=0.1, seq_len=1024, batch_size=16, lr=3e-4, warmup=600,
                    steps=steps_for(16, 1024), eval_every=700, gen_every=2100,
                    num_workers=12, augment=True, aug_p_thin=0.5, aug_thin_lo=0.85,
                    notes="Encoder bidireccional del prefijo + decoder causal. Unico "
                          "modelo disenado para el escenario de la prueba."))

    E.append(Config(name="modern_aug", model="modern", family="token",
                    d_model=512, n_layers=8, n_heads=8, d_ff=2048, dropout=0.1,
                    tie_weights=True, seq_len=1024, batch_size=16, lr=3e-4, warmup=600,
                    steps=steps_for(16, 1024), eval_every=700, gen_every=2100,
                    num_workers=12, augment=True, aug_p_thin=0.5, aug_thin_lo=0.85,
                    notes="Transformer moderno: RoPE + RMSNorm + SwiGLU + QK-norm."))

    E.append(Config(name="hier_aug", model="hierarchical", family="token",
                    d_model=512, n_layers=8, n_heads=8, d_ff=2048, dropout=0.1,
                    seq_len=1024, batch_size=16, lr=3e-4, warmup=600,
                    steps=steps_for(16, 1024), eval_every=700, gen_every=2100,
                    num_workers=12, augment=True, aug_p_thin=0.5, aug_thin_lo=0.85,
                    notes="Jerarquico coarse-to-fine (AudioLM): contorno melodico "
                          "primero, armonia despues."))

    # --- 9. ENTRENAMIENTO LARGO --------------------------------------------
    # Los 3.7 epocas de la comparativa infraentrenan: el gap train-val es ~0.03
    # (NEGATIVO en lstm, -0.013), o sea cero memorizacion, y val_bpt seguia
    # bajando al cortar. Ademas el coseno decae al 5% del LR maximo al final del
    # horizonte, asi que la curva se aplana por el PLANIFICADOR, no por el modelo:
    # estirar el horizonte estira tambien el coseno.
    # RoPE permite pagarlo: 10 epocas cuestan lo que 6 con el skewing de 2018.
    E.append(Config(name="estilo_llama_10ep", model="modern", family="token",
                    d_model=512, n_layers=8, n_heads=8, d_ff=2048, dropout=0.1,
                    tie_weights=True, seq_len=1024, batch_size=32,
                    lr=4e-4, warmup=600, min_lr_frac=0.02,
                    steps=steps_for(32, 1024, LONG_BUDGET),
                    eval_every=800, gen_every=2400,
                    gpu_data=True, augment=True,
                    aug_p_transpose=0.9, aug_p_thin=0.5, aug_thin_lo=0.85,
                    notes="Transformer moderno (RoPE+RMSNorm+SwiGLU), ~10 epocas, batch 32, "
                          "datos y augmentacion ENTERAMENTE EN GPU. El cuello de CPU "
                          "(1400 ms/batch de augmentacion) queda en 1.43 ms."))

    # --- 10. COMPARACION LIMPIA de arquitectura ------------------------------
    # estilo_llama_10ep batio a music_transformer (1.67 vs 1.83 bits/paso) pero con 10 epocas,
    # augmentacion y batch 32 frente a 3.7 epocas, sin augmentacion y batch 16.
    # Esa diferencia NO se puede atribuir a la arquitectura. Este experimento
    # aisla la variable: ModernTransformer con EXACTAMENTE el presupuesto de
    # music_transformer (mismos pasos, mismo batch, misma lr, sin augmentacion).
    E.append(Config(name="estilo_llama_presupuesto_mt", model="modern", family="token",
                    d_model=512, n_layers=8, n_heads=8, d_ff=2048, dropout=0.1,
                    tie_weights=True, seq_len=1024, batch_size=16,
                    lr=3e-4, warmup=600, min_lr_frac=0.05,
                    steps=steps_for(16, 1024), eval_every=700, gen_every=2100,
                    gpu_data=True, augment=False,
                    notes="ModernTransformer con el presupuesto EXACTO de music_transformer "
                          "(3.7 epocas, batch 16, sin augmentacion). Aisla la "
                          "contribucion de la arquitectura RoPE/RMSNorm/SwiGLU."))

    # --- 11. EL RUN DEFINITIVO ------------------------------------------------
    # Config elegida MIDIENDO que cabe y rinde en 12.9 GB (bf16, L=1024):
    #   d512 L8  B32 -> 25.8 M par, 4.11 it/s,  7.2 GB, 134538 tok/s  <- optimo
    #   d640 L10 B32 -> 49.7 M par, 2.34 it/s, 11.2 GB,  76777 tok/s
    #   d768 L12 B24 -> 85.1 M par, 0.43 it/s, 12.6 GB,  10632 tok/s  (pagina)
    # Un modelo mayor NO conviene: el corpus son 32.2 M tokens unicos, asi que
    # con 70 M parametros se verian 2.3x menos epocas. El cuello es el DATO, no
    # la capacidad. Se maximiza tokens/s y se gastan las horas en epocas.
    E.append(Config(name="estilo_llama_24ep", model="modern", family="token",
                    d_model=512, n_layers=8, n_heads=8, d_ff=2048, dropout=0.1,
                    tie_weights=True, seq_len=1024, batch_size=32,
                    lr=4e-4, warmup=800, min_lr_frac=0.02,
                    steps=steps_for(32, 1024, MAX_BUDGET),
                    eval_every=1500, gen_every=4500,
                    gpu_data=True, augment=True, compile_model=False,
                    aug_p_transpose=0.9, aug_p_thin=0.5, aug_thin_lo=0.85,
                    notes="RUN DEFINITIVO: ModernTransformer (RoPE+RMSNorm+SwiGLU), 24 "
                          "epocas, batch 32, datos y augmentacion en GPU, torch.compile. "
                          "Config elegida midiendo el optimo de tokens/s en 12.9 GB. "
                          "torch.compile DESACTIVADO: requiere Triton, que no existe "
                          "en Windows (TritonMissing)."))

    # --- 12. PERCEIVER AR (Hawthorne et al., ICML 2022) -----------------------
    # El unico de los cuatro papers investigados que aplica a nuestros datos.
    # En MAESTRO reporta NLL 1.82 frente a 1.84 del Music Transformer, PERO esa
    # cifra NO es trasladable: MAESTRO tiene velocity, duracion y timing continuo
    # (vocabulario de 388), y nosotros onsets binarios a 50 ms (vocabulario 155).
    # n_latents = seq_len a proposito: con menos latentes se pierde la mitad de
    # la senal de entrenamiento (medido: una bpt real de 1.70 se reportaria como
    # 4.49). Con n_latents = L el modelo es un decoder causal con un cross-attend
    # inicial, y su ventaja de contexto largo NO aparece: nuestras piezas son de
    # ~3000 tokens y caben enteras. Se entrena para MEDIRLO, no para suponerlo.
    E.append(Config(name="perceiver_ar", model="perceiver_ar", family="token",
                    d_model=512, n_layers=8, n_heads=8, d_ff=2048, dropout=0.1,
                    tie_weights=True, seq_len=1024, batch_size=32, n_latents=1024,
                    lr=4e-4, warmup=600, min_lr_frac=0.02,
                    steps=steps_for(32, 1024, LONG_BUDGET),
                    eval_every=1000, gen_every=3000,
                    gpu_data=True, augment=True,
                    aug_p_transpose=0.9, aug_p_thin=0.5, aug_thin_lo=0.85,
                    notes="Perceiver AR (ICML 2022) con el mismo presupuesto y tamano "
                          "que estilo_llama_10ep. n_latents=1024=seq_len: con menos, la mitad "
                          "de las posiciones no tienen prediccion y contaminan la "
                          "perdida (medido: bpt 3.52 en vez de ~1.6)."))

    # --- 13. CONTEXTO LARGO, ENTRENADO EN SERIO -------------------------------
    # Observacion del usuario, confirmada midiendo a IGUALDAD de epocas
    # equivalentes (los tokens por paso son los mismos, 16384, asi que la
    # comparacion es justa):
    #     epocas equiv.   0.7      1.0      1.5
    #     music_transformer_ctx2048       2.1634   1.9439   1.8427   <- el mejor en todos los puntos
    #     tft      2.1915   2.0921   2.0033
    #     estilo_llama_10ep       --    2.1663   2.0065
    #     music_transformer       2.5942   2.3275   2.0626
    #     estilo_llama_24ep       --       --    1.9826
    # music_transformer_ctx2048 iba por delante de TODOS y se corto a 1.58 epocas de 3.7 (se pauso
    # para liberar la GPU y no se reanudo). Su 1.8378 salio con el 15% del
    # entrenamiento que recibio estilo_llama_24ep. Con contexto 2048 y RoPE, que
    # extrapola mejor que el skewing, esto merece el run completo.
    E.append(Config(name="estilo_llama_ctx2048_10ep", model="modern", family="token",
                    d_model=512, n_layers=8, n_heads=8, d_ff=2048, dropout=0.1,
                    tie_weights=True, seq_len=2048, batch_size=16,
                    lr=4e-4, warmup=800, min_lr_frac=0.02,
                    steps=steps_for(16, 2048, LONG_BUDGET),
                    eval_every=1000, gen_every=3000,
                    gpu_data=True, augment=True,
                    aug_p_transpose=0.9, aug_p_thin=0.5, aug_thin_lo=0.85,
                    notes="Contexto 2048 con RoPE y ~10 epocas equivalentes. A igualdad "
                          "de tokens, el contexto largo iba por delante de todos los "
                          "modelos de contexto 1024."))

    # --- 14. CONTEXTO 2048 A TOPE: la comparacion decisiva --------------------
    # estilo_llama_ctx2048_10ep (2048, 10 epocas) dio 1.6246 frente a 1.6522 de estilo_llama_10ep
    # (1024, 10 epocas): el contexto largo gana a igualdad de presupuesto, y la
    # ventaja CRECE con el entrenamiento (+0.016 a 2 epocas, +0.071 a 7).
    # Esto lo lleva a las MISMAS 24 epocas equivalentes que estilo_llama_24ep (1.5684),
    # que es la unica forma de saber si la ventaja se mantiene o se estrecha.
    E.append(Config(name="estilo_llama_ctx2048_24ep", model="modern", family="token",
                    d_model=512, n_layers=8, n_heads=8, d_ff=2048, dropout=0.1,
                    tie_weights=True, seq_len=2048, batch_size=16,
                    lr=4e-4, warmup=800, min_lr_frac=0.02,
                    steps=steps_for(16, 2048, MAX_BUDGET),
                    eval_every=1500, gen_every=4500,
                    gpu_data=True, augment=True,
                    aug_p_transpose=0.9, aug_p_thin=0.5, aug_thin_lo=0.85,
                    notes="Contexto 2048 con 24 epocas equivalentes, el MISMO presupuesto "
                          "que estilo_llama_24ep (1024). Comparacion decisiva de contexto."))

    # --- 6. contexto largo: estructura musical de mayor alcance ----------------
    E.append(Config(name="music_transformer_ctx2048", model="music_transformer", family="token",
                    d_model=512, n_layers=8, n_heads=8, d_ff=2048, dropout=0.1,
                    rel_attn=True, max_rel_dist=1024, tie_weights=True,
                    seq_len=2048, batch_size=8, lr=3e-4, warmup=600,
                    steps=steps_for(8, 2048), eval_every=700, gen_every=2100,
                    notes="music_transformer con contexto 2048 tokens (~154 s de musica) para capturar "
                          "estructura de mas largo alcance."))
    return E


def run_isolated(c: Config, resume: bool = False):
    """Entrena un experimento en un SUBPROCESO independiente.

    Cada entrenamiento va en su propio proceso por dos razones:
      * un crash nativo no arrastra al resto de la tanda. Ocurrio de verdad: un
        segmentation fault (codigo 139) de torch/CUDA al cerrar un experimento
        mato la tanda entera y se llevo los cinco pendientes.
      * la memoria de CUDA y los workers del DataLoader se liberan de golpe al
        terminar el proceso, sin depender de que el recolector de basura haga
        su trabajo entre experimentos.
    Devuelve (codigo de salida, etiqueta legible).
    """
    c.save()                                     # experiments/<name>/config.json
    cmd = [sys.executable, str(ROOT / "src" / "train.py"),
           "--config", str(c.dir() / "config.json")]
    if resume:
        cmd.append("--resume")
    r = subprocess.run(cmd, cwd=str(ROOT))
    code = r.returncode
    if code == 0:
        return 0, "OK"
    if code < 0 or code > 128:
        return code, "CRASH (senal %d)" % (abs(code) if code < 0 else code - 128)
    return code, "FALLO (%d)" % code


def apply_smoke(c: Config) -> Config:
    c.name = "smoke_" + c.name
    c.steps = 150; c.eval_every = 75; c.gen_every = 150; c.warmup = 20
    c.log_every = 25; c.val_windows = 24; c.n_gen_samples = 4
    c.gen_steps = 200; c.train_windows = 4000; c.num_workers = 2
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="run corto para validar el pipeline")
    ap.add_argument("--only", type=str, default=None, help="nombres separados por coma")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--no-retry", action="store_true",
                    help="no reintentar un experimento que falle")
    a = ap.parse_args()

    exps = matrix()
    if a.list:
        print("presupuesto: %d M tokens por experimento (~%.1f epocas del corpus de train)"
              % (TOKEN_BUDGET / 1e6, TOKEN_BUDGET / 32_245_043))
        print("%-12s %-18s %-6s %7s %6s %5s %6s %s" %
              ("nombre", "modelo", "fam", "pasos", "batch", "ctx", "d_model", "tokens"))
        for c in exps:
            print("%-12s %-18s %-6s %7d %6d %5d %6d  %5.0f M" %
                  (c.name, c.model, c.family, c.steps, c.batch_size, c.seq_len,
                   c.d_model, c.steps * c.batch_size * c.seq_len / 1e6))
        return
    if a.only:
        want = {s.strip() for s in a.only.split(",")}
        exps = [c for c in exps if c.name in want]
        if not exps:
            print("ningun experimento coincide con %s" % want)
            return
    if a.smoke:
        exps = [apply_smoke(c) for c in exps]

    results = []
    for c in exps:
        print("\n" + "=" * 78)
        print("EXPERIMENTO %s  (%s, familia %s, %d pasos)" % (c.name, c.model, c.family, c.steps))
        print("=" * 78, flush=True)
        t0 = time.time()
        code, status = run_isolated(c, resume=a.resume)
        if code != 0 and not a.no_retry:
            print("\n[%s] salio con codigo %d; se reintenta una vez reanudando desde last.pt"
                  % (c.name, code), flush=True)
            code, status = run_isolated(c, resume=True)
            if code == 0:
                status = "OK (reintento)"
        results.append((c.name, status, (time.time() - t0) / 60))

    print("\n" + "=" * 78)
    print("RESUMEN DE LA TANDA")
    for n, s, mins in results:
        print("  %-20s %-6s %6.1f min" % (n, s, mins))
    try:
        import registry as reg
        df = reg.rebuild_leaderboard()
        print("\nLEADERBOARD\n" + df.to_string(index=False))
    except Exception as e:
        print("no se pudo construir el leaderboard: %s" % e)


if __name__ == "__main__":
    main()
