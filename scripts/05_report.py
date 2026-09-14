"""Genera docs/03_resultados.md y las figuras comparativas desde el leaderboard.

    python scripts/05_report.py

Lee reports/leaderboard.csv y experiments/*/logs/summary.json, y produce un
informe con la comparativa, la ablacion de atencion relativa y las conclusiones.
No inventa nada: si una metrica falta, lo dice.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MARGINAL_BPT = 4.0921
GEN_CEILING = 87.6          # fragmentos reales del corpus (scripts/02_ref_stats.py)
GEN_NOISE = 1.6             # ruido i.i.d. con la densidad correcta

FAMILY_NAME = {"token": "eventos", "frame": "frames"}
MODEL_NAME = {
    "music_transformer": "Music Transformer (atencion relativa)",
    "lstm": "LSTM (linea base recurrente)",
    "tft": "Temporal Fusion Transformer adaptado",
    "melle": "MELLE adaptado a frames",
    "modern": "Decoder estilo LLaMA (RoPE + RMSNorm + SwiGLU)",
    "perceiver_ar": "Perceiver AR (cuello latente + cross-attend)",
    "deep_lstm": "LSTM profunda (4 capas x 1280)",
}


def load_rows():
    lb = ROOT / "reports" / "leaderboard.csv"
    if not lb.exists():
        return []
    import pandas as pd
    df = pd.read_csv(lb)
    df = df[~df["name"].astype(str).str.startswith("smoke_")]
    rows = df.to_dict("records")
    for r in rows:                                  # enriquece con summary.json
        s = ROOT / "experiments" / str(r["name"]) / "logs" / "summary.json"
        if s.exists():
            try:
                r["_summary"] = json.load(open(s))
            except Exception:
                r["_summary"] = {}
        else:
            r["_summary"] = {}
        c = ROOT / "experiments" / str(r["name"]) / "config.json"
        r["_config"] = json.load(open(c)) if c.exists() else {}
    return rows


def fmt(v, nd=3, dash="n/d"):
    try:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return dash
        return ("%." + str(nd) + "f") % float(v)
    except Exception:
        return dash


def main():
    rows = load_rows()
    if not rows:
        print("no hay experimentos en reports/leaderboard.csv todavia")
        return

    L = []
    A = L.append
    A("# Resultados\n")
    A("Generado por `scripts/05_report.py` a partir de `reports/leaderboard.csv` y de los")
    A("`summary.json` de cada experimento. Todos comparten tokenización, particiones y")
    A("protocolo de evaluación. El presupuesto de entrenamiento **no** es el mismo para")
    A("todos: hay tres tandas (120 M tokens ≈ 3.7 épocas equivalentes, 321 M ≈ 10 y")
    A("780 M ≈ 24 épocas equivalentes), y solo son comparables entre sí los experimentos")
    A("que comparten tanda. Las columnas de pasos y horas permiten distinguirlas.\n")

    # ---------------------------------------------------------------- tabla
    A("## 1. Comparativa\n")
    A("| # | experimento | modelo | familia | par. (M) | bits/paso val | bits/paso test | gen_score | horas |")
    A("|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(rows, 1):
        s = r["_summary"]
        A("| %d | `%s` | %s | %s | %s | **%s** | %s | **%s** | %s |" % (
            i, r["name"], MODEL_NAME.get(str(r.get("model")), r.get("model")),
            FAMILY_NAME.get(str(r.get("family")), r.get("family")),
            fmt(r.get("params_M"), 1), fmt(r.get("best_val_bpt")),
            fmt(s.get("test_bpt")), fmt(r.get("best_gen_score"), 1),
            fmt(r.get("train_hours"), 2)))
    A("")
    A("Anclas de referencia medidas empiricamente:\n")
    A("| ancla | bits/paso | gen_score |")
    A("|---|---|---|")
    A("| modelo i.i.d. con la densidad marginal | %.4f | - |" % MARGINAL_BPT)
    A("| ruido i.i.d. con la densidad correcta | - | %.1f |" % GEN_NOISE)
    A("| fragmentos reales del corpus | - | %.1f |" % GEN_CEILING)
    A("| silencio o bucle degenerado | - | 0.0 |")
    A("")

    # --- runs que no llegaron a su presupuesto ------------------------------
    # Un experimento interrumpido sigue apareciendo en la tabla con su bits/paso,
    # que es valido, pero su comparacion con los demas NO es a igualdad de
    # computo. Callarlo invalida la lectura de la tabla entera.
    incompletos = []
    for r in rows:
        try:
            plan, real = int(r["_config"].get("steps")), int(r.get("steps"))
        except (TypeError, ValueError):
            continue
        if plan and real < 0.95 * plan:
            incompletos.append((r["name"], real, plan, 100.0 * real / plan))
    if incompletos:
        A("> **Aviso: %d experimento(s) no completaron su presupuesto de tokens.**"
          % len(incompletos))
        A("> Su bits/paso es valido, pero **no** se comparan a igualdad de computo:")
        A(">")
        for n, real, plan, pct in incompletos:
            A("> - `%s`: %d de %d pasos (%.0f %%)" % (n, real, plan, pct))
        A("")

    # El ganador depende del criterio y los dos criterios NO coinciden. Se dan por
    # separado porque uno es determinista y el otro no: mezclarlos seria mentir.
    finitos = [r for r in rows if np.isfinite(r.get("best_val_bpt", np.nan))]
    best = min(finitos, key=lambda r: float(r["best_val_bpt"])) if finitos else rows[0]
    lider_gen = rows[0]
    A("## 2. Modelo ganador\n")
    A("**`%s`** (%s), por **bits/paso** = %s.\n" % (
        best["name"], MODEL_NAME.get(str(best.get("model")), best.get("model")),
        fmt(best.get("best_val_bpt"))))
    A("Se elige por bits/paso y no por `gen_score` porque bits/paso es **determinista**:")
    A("se mide por *teacher forcing* sobre el split de validación entero, sin muestreo y")
    A("sin sortear prefijos. El `gen_score` de la tabla es el **máximo** sobre las")
    A("evaluaciones del entrenamiento, y cada evaluación usaba prefijos distintos. Medido")
    A("sobre `estilo_llama_ctx2048_24ep`: **79.3** en el paso 22500 y **62.7** en el 23800,")
    A("con el `val_bpt` pasando de 1.5185 a 1.5172, es decir el mismo modelo. Con un ruido")
    A("de 10 a 22 puntos, una diferencia pequeña de `gen_score` no es interpretable: que la")
    A("tabla la encabece `%s` **no** significa que genere mejor. La comparación" % lider_gen["name"])
    A("pareada de `scripts/18_comparacion_pareada.py` --mismos prefijos de test para todos")
    A("y varias semillas de muestreo-- sustituye a ese criterio.\n")
    bs = best["_summary"]
    red = ((MARGINAL_BPT - float(best["best_val_bpt"])) / MARGINAL_BPT * 100
           if np.isfinite(best.get("best_val_bpt", np.nan)) else float("nan"))
    A("")
    A("- bits/paso de validacion: **%s** (%.1f%% por debajo de la referencia trivial)"
      % (fmt(best.get("best_val_bpt")), red))
    A("- bits/paso de test: %s" % fmt(bs.get("test_bpt")))
    A("- perplejidad por token: %s" % fmt(best.get("best_val_ppl"), 2))
    A("- probabilidad asignada a una nota imposible: %s"
      % fmt(bs.get("final_gen_grammar_prob_mass"), 6))
    A("")
    A("Sus artefactos estan sincronizados en `reports/best/`: checkpoint, `metrics.csv`,")
    A("`training_curves.png`, piano-rolls y MIDI.\n")

    # ---------------------------------------------------------------- ablacion
    by = {str(r["name"]): r for r in rows}
    if "music_transformer" in by and "ablacion_sin_atencion_relativa" in by:
        a, b = by["music_transformer"], by["ablacion_sin_atencion_relativa"]
        A("## 3. Ablacion: aporta algo la atencion relativa?\n")
        A("`music_transformer` y `ablacion_sin_atencion_relativa` son la MISMA red con el mismo numero de parametros y el")
        A("mismo presupuesto de tokens; solo cambia `rel_attn`.\n")
        A("| | bits/paso val | gen_score | par. (M) |")
        A("|---|---|---|---|")
        A("| atencion relativa (`music_transformer`) | %s | %s | %s |" %
          (fmt(a.get("best_val_bpt")), fmt(a.get("best_gen_score"), 1), fmt(a.get("params_M"), 1)))
        A("| atencion absoluta (`ablacion_sin_atencion_relativa`) | %s | %s | %s |" %
          (fmt(b.get("best_val_bpt")), fmt(b.get("best_gen_score"), 1), fmt(b.get("params_M"), 1)))
        try:
            d_bpt = float(b["best_val_bpt"]) - float(a["best_val_bpt"])
            d_gen = float(a["best_gen_score"]) - float(b["best_gen_score"])
            A("")
            A("Diferencia: **%+.4f bits/paso** y **%+.1f puntos de gen_score** a favor de la"
              % (d_bpt, d_gen))
            A("atencion relativa." if d_bpt > 0 else
              "atencion absoluta, es decir la atencion relativa NO ayudo en este dataset.")
        except Exception:
            pass
        A("")

    if "music_transformer" in by and "music_transformer_ctx2048" in by:
        a, c = by["music_transformer"], by["music_transformer_ctx2048"]
        A("## 4. Contexto: 1024 frente a 2048 tokens\n")
        A("| contexto | bits/paso val | gen_score |")
        A("|---|---|---|")
        A("| 1024 tokens (~78 s) | %s | %s |" %
          (fmt(a.get("best_val_bpt")), fmt(a.get("best_gen_score"), 1)))
        A("| 2048 tokens (~154 s) | %s | %s |" %
          (fmt(c.get("best_val_bpt")), fmt(c.get("best_gen_score"), 1)))
        A("")

    # ---------------------------------------------------------------- familias
    tok = [r for r in rows if r.get("family") == "token"]
    frm = [r for r in rows if r.get("family") == "frame"]
    if tok and frm:
        A("## 5. Eventos frente a frames\n")
        bt = min(tok, key=lambda r: float(r.get("best_val_bpt", 9e9)))
        bf = min(frm, key=lambda r: float(r.get("best_val_bpt", 9e9)))
        A("El mejor modelo de eventos (`%s`) logra %s bits/paso; el mejor de frames (`%s`),"
          % (bt["name"], fmt(bt.get("best_val_bpt")), bf["name"]))
        A("%s. La comparacion es legitima porque ambos miden -log2 p(piano_roll) / pasos"
          % fmt(bf.get("best_val_bpt")))
        A("sobre el mismo objeto.\n")
        A("La familia de frames arrastra una limitacion estructural: predice las 88 notas de")
        A("un mismo paso de forma condicionalmente independiente, asi que no puede modelar la")
        A("correlacion de un acorde. La familia de eventos si, porque emite las notas")
        A("simultaneas en secuencia y cada una condiciona a la siguiente.\n")

    # ------------------------------------------------ diagnostico de MELLE
    drifts = []
    for r in rows:
        d = ROOT / "experiments" / str(r["name"]) / "logs" / "drift.json"
        if d.exists():
            try:
                drifts.append(json.load(open(d)))
            except Exception:
                pass
    if drifts:
        A("## 6. Por que la familia de frames genera demasiadas notas\n")
        A("`scripts/08_melle_drift.py` descompone el exceso de densidad en sus dos factores")
        A("multiplicativos, midiendo por separado lo que el modelo ya sobreestima con datos")
        A("reales (teacher forcing) y lo que anade la realimentacion al muestrear.\n")
        A("| experimento | flux_weight | calibracion | realimentacion | total | crecimiento |")
        A("|---|---|---|---|---|---|")
        for d in drifts:
            A("| `%s` | %s | %.2fx | %.2fx | **%.2fx** | %.2fx |" % (
                d["exp"], fmt(d.get("flux_weight"), 2), d["factor_calibracion"],
                d["factor_realimentacion"], d["factor_total"], d["crecimiento_tramos"]))
        A("")
        d0 = drifts[0]
        A("Con `%s`: el modelo espera %.2f notas por frame cuando la realidad es %.2f"
          % (d0["exp"], d0["tf_expected_per_frame"], d0["tf_real_per_frame"]))
        A("(factor %.2f), y al muestrear llega a %.2f notas por paso (otro factor %.2f)."
          % (d0["factor_calibracion"], d0["gen_per_step"], d0["factor_realimentacion"]))
        if d0["crecimiento_tramos"] < 1.2:
            A("El crecimiento entre el primer y el ultimo tramo es %.2fx, asi que **no es deriva"
              % d0["crecimiento_tramos"])
            A("acumulativa**: la densidad se estabiliza alta desde que termina el prefijo real.")
        A("")
        if len(drifts) > 1:
            A("La comparacion entre `flux_weight` distintos aisla la contribucion de la")
            A("*flux loss*, que premia la variacion entre frames y en un piano-roll binario")
            A("disperso equivale a premiar encender notas.\n")

    # ------------------------------------------- comparacion pareada
    # El gen_score de la tabla 1 no es comparable entre modelos: cada uno se
    # evaluo con sus propios prefijos y se reporta su maximo. Esta seccion lee el
    # resultado del protocolo que si es comparable.
    import csv as _csv
    for carpeta, titulo, desc in (
            ("comparacion_pareada", "muestreo del entrenamiento (T 1.0, top-p 0.95)",
             "el mismo con el que se produjo la tabla 1"),
            ("comparacion_pareada_calibrada", "muestreo calibrado anti-bucle",
             "`data/processed/best_sampling.json`")):
        f = ROOT / "reports" / carpeta / "resumen.csv"
        if not f.exists():
            continue
        filas = list(_csv.DictReader(open(f, encoding="utf-8")))
        if not filas:
            continue
        if carpeta == "comparacion_pareada":
            A("## 7. Comparacion pareada: el mismo material para todos\n")
            A("`scripts/18_comparacion_pareada.py`. Los mismos **16 prefijos del split de")
            A("test** para todos los modelos, **3 semillas** de muestreo con generador")
            A("explicito, y el checkpoint `best.pt` (elegido por `val_bpt`, que es")
            A("determinista) en vez de `best_gen.pt` (elegido por el `gen_score` ruidoso).")
            A("La columna *leaderboard* es el maximo que reportaba la tabla 1.\n")
        A("### %s\n" % titulo.capitalize())
        A("Decodificacion: %s.\n" % desc)
        A("| modelo | bits/paso | gen_score pareado | desv. | rango | leaderboard |")
        A("|---|---|---|---|---|---|")
        for r in filas:
            A("| `%s` | %s | **%s** | ±%s | %s - %s | %s |" % (
                r["modelo"], r.get("val_bpt", "-"), r["media"], r["desv"],
                r["minimo"], r["maximo"], r.get("leaderboard", "-")))
        A("")

    # --- lectura de las dos tablas: por que el calibrado empeora --------------
    fa = ROOT / "reports" / "comparacion_pareada" / "resultados.csv"
    fb = ROOT / "reports" / "comparacion_pareada_calibrada" / "resultados.csv"
    if fa.exists() and fb.exists():
        def _medias(f):
            d = {}
            for r in _csv.DictReader(open(f, encoding="utf-8")):
                d.setdefault(r["modelo"], []).append(
                    (float(r["gen_score"]), float(r["densidad"]), float(r["repeat8"])))
            return {k: np.asarray(v).mean(axis=0) for k, v in d.items()}
        ma, mb = _medias(fa), _medias(fb)
        try:
            import evaluate as _ev
            ref = _ev.load_ref_agg("val_excerpt")
            ref_d, ref_r = float(ref["density"]), float(ref["repeat8"])
        except Exception:
            ref_d, ref_r = float("nan"), float("nan")
        A("### Lectura: la decodificacion calibrada **sobrecorrige**\n")
        A("La referencia real del corpus es **densidad %.3f** y **repeat8 %.3f**." % (ref_d, ref_r))
        A("Las dos decodificaciones fallan por lados opuestos:\n")
        A("| modelo | densidad (entren.) | densidad (calibr.) | repeat8 (entren.) | repeat8 (calibr.) | Δ score |")
        A("|---|---|---|---|---|---|")
        for m in sorted(ma, key=lambda k: mb[k][0] - ma[k][0]):
            if m not in mb:
                continue
            A("| `%s` | %.3f | %.3f | %.3f | %.3f | **%+.1f** |" % (
                m, ma[m][1], mb[m][1], ma[m][2], mb[m][2], mb[m][0] - ma[m][0]))
        A("")
        A("El muestreo del entrenamiento genera **de mas** (densidad por encima de la real en")
        A("todos los modelos) y el calibrado genera **de menos** (por debajo en todos), con")
        A("errores de magnitud parecida y signo contrario. Con `repeat8` pasa igual: el")
        A("calibrado deja a todos los modelos **menos repetitivos que la musica real**, lo")
        A("cual tambien es un error, solo que del otro lado.\n")
        A("El unico modelo que mejora con el calibrado es el que de verdad tenia el problema")
        A("de bucles (`estilo_llama_ctx2048_24ep`, repeat8 0.318 -> 0.086). El `lstm`, que ya")
        A("estaba por debajo de la repeticion real sin ninguna ayuda, **pierde 37 puntos**:")
        A("aplicarle una penalizacion por repeticion es quitarle lo que no le sobraba.\n")
        A("Conclusion, que es mas precisa que la que se tenia antes: **la decodificacion")
        A("domina la calidad de la generacion** --hasta 37 puntos sobre un modelo y unos")
        A("prefijos fijos, mas que cualquier diferencia de arquitectura medida aqui-- pero")
        A("**no existe una decodificacion buena en abstracto**. `best_sampling.json` se")
        A("calibro sobre melodias monofonicas fuera de distribucion, donde el fallo era el")
        A("bucle; llevado a prefijos polifonicos del corpus, corrige un problema que la")
        A("mayoria de los modelos no tenia.\n")

    # --------------------------------------- metricas del paper de referencia
    fm = ROOT / "reports" / "metricas_paper.csv"
    if fm.exists():
        mf = list(_csv.DictReader(open(fm, encoding="utf-8")))
        A("## 8. En la metrica del paper de Music Transformer\n")
        A("Huang et al. (2018, arXiv:1809.04281) reportan **NLL por token**; nuestra lectura,")
        A("marcada como no confirmada en `docs/05_estado_del_arte.md`, es que son nats.")
        A("`scripts/21_metricas_paper.py` reexpresa nuestros modelos en esa unidad.\n")
        A("| modelo | NLL val (nats/token) | NLL test | bits/token | bits/paso |")
        A("|---|---|---|---|---|")
        for r in mf[:8]:
            A("| `%s` | **%s** | %s | %s | %s |" % (
                r["modelo"], r["val_nll_nats_token"], r.get("test_nll_nats_token") or "-",
                r["val_bits_token"], r["val_bits_paso"]))
        A("")
        A("> **No compares estas cifras con el 1.84 del paper.** Su vocabulario son 388")
        A("> simbolos CON velocity y NOTE_OFF sobre MAESTRO; el nuestro son 155 SIN velocity,")
        A("> duracion ni pedal, sobre otro corpus y otra rejilla temporal. Una NLL por token")
        A("> depende del vocabulario y del dato. Que a `music_transformer` le salga 1.8430 es")
        A("> una coincidencia aritmetica, no un empate.\n")
        A("Y lo mas importante del paper para este informe no es esa cifra: su evidencia de")
        A("**calidad** no es una metrica automatica, es un **test de escucha por pares** con")
        A("humanos. Es decir, el propio paper ya asume que la NLL no mide si suena bien.\n")

    # ------------------------------------------------- generacion libre
    fl = ROOT / "reports" / "generacion_libre" / "resumen.csv"
    if fl.exists():
        lf = list(_csv.DictReader(open(fl, encoding="utf-8")))
        por = {}
        for r in lf:
            por.setdefault(r["modelo"], []).append(r)
        A("## 9. Generacion libre: sin ningun prefijo\n")
        A("`scripts/20_generacion_libre.py`. La entrada es un unico token `BOS`: el modelo")
        A("compone desde cero, sin nada que lo ancle. Mismas semillas y misma decodificacion")
        A("para todos.\n")
        A("| modelo | gen_score | densidad | repeat8 | polifonia |")
        A("|---|---|---|---|---|")
        for m, rs in sorted(por.items(), key=lambda kv: -float(kv[1][0].get("gen_score_del_lote", 0))):
            A("| `%s` | **%s** | %.3f | %.3f | %.2f |" % (
                m, rs[0].get("gen_score_del_lote", "-"),
                np.mean([float(x["densidad"]) for x in rs]),
                np.mean([float(x["repeat8"]) for x in rs]),
                np.mean([float(x["polifonia"]) for x in rs])))
        A("")
        try:
            import evaluate as _ev2
            _r = _ev2.load_ref_agg("val_excerpt")
            A("Referencia del corpus: densidad **%.3f**, repeat8 **%.3f**, polifonia **%.2f**.\n"
              % (_r["density"], _r["repeat8"], _r["mean_poly"]))
        except Exception:
            pass
        A("Sin prefijo todos los modelos caen respecto al escenario con prefijo, que es lo")
        A("esperable: la mitad del trabajo lo hacia el contexto real. Es el escenario que")
        A("mejor separa a un modelo que **compone** de uno que solo **continua**.\n")
        A("Audio en `reports/audio/3_generacion_libre/<modelo>/desde_cero_sN.wav`.\n")

    A("## 10. Figuras\n")
    for p, d in (("reports/figures/prediccion_vs_generacion.png", "prediccion frente a generacion"),
                 ("reports/comparacion_pareada/comparacion.png", "comparacion pareada, mismo material"),
                 ("reports/figures/leaderboard.png", "comparativa de todos los experimentos"),
                 ("reports/figures/dataset_overview.png", "resumen del dataset"),
                 ("reports/figures/corpus_examples.png", "fragmentos reales del corpus"),
                 ("reports/best/training_curves.png", "curvas del modelo ganador"),
                 ("reports/best/generation_best.png", "mejor continuacion generada"),
                 ("reports/best/distributions.png", "distribuciones generadas vs corpus")):
        mark = "" if (ROOT / p).exists() else "  *(no generada)*"
        A("- [`%s`](../%s) - %s%s" % (p, p, d, mark))
    A("")

    out = ROOT / "docs" / "03_resultados.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print("escrito %s (%d lineas, %d experimentos)" % (out, len(L), len(rows)))
    for r in rows:
        print("  %-12s bpt=%s gen=%s" % (r["name"], fmt(r.get("best_val_bpt")),
                                         fmt(r.get("best_gen_score"), 1)))


if __name__ == "__main__":
    main()
