"""Par A/B: el MISMO prefijo, continuado por el modelo y por la pieza real.

    python scripts/17_par_real_vs_modelo.py                      # los casos por defecto
    python scripts/17_par_real_vs_modelo.py --exp lstm --m 1     # uno concreto
    python scripts/17_par_real_vs_modelo.py --todos              # las 3 muestras de cada modelo

Por que hace falta: los wav de 1_continuaciones_corpus contienen SOLO lo que
genero el modelo, sin el prefijo. Asi no se puede juzgar lo unico que de verdad
mide ese escenario -- si la continuacion sigue al prefijo -- ni compararla con lo
que la pieza hacia realmente a partir de ese punto.

Como se reconstruye el prefijo: `make_token_prompts` es determinista dada su
semilla, y la semilla fue 1000 + paso, con el paso escrito en el nombre del npz.
Se reproducen los prompts, se localiza el prefijo dentro de la pieza original
buscando la subsecuencia exacta de tokens, y de ahi se corta lo que seguia.

Salida, junto a los wav existentes de cada modelo:
    <exp>_m01_par_modelo.wav    prefijo + continuacion del modelo
    <exp>_m01_par_real.wav      prefijo + continuacion REAL de la pieza

AVISO: `scripts/11_audio_gallery.py` borra y rehace 1_continuaciones_corpus, asi
que si lo vuelves a lanzar hay que relanzar este despues.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import audio_play                                    # noqa: E402
from midi_export import roll_to_midi                 # noqa: E402
import prompts as pr                                 # noqa: E402
from data.datasets import get_tokens, get_roll       # noqa: E402
from data.tokenizer import decode_tokens             # noqa: E402

STEP_SEC = 0.05
GEN_STEPS = 800
AUDIO = ROOT / "reports" / "audio" / "1_continuaciones_corpus"

# Los dos que se estaban comparando; --todos recorre los 14 modelos.
POR_DEFECTO = [("estilo_llama_ctx2048_24ep", 1), ("estilo_llama_24ep", 3)]


def synth_par(rolls: dict, destino_de) -> None:
    """Sintetiza varias pistas con UNA ganancia comun.

    `synthesize_musicbox_roll` normaliza por el pico de cada senal. Si cada
    fichero se normaliza por su cuenta, el prefijo -- que es identico en los dos
    -- suena a volumenes distintos (aqui salia un factor 1.11), y al comparar se
    confunde un cambio de densidad con un cambio de calidad. Se sintetiza sin
    normalizar y se escala todo por el mismo pico.
    """
    crudos = {}
    for etiqueta, roll in rolls.items():
        audio, sr = audio_play.synthesize_musicbox_roll(
            np.asarray(roll), step_sec=STEP_SEC, note_min=21,
            representation="onset", normalize=False)
        crudos[etiqueta] = audio
    pico = max((float(np.abs(a).max()) for a in crudos.values()), default=0.0)
    k = 0.95 / pico if pico > 0 else 1.0
    for etiqueta, audio in crudos.items():
        audio_play.save_wav(destino_de(etiqueta), audio * k, sr)


def paso_del_npz(exp: str) -> int | None:
    """El paso cuya generacion se convirtio en audio (el de mejor gen_score)."""
    bj = ROOT / "experiments" / exp / "logs" / "best.json"
    if bj.exists():
        try:
            d = json.load(open(bj, encoding="utf-8"))
            for k in ("best_gen_score_step", "best_gen_step"):
                if d.get(k):
                    return int(d[k])
        except Exception:
            pass
    gens = sorted((ROOT / "experiments" / exp / "generations").glob("gen_step*.npz"))
    return int(gens[-1].stem.replace("gen_step", "")) if gens else None


def localizar(pref_tokens: np.ndarray, tk: np.ndarray) -> int | None:
    """Offset del prefijo dentro de la pieza, por coincidencia exacta de tokens."""
    n = len(pref_tokens)
    if n == 0 or len(tk) < n:
        return None
    # se compara primero el token inicial para no recorrer toda la pieza a ciegas
    for j in np.flatnonzero(tk[: len(tk) - n + 1] == pref_tokens[0]):
        if np.array_equal(tk[j:j + n], pref_tokens):
            return int(j)
    return None


def un_caso(exp: str, m: int) -> bool:
    """m es 1-indexado, como en el nombre del fichero (_m01 -> m=1)."""
    i = m - 1
    paso = paso_del_npz(exp)
    if paso is None:
        print("  %s: sin generaciones" % exp); return False
    npz = ROOT / "experiments" / exp / "generations" / ("gen_step%07d.npz" % paso)
    z = np.load(npz)
    if i >= len(z["rolls"]):
        print("  %s: no tiene muestra m%02d" % (exp, m)); return False
    if "srcs" not in z.files:
        print("  %s: el npz no guarda la pieza de origen" % exp); return False

    cfg = json.load(open(ROOT / "experiments" / exp / "config.json", encoding="utf-8"))
    px, pst, srcs = pr.make_token_prompts(int(cfg.get("n_gen_samples", 16)),
                                          int(cfg.get("gen_prime_steps", 200)),
                                          seed=1000 + paso)
    if int(srcs[i]) != int(z["srcs"][i]) or int(pst[i]) != int(z["prime_steps"][i]):
        # si no cuadra, el prefijo reconstruido NO es el que vio el modelo: mejor
        # no escribir nada que escribir un par enganoso
        print("  %s m%02d: los prompts no se reproducen, se omite" % (exp, m)); return False

    pref_tokens = px[i].numpy()[1:]                  # sin BOS
    tk = np.asarray(get_tokens(int(srcs[i])))
    s = localizar(pref_tokens, tk)
    if s is None:
        print("  %s m%02d: prefijo no localizado en la pieza" % (exp, m)); return False

    roll = get_roll(int(srcs[i]))
    ini = len(decode_tokens(tk[:s]))                 # pasos de la pieza antes del prefijo
    n_pref = int(pst[i])
    prefijo = np.asarray(roll[ini:ini + n_pref], np.uint8)
    real = np.asarray(roll[ini + n_pref: ini + n_pref + GEN_STEPS], np.uint8)
    modelo = np.asarray(z["rolls"][i], np.uint8)

    if len(real) == 0:
        print("  %s m%02d: la pieza se acaba en el prefijo, sin continuacion real" % (exp, m))
        return False

    d = AUDIO / exp
    d.mkdir(parents=True, exist_ok=True)
    base = "%s_m%02d_par" % (exp, m)
    pistas = {et: np.concatenate([prefijo, cont])
              for et, cont in (("modelo", modelo), ("real", real))}
    synth_par(pistas, lambda et: d / ("%s_%s.wav" % (base, et)))
    for et, full in pistas.items():
        roll_to_midi(full, d / ("%s_%s.mid" % (base, et)))
    print("  %-26s m%02d | pieza %d (val) | prefijo %.1f s | modelo %d notas, real %d notas"
          % (exp, m, int(srcs[i]), n_pref * STEP_SEC, int(modelo.sum()), int(real.sum())))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=None)
    ap.add_argument("--m", type=int, default=1, help="numero de muestra, como en _m01")
    ap.add_argument("--todos", action="store_true", help="las 3 muestras de cada modelo")
    a = ap.parse_args()

    if a.todos:
        casos = [(p.name, m) for p in sorted((ROOT / "experiments").iterdir())
                 if p.is_dir() and not p.name.startswith("smoke_") for m in (1, 2, 3)]
    elif a.exp:
        casos = [(a.exp, a.m)]
    else:
        casos = POR_DEFECTO

    print("prefijo + continuacion, en dos versiones (modelo y pieza real):")
    n = sum(un_caso(e, m) for e, m in casos)
    print("\n%d de %d pares escritos en reports/audio/1_continuaciones_corpus/<modelo>/" % (n, len(casos)))


if __name__ == "__main__":
    main()
