"""Reorganiza todo el audio en una estructura unica y coherente.

    python scripts/15_organizar_audio.py

Consolida en reports/audio/, ordenado POR ESCENARIO y dentro por modelo:

  reports/audio/
    0_referencia_corpus/        musica REAL del dataset (el objetivo)
    1_continuaciones_corpus/    continuaciones de piezas del corpus, por modelo
    2_prefijos_de_prueba/       continuaciones de los .npz de evaluacion, por modelo
    3_generacion_libre/         generado sin ningun prefijo

Por que asi: el criterio es DE DONDE SALE EL PREFIJO, que es lo que determina si
un resultado es comparable con otro. Ordenar por modelo primero mezclaria
escenarios que no se pueden comparar entre si.

Es idempotente: reconstruye el arbol desde las fuentes cada vez.
"""
from __future__ import annotations
import json, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIO = ROOT / "reports" / "audio"

ESCENARIOS_PREFIJOS = {"0", "1", "2", "3"}

ESCENARIOS = {
    "0_referencia_corpus": "musica real del dataset, sin modelo: el objetivo",
    "1_continuaciones_corpus": "prefijo de una pieza del corpus (caso de entrenamiento)",
    "2_prefijos_de_prueba": "prefijo de los .npz de evaluacion (caso de la prueba)",
    "3_generacion_libre": "sin prefijo: el modelo compone desde cero",
}


def canon() -> dict:
    """Mapa nombre_de_trabajo -> nombre final, tal y como lo dejo el script 16."""
    p = ROOT / "reports" / "nombres_experimentos.json"
    if not p.exists():
        return {}
    try:
        return json.load(open(p, encoding="utf-8")).get("mapa", {})
    except Exception:
        return {}


def sanear_nombres(raiz: Path) -> int:
    """Carpetas de modelo que todavia llevan un nombre de trabajo.

    Si el destino ya existe es que una tanda posterior ya escribio ese audio con
    el nombre nuevo: la carpeta vieja es un duplicado del mismo modelo y se
    descarta (el arbol se vuelve a rellenar desde las fuentes mas abajo).
    """
    m = canon()
    if not raiz.exists() or not m:
        return 0
    n = 0
    # de mas profundo a menos: renombrar un padre invalidaria las rutas hijas
    for d in sorted((p for p in raiz.rglob("*") if p.is_dir()),
                    key=lambda p: len(p.parts), reverse=True):
        nuevo = m.get(d.name)
        if not nuevo:
            continue
        dst = d.parent / nuevo
        if dst.exists():
            shutil.rmtree(d, ignore_errors=True)
        else:
            d.rename(dst)
        n += 1
    return n


def mover(origen: Path, destino: Path, patron: str = "*") -> int:
    """Copia los ficheros de audio de una carpeta a otra. Devuelve cuantos."""
    if not origen.exists():
        return 0
    destino.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(origen.glob(patron)):
        if f.suffix.lower() in (".wav", ".mid", ".png", ".npz") and f.is_file():
            shutil.copy2(f, destino / f.name)
            n += 1
    return n


def scores() -> dict:
    """bits/paso y gen_score de cada modelo, para el indice."""
    lb = ROOT / "reports" / "leaderboard.csv"
    if not lb.exists():
        return {}
    import pandas as pd
    out = {}
    for _, r in pd.read_csv(lb).iterrows():
        out[str(r["name"])] = (r.get("best_val_bpt"), r.get("best_gen_score"))
    return out


def main():
    # No se renombra la carpeta: en Windows el explorador la bloquea si esta
    # abierta. Se construye el arbol nuevo dentro y se borran las viejas al final.
    AUDIO.mkdir(parents=True, exist_ok=True)
    viejo = AUDIO                               # las fuentes viven en la propia audio/
    antiguas = [d for d in AUDIO.iterdir()
                if d.is_dir() and d.name[0].isdigit() and "_" in d.name
                and d.name.split("_", 1)[0] not in ESCENARIOS_PREFIJOS]

    total = 0
    k = sanear_nombres(AUDIO)
    if k:
        print("%d carpetas de audio con nombre viejo, renombradas o deduplicadas" % k)

    # --- 0. referencia del corpus ---
    for src in (viejo / "00_CORPUS_real",):
        total += mover(src, AUDIO / "0_referencia_corpus")

    # --- 1. continuaciones desde el corpus, por modelo ---
    for d in sorted(viejo.glob("*_*")):         # carpetas del tipo 03_modelo_genNN
        if d.name.startswith("00_") or d.name[0] in ESCENARIOS_PREFIJOS:
            continue                            # no reprocesar el arbol nuevo
        partes = d.name.split("_", 1)[1]        # quita el prefijo numerico
        modelo = partes.rsplit("_gen", 1)[0]
        total += mover(d, AUDIO / "1_continuaciones_corpus" / modelo)
    for d in sorted((ROOT / "reports" / "escuchar").glob("*corpus*")):
        modelo = d.name.split("corpus_", 1)[-1]
        total += mover(d, AUDIO / "1_continuaciones_corpus" / modelo)

    # --- 2. prefijos de prueba, por conjunto y modelo ---
    ext = ROOT / "reports" / "external_eval"
    for conj in sorted(p for p in ext.iterdir() if p.is_dir()) if ext.exists() else []:
        nombre = ("melodias_5s" if "5s" in conj.name else
                  "muestras_densas" if "02" in conj.name else conj.name)
        for mod in sorted(p for p in conj.iterdir() if p.is_dir()):
            total += mover(mod, AUDIO / "2_prefijos_de_prueba" / nombre / mod.name)
    esc = ROOT / "reports" / "escuchar"
    for d in sorted(esc.glob("*melodias*")) if esc.exists() else []:
        modelo = d.name.split("melodias_", 1)[-1]
        total += mover(d, AUDIO / "2_prefijos_de_prueba" / "melodias_5s" / modelo)
    for d in sorted(esc.glob("*densas*")) if esc.exists() else []:
        modelo = d.name.split("densas_", 1)[-1]
        total += mover(d, AUDIO / "2_prefijos_de_prueba" / "muestras_densas" / modelo)

    # --- 3. generacion libre ---
    for d in (sorted(esc.glob("*desde_cero*")) if esc.exists() else []):
        total += mover(d, AUDIO / "3_generacion_libre" / "estilo_llama_ctx2048_24ep")

    # --- indice ---
    sc = scores()
    L = ["# Audio generado\n",
         "Organizado **por escenario**, porque es de donde sale el prefijo lo que",
         "determina si dos resultados son comparables entre si. Dentro de cada",
         "escenario, una carpeta por modelo.\n",
         "Cada `.wav` tiene su `.mid` (suena mejor con un piano real) y su `.png`",
         "con el piano-roll, donde la zona sombreada es el prefijo.\n"]
    for esc_name, desc in ESCENARIOS.items():
        d = AUDIO / esc_name
        if not d.exists():
            continue
        n = len(list(d.rglob("*.wav")))
        L.append("## `%s/` — %d archivos" % (esc_name, n))
        L.append("")
        L.append("%s\n" % desc)
        subs = sorted(p for p in d.iterdir() if p.is_dir())
        if subs:
            L.append("| modelo | archivos | bits/paso | gen_score |")
            L.append("|---|---|---|---|")
            for s in subs:
                k = len(list(s.rglob("*.wav")))
                b, g = sc.get(s.name, (None, None))
                L.append("| `%s` | %d | %s | %s |" % (
                    s.name, k,
                    ("%.4f" % b) if b == b and b is not None else "—",
                    ("%.1f" % g) if g == g and g is not None else "—"))
            L.append("")
    L += ["## Por donde empezar\n",
          "1. `0_referencia_corpus/` — musica real, para calibrar el oido.",
          "2. `1_continuaciones_corpus/` — el caso para el que se entreno el modelo.",
          "3. `2_prefijos_de_prueba/melodias_5s/` — compara `estilo_llama_10ep` (gen_score 17.4)",
          "   con `estilo_llama_ctx2048_24ep` (gen_score 6.6 pero MEJOR bits/paso). El de mejor",
          "   metrica de prediccion genera peor: es el resultado central del laboratorio.",
          "4. `3_generacion_libre/` — lo que el modelo compone sin ninguna pista.\n",
          "## Que esperar\n",
          "El corpus son 714 h de piano interpretado polifonico. Las cinco melodias",
          "conocidas son monofonicas y estan FUERA de esa distribucion: el modelo las",
          "continua en su propio estilo, no reconoce la cancion. El techo medido para",
          "ese escenario es 21.56, no 87.6.\n",
          "Al escuchar, juzga: mantiene la tonalidad y el registro del prefijo?",
          "respeta su densidad y textura? evita quedarse en bucle?\n"]
    (AUDIO / "LEEME.md").write_text("\n".join(L), encoding="utf-8")

    for d in antiguas:                          # carpetas del esquema anterior
        shutil.rmtree(d, ignore_errors=True)
    for obsoleta in ("escuchar", "inferencia"):
        p = ROOT / "reports" / obsoleta
        if p.exists():
            shutil.rmtree(p)

    print("reports/audio/ reconstruido: %d ficheros" % total)
    for esc_name in ESCENARIOS:
        d = AUDIO / esc_name
        if d.exists():
            print("  %-26s %3d wav" % (esc_name, len(list(d.rglob("*.wav")))))
    print("\nindice: reports/audio/LEEME.md")


if __name__ == "__main__":
    main()
