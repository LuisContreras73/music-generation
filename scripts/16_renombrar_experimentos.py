"""Renombra los experimentos a nombres descriptivos, de forma segura y reversible.

    python scripts/16_renombrar_experimentos.py --dry-run    # ver que haria
    python scripts/16_renombrar_experimentos.py              # aplicarlo

Que cambia:
  * el directorio experiments/<viejo>/ -> experiments/<nuevo>/
  * el campo "name" dentro de config.json
  * el campo "name" dentro de config y de la config guardada en cada checkpoint
  * el campo "name" de logs/summary.json y logs/best.json
  * SEGUNDA FASE (--no-sync para saltarla): toda referencia al nombre viejo que
    quede escrita en docs/, reports/, scripts/, src/ y en los propios JSON de
    cada experimento (las rutas "root", "metrics_csv", "checkpoints", las notas
    y las claves con las que 05_report.py arma la tabla de ablacion). Sin esta
    fase el renombrado deja punteros rotos: el informe pierde las secciones que
    busca por nombre y los summary.json apuntan a carpetas que ya no existen.
  * los directorios de reports/external_eval/<conjunto>/<viejo>/

Por que: los nombres de trabajo (mt_base, modern_gpu, modern_max) son jerga
interna y no dicen que distingue a cada experimento. Los nuevos nombran la
ARQUITECTURA y la VARIABLE que cambia, que es lo que hace legible una tabla de
resultados sin tener que explicarla.

Deja un mapa en reports/nombres_experimentos.json para poder revertirlo.
"""
from __future__ import annotations
import argparse, json, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MAPA = {
    "mt_base":         "music_transformer",
    "mt_norel":        "ablacion_sin_atencion_relativa",
    "mt_aug":          "music_transformer_con_augmentacion",
    "mt_long":         "music_transformer_ctx2048",
    "modern_gpu":      "transformer_moderno_10ep",
    "modern_max":      "transformer_moderno_24ep",
    "modern_long_max": "transformer_moderno_ctx2048_10ep",
    "modern_long_24":  "transformer_moderno_ctx2048_24ep",
    # segunda pasada: "estilo_llama" sustenta de donde sale la receta (RoPE +
    # RMSNorm + SwiGLU, la de LLaMA/PaLM/Gemma) SIN afirmar que sea LLaMA ni que
    # use sus pesos: esta entrenado desde cero con 25.8 M parametros sobre
    # musica simbolica, no es un modelo de lenguaje reaprovechado.
    "transformer_moderno_10ep":         "estilo_llama_10ep",
    "transformer_moderno_24ep":         "estilo_llama_24ep",
    "transformer_moderno_ctx2048_10ep": "estilo_llama_ctx2048_10ep",
    "transformer_moderno_ctx2048_24ep": "estilo_llama_ctx2048_24ep",
    "lstm_base":       "lstm",
    "deep_lstm_aug":   "deep_lstm",
    "melle_base":      "melle",
    "melle_noflux":    "ablacion_melle_sin_flux",
    "tft_base":        "tft",
    "perceiver_ar":    "perceiver_ar",
}


def canonico() -> dict:
    """Cierre transitivo del mapa: nombre de trabajo -> nombre final.

    El renombrado se hizo en dos pasadas (mt_base -> ... -> music_transformer,
    modern_gpu -> transformer_moderno_10ep -> estilo_llama_10ep), asi que un
    fichero viejo puede citar cualquier eslabon de la cadena.
    """
    out = {}
    for viejo in MAPA:
        n, visto = viejo, {viejo}
        while n in MAPA and MAPA[n] != n and MAPA[n] not in visto:
            n = MAPA[n]; visto.add(n)
        if n != viejo:
            out[viejo] = n
    return out


def sincronizar_referencias(dry: bool = False) -> int:
    """Reescribe los nombres viejos que quedan citados en ficheros de texto."""
    import re
    canon = canonico()
    # de mas largo a mas corto: evita que un nombre sea prefijo de otro
    patron = re.compile(r"(?<![A-Za-z0-9_])(%s)(?![A-Za-z0-9_])"
                        % "|".join(sorted(map(re.escape, canon), key=len, reverse=True)))
    objetivos = []
    for pat in ("docs/*.md", "reports/**/*.md", "reports/**/*.json", "reports/*.csv",
                "experiments/*/config.json", "experiments/*/logs/*.json",
                "scripts/*.py", "src/**/*.py"):
        objetivos += sorted(ROOT.glob(pat))
    yo = Path(__file__).resolve()
    n_fich = 0
    for f in objetivos:
        if f.resolve() == yo:                   # este fichero ES el mapa
            continue
        try:
            t = f.read_text(encoding="utf-8")
        except Exception:
            continue
        t2 = patron.sub(lambda m: canon[m.group(1)], t)
        if t2 != t:
            n_fich += 1
            n_ref = sum(1 for _ in patron.finditer(t))
            print("  %-52s %d referencias" % (f.relative_to(ROOT).as_posix(), n_ref))
            if not dry:
                f.write_text(t2, encoding="utf-8")
    # directorios de evaluacion externa, que llevan el nombre del experimento
    ext = ROOT / "reports" / "external_eval"
    for conj in (sorted(p for p in ext.iterdir() if p.is_dir()) if ext.exists() else []):
        for d in sorted(p for p in conj.iterdir() if p.is_dir()):
            nuevo = canon.get(d.name)
            if not nuevo or (conj / nuevo).exists():
                continue
            print("  %-52s -> %s" % ((d.relative_to(ROOT)).as_posix(), nuevo))
            if not dry:
                d.rename(conj / nuevo)
            n_fich += 1
    return n_fich


def renombrar_json(p: Path, viejo: str, nuevo: str) -> bool:
    if not p.exists():
        return False
    try:
        d = json.load(open(p, encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(d, dict) or d.get("name") != viejo:
        return False
    d["name"] = nuevo
    json.dump(d, open(p, "w", encoding="utf-8"), indent=2)
    return True


def renombrar_checkpoint(p: Path, viejo: str, nuevo: str) -> bool:
    """El checkpoint guarda la config; si no se actualiza, al recargarlo el
    modelo escribiria en el directorio antiguo."""
    if not p.exists():
        return False
    import torch
    try:
        ck = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:
        print("    aviso: no se pudo abrir %s (%s)" % (p.name, type(e).__name__))
        return False
    cambiado = False
    for clave in ("config", "cfg"):
        c = ck.get(clave)
        if isinstance(c, dict) and c.get("name") == viejo:
            c["name"] = nuevo; cambiado = True
    if ck.get("name") == viejo:
        ck["name"] = nuevo; cambiado = True
    if cambiado:
        tmp = p.with_suffix(".tmp")
        torch.save(ck, tmp)
        tmp.replace(p)                       # escritura atomica
    return cambiado


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-sync", action="store_true",
                    help="no reescribir las referencias en docs/reports/scripts")
    a = ap.parse_args()

    exp_root = ROOT / "experiments"
    presentes = {d.name for d in exp_root.iterdir() if d.is_dir()}
    hechos = {}
    for viejo, nuevo in MAPA.items():
        if viejo not in presentes:
            continue
        if viejo == nuevo:
            hechos[viejo] = nuevo
            continue
        dst = exp_root / nuevo
        if dst.exists():
            print("  %-18s -> %-36s YA EXISTE, se omite" % (viejo, nuevo))
            continue
        print("  %-18s -> %s" % (viejo, nuevo))
        if a.dry_run:
            continue
        (exp_root / viejo).rename(dst)
        renombrar_json(dst / "config.json", viejo, nuevo)
        for f in ("summary.json", "best.json"):
            renombrar_json(dst / "logs" / f, viejo, nuevo)
        for ck in ("best.pt", "best_gen.pt", "last.pt"):
            renombrar_checkpoint(dst / "checkpoints" / ck, viejo, nuevo)
        hechos[viejo] = nuevo

    if not a.no_sync:
        print("\nreferencias a nombres viejos que quedaban escritas:")
        if not sincronizar_referencias(dry=a.dry_run):
            print("  (ninguna: ya estaba todo sincronizado)")

    if a.dry_run:
        print("\n(dry-run: no se ha modificado nada)")
        return

    # Se guarda el cierre transitivo COMPLETO, no solo lo hecho en esta pasada:
    # el fichero sirve para releer resultados antiguos, y una ejecucion posterior
    # (idempotente) no debe vaciarlo.
    json.dump({"mapa": canonico(),
               "aplicado_ahora": hechos,
               "nota": "nombre de trabajo -> nombre final; para revertir, invertir el mapa"},
              open(ROOT / "reports" / "nombres_experimentos.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print("\n%d experimentos renombrados" % len(hechos))
    print("mapa guardado en reports/nombres_experimentos.json")


if __name__ == "__main__":
    main()
