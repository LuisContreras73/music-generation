"""Reproduce el fallo de serializacion del memmap en Windows (spawn).

Antes del arreglo: OSError [Errno 22] al serializar numpy.memmap de 566 MB
hacia el proceso worker. Con memmap perezoso + __getstate__ no viaja nada.
"""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pickle
import torch
from torch.utils.data import DataLoader
from data.datasets import TokenWindows, FrameWindows


def main():
    ok = True
    for name, DS, nb in (("TokenWindows", TokenWindows, 2), ("FrameWindows", FrameWindows, 2)):
        ds = DS("val", 1024, 64)
        blob = pickle.dumps(ds)
        print("%-14s pickle del dataset: %7.1f KB (antes seria ~566 MB en FrameWindows)"
              % (name, len(blob) / 1024))
        ok &= len(blob) < 5_000_000
        for nw in (0, 3):
            t0 = time.time()
            dl = DataLoader(ds, batch_size=8, num_workers=nw,
                            persistent_workers=nw > 0, prefetch_factor=2 if nw else None)
            got = 0
            try:
                for i, b in enumerate(dl):
                    got += int(b[0].shape[0])
                    if i + 1 >= nb:
                        break
                # segundo pase: fuerza recrear iteradores (donde fallaba el eval)
                for i, b in enumerate(dl):
                    got += int(b[0].shape[0])
                    if i + 1 >= nb:
                        break
                print("   num_workers=%d  ok, %d muestras en %.1fs" % (nw, got, time.time() - t0))
            except Exception as e:
                ok = False
                print("   num_workers=%d  FALLO: %s: %s" % (nw, type(e).__name__, str(e)[:120]))
            del dl
    print("RESULTADO: %s" % ("PASA" if ok else "FALLA"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
