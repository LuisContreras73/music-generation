"""Verificacion por gradiente de causalidad estricta.

Para un modelo AR valido:  d logits[:, t] / d input[:, t'] == 0  para todo t' > t.
Se comprueba retropropagando desde una posicion intermedia y midiendo la norma
del gradiente sobre las entradas futuras.
"""
from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def check_token_causality(model, L: int = 32, V: int = 155, t: int | None = None,
                          device: str = "cpu", verbose: bool = True) -> bool:
    """Usa embeddings one-hot diferenciables si el modelo lo permite; si no,
    perturba la entrada discreta y comprueba que los logits<=t no cambian."""
    model = model.to(device).eval()
    t = L // 2 if t is None else t
    torch.manual_seed(0)
    x = torch.randint(3, V, (2, L), device=device)
    with torch.no_grad():
        base = model(x)
    x2 = x.clone()
    x2[:, t + 1:] = torch.randint(3, V, (2, L - t - 1), device=device)
    with torch.no_grad():
        pert = model(x2)
    d_past = (base[:, : t + 1] - pert[:, : t + 1]).abs().max().item()
    d_fut = (base[:, t + 1:] - pert[:, t + 1:]).abs().max().item()
    ok = d_past < 1e-4
    if verbose:
        print(f"  [token] max|delta logits pasado|={d_past:.3e} (debe ser ~0)  "
              f"futuro={d_fut:.3e} (debe ser >0)   -> {'OK' if ok else 'FUGA CAUSAL'}")
    if d_fut < 1e-8 and verbose:
        print("  AVISO: perturbar el futuro no cambio nada; el modelo puede estar ignorando la entrada")
    return ok


def check_frame_causality(model, T: int = 32, P: int = 88, t: int | None = None,
                          device: str = "cpu", verbose: bool = True) -> bool:
    model = model.to(device).eval()
    t = T // 2 if t is None else t
    torch.manual_seed(0)
    x = (torch.rand(2, T, P, device=device) < 0.05).float().requires_grad_(True)
    out = model(x)
    logits = out["logits"] if isinstance(out, dict) else out
    logits[:, t].sum().backward()
    g = x.grad.abs()
    g_fut = g[:, t + 1:].max().item()
    g_past = g[:, : t + 1].max().item()
    ok = g_fut < 1e-8
    if verbose:
        print(f"  [frame] max|grad futuro|={g_fut:.3e} (debe ser 0)  "
              f"pasado={g_past:.3e} (debe ser >0)  -> {'OK' if ok else 'FUGA CAUSAL'}")
    return ok


def check_shapes_token(model, L: int = 32, V: int = 155, device: str = "cpu") -> bool:
    x = torch.randint(3, V, (2, L), device=device)
    y = model.to(device)(x)
    ok = tuple(y.shape) == (2, L, V) and torch.isfinite(y).all().item()
    print(f"  shapes token: {tuple(y.shape)} esperado (2,{L},{V}) finito={torch.isfinite(y).all().item()} -> {'OK' if ok else 'FALLO'}")
    return ok


def check_shapes_frame(model, T: int = 32, P: int = 88, device: str = "cpu") -> bool:
    x = (torch.rand(2, T, P, device=device) < 0.05).float()
    out = model.to(device)(x)
    assert isinstance(out, dict) and "logits" in out, "FrameARModel.forward debe devolver dict con 'logits'"
    lg = out["logits"]
    ok = tuple(lg.shape) == (2, T, P) and torch.isfinite(lg).all().item()
    print(f"  shapes frame: {tuple(lg.shape)} esperado (2,{T},{P}) -> {'OK' if ok else 'FALLO'}")
    return ok
