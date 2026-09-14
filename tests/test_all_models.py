"""Verificacion final de los 4 modelos: contrato, causalidad y determinismo."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import torch
from config import Config
from models import build_model, MODEL_FAMILY
from tests.test_causality import (check_token_causality, check_shapes_token,
                                  check_frame_causality, check_shapes_frame)

CFGS = dict(
    music_transformer=dict(d_model=256, n_layers=4, n_heads=4, d_ff=512, rel_attn=True),
    lstm=dict(d_model=256, hidden=256, n_layers=2),
    tft=dict(d_model=256, n_layers=4, n_heads=4, d_ff=512, hidden=256),
    melle=dict(d_model=256, n_layers=3, n_heads=4, d_ff=512, latent_dim=32),
)
ok_all = True
for key, kw in CFGS.items():
    print("=" * 70)
    cfg = Config(name="t", model=key, family=MODEL_FAMILY[key], seq_len=128, **kw)
    m = build_model(cfg)
    print("%s -> %s" % (key, m.param_report()))
    if cfg.family == "token":
        ok = check_shapes_token(m, L=48) and check_token_causality(m, L=48)
        # comprobacion mas estricta: perturbar SOLO x[:,t] y exigir delta 0 antes de t
        m.eval()
        x = torch.randint(3, 155, (2, 48))
        with torch.no_grad():
            base = m(x)
        for t in (0, 1, 11, 47):
            x2 = x.clone(); x2[:, t] = (x2[:, t] + 7) % 152 + 3
            with torch.no_grad():
                pert = m(x2)
            d = (base[:, :t] - pert[:, :t]).abs().max().item() if t > 0 else 0.0
            dt = (base[:, t] - pert[:, t]).abs().max().item()
            good = d == 0.0
            ok &= good
            print("    t=%2d  max|delta antes de t|=%.3e (exige 0)  |delta en t|=%.3e  %s"
                  % (t, d, dt, "OK" if good else "FALLO"))
    else:
        ok = check_shapes_frame(m, T=48) and check_frame_causality(m, T=48)
        m.eval()
        x = (torch.rand(2, 48, 88) < 0.05).float()
        with torch.no_grad():
            a = m(x)["logits"]; b = m(x)["logits"]
        det = torch.equal(a, b)
        m.train()
        with torch.no_grad():
            c = m(x)["logits"]; d = m(x)["logits"]
        sto = not torch.equal(c, d)
        print("    eval determinista: %s | train estocastico (muestreo latente): %s" % (det, sto))
        ok &= det and sto
    # entradas extremas
    if cfg.family == "token":
        for L in (1, 2, 128):
            y = m(torch.randint(3, 155, (1, L)))
            fin = torch.isfinite(y).all().item()
            ok &= fin and tuple(y.shape) == (1, L, 155)
        y = m(torch.zeros(1, 16, dtype=torch.long))          # todo PAD
        ok &= torch.isfinite(y).all().item()
        print("    L=1,2,128 y entrada todo-PAD: finito y con la forma correcta")
    else:
        y = m(torch.zeros(1, 16, 88))["logits"]              # todo ceros
        ok &= torch.isfinite(y).all().item()
        print("    entrada todo-ceros: finito")
    ok_all &= ok
    print("  %s: %s" % (key, "PASA" if ok else "FALLA"))
print("=" * 70)
print("RESULTADO GLOBAL: %s" % ("TODOS LOS MODELOS PASAN" if ok_all else "HAY FALLOS"))
sys.exit(0 if ok_all else 1)
