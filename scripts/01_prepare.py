"""train.npz -> corpus tokenizado + rolls bit-empaquetados + splits.

Salidas en data/processed/:
  tokens.bin        uint8 plano, todas las secuencias concatenadas (BOS..EOS)
  tokens_meta.npz   tok_offsets, roll_lens, ids
  rolls_packed.bin  uint8 memmap [N_steps, 11]  (np.packbits sobre las 88 notas)
  splits.json       indices de secuencia train/val/test (seed fija)
"""
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data.tokenizer import encode_roll_fast, decode_tokens, VOCAB_SIZE   # noqa: E402

OUT = ROOT / "data" / "processed"; OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()

z = np.load(ROOT / "train.npz")
offsets = z["offsets"].astype(np.int64); ids = z["ids"]
R = z["rolls_flat"]
n_seq = len(ids)
print(f"[{time.time()-t0:5.1f}s] rolls_flat {R.shape} cargado", flush=True)

# ---------- 1. rolls bit-empaquetados (para modelos frame-level) ----------
packed_path = OUT / "rolls_packed.bin"
if not packed_path.exists():
    P = np.packbits(R, axis=1)                      # [N, 11]
    P.tofile(packed_path)
    print(f"[{time.time()-t0:5.1f}s] rolls_packed.bin {P.shape} "
          f"({packed_path.stat().st_size/1e6:.0f} MB)", flush=True)
    del P

# ---------- 2. tokenizacion ----------
tok_chunks, tok_lens, roll_lens = [], np.zeros(n_seq, np.int64), np.zeros(n_seq, np.int64)
for i in range(n_seq):
    a, b = offsets[i], offsets[i + 1]
    tk = encode_roll_fast(R[a:b])
    tok_chunks.append(tk); tok_lens[i] = len(tk); roll_lens[i] = b - a
    if i % 2000 == 0:
        print(f"[{time.time()-t0:5.1f}s] tokenizando {i}/{n_seq}", flush=True)
tokens = np.concatenate(tok_chunks); del tok_chunks
tok_offsets = np.r_[0, np.cumsum(tok_lens)]
tokens.tofile(OUT / "tokens.bin")
np.savez(OUT / "tokens_meta.npz", tok_offsets=tok_offsets, roll_lens=roll_lens,
         tok_lens=tok_lens, ids=ids)
print(f"[{time.time()-t0:5.1f}s] tokens.bin {len(tokens):,} tokens "
      f"({len(tokens)/1e6:.1f} M, {(OUT/'tokens.bin').stat().st_size/1e6:.0f} MB)", flush=True)

# ---------- 3. verificacion de reversibilidad sobre datos REALES ----------
rng = np.random.default_rng(0)
bad = 0
for i in rng.choice(n_seq, 40, replace=False):
    a, b = offsets[i], offsets[i + 1]
    orig = R[a:b]
    rec = decode_tokens(tokens[tok_offsets[i]:tok_offsets[i + 1]], max_steps=b - a)
    if rec.shape != orig.shape or not (rec == orig).all():
        bad += 1; print("  MISMATCH seq", i, orig.shape, rec.shape)
print(f"[{time.time()-t0:5.1f}s] round-trip exacto en 40 secuencias reales: {bad == 0}", flush=True)

# ---------- 4. splits por secuencia (sin fuga de ventanas) ----------
perm = np.random.default_rng(1234).permutation(n_seq)
n_val = n_test = int(round(0.05 * n_seq))
splits = dict(test=sorted(perm[:n_test].tolist()),
              val=sorted(perm[n_test:n_test + n_val].tolist()),
              train=sorted(perm[n_test + n_val:].tolist()))
json.dump({k: v for k, v in splits.items()}, open(OUT / "splits.json", "w"))

tps = {k: int(tok_lens[v].sum()) for k, v in splits.items()}
steps = {k: int(roll_lens[v].sum()) for k, v in splits.items()}
summary = dict(vocab_size=VOCAB_SIZE, n_seq=n_seq, n_tokens_total=int(len(tokens)),
               tokens_per_split=tps, steps_per_split=steps,
               n_seq_per_split={k: len(v) for k, v in splits.items()},
               tokens_per_step=float(len(tokens) / roll_lens.sum()))
json.dump(summary, open(OUT / "corpus_summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
u, c = np.unique(tokens, return_counts=True)
np.save(OUT / "token_freq.npy", np.bincount(tokens, minlength=VOCAB_SIZE))
print(f"tokens distintos usados: {len(u)}/{VOCAB_SIZE}")
print(f"[{time.time()-t0:5.1f}s] LISTO")
