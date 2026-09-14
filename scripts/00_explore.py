"""Analisis exploratorio de train.npz -> data/processed/data_stats.json + reports/figures."""
import json, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"; OUT.mkdir(parents=True, exist_ok=True)

t0 = time.time()
z = np.load(ROOT / "train.npz")
offsets = z["offsets"].astype(np.int64)
ids = z["ids"]
step_sec = float(z["step_sec"][0]); note_min = int(z["note_min"][0]); note_max = int(z["note_max"][0])
print(f"loading rolls_flat ... ({time.time()-t0:.1f}s)", flush=True)
R = z["rolls_flat"]                      # (N,88) uint8
print(f"loaded {R.shape} in {time.time()-t0:.1f}s", flush=True)

lens = np.diff(offsets)
N, P = R.shape
per_step = R.sum(1).astype(np.int16)     # onsets simultaneos por paso
pitch_hist = R.sum(0).astype(np.int64)   # histograma de pitch
nz = per_step > 0

stats = dict(
    n_sequences=int(len(ids)), n_steps_total=int(N), n_positions=int(P),
    step_sec=step_sec, note_min=note_min, note_max=note_max,
    representation=str(z["representation"][0]),
    total_onsets=int(per_step.sum()),
    density=float(R.mean()),
    frac_empty_steps=float(1 - nz.mean()),
    mean_onsets_per_step=float(per_step.mean()),
    mean_onsets_per_active_step=float(per_step[nz].mean()),
    polyphony_hist={int(k): int(v) for k, v in zip(*np.unique(per_step, return_counts=True))},
    seq_len=dict(min=int(lens.min()), p1=int(np.percentile(lens,1)), p25=int(np.percentile(lens,25)),
                 median=int(np.median(lens)), p75=int(np.percentile(lens,75)),
                 p99=int(np.percentile(lens,99)), max=int(lens.max()), mean=float(lens.mean())),
    duration_sec=dict(median=float(np.median(lens)*step_sec), total_hours=float(N*step_sec/3600)),
    pitch_hist=pitch_hist.tolist(),
)

# --- gaps entre pasos con onset (dentro de cada secuencia) ---
gaps = []
for i in range(len(ids)):
    a, b = offsets[i], offsets[i+1]
    idx = np.flatnonzero(nz[a:b])
    if len(idx) > 1:
        gaps.append(np.diff(idx))
gaps = np.concatenate(gaps)
gu, gc = np.unique(gaps, return_counts=True)
stats["gap_hist"] = {int(k): int(v) for k, v in zip(gu[:80], gc[:80])}
stats["gap"] = dict(mean=float(gaps.mean()), median=int(np.median(gaps)),
                    p90=int(np.percentile(gaps,90)), p99=int(np.percentile(gaps,99)),
                    max=int(gaps.max()),
                    frac_le_1=float((gaps<=1).mean()), frac_le_4=float((gaps<=4).mean()),
                    frac_le_16=float((gaps<=16).mean()), frac_le_32=float((gaps<=32).mean()))
stats["n_active_steps"] = int(nz.sum())
# eventos si tokenizamos: 1 token por onset + 1 token de shift por paso activo
stats["est_tokens_note_shift"] = int(per_step.sum() + nz.sum())

# --- estructura ritmica: onsets por posicion en grilla de 16 pasos ---
pos16 = np.zeros(16, np.int64)
for i in range(min(len(ids), 3000)):
    a, b = offsets[i], offsets[i+1]
    s = per_step[a:b]
    m = (len(s)//16)*16
    pos16 += s[:m].reshape(-1,16).sum(0)
stats["onsets_by_grid16"] = pos16.tolist()

json.dump(stats, open(OUT/"data_stats.json","w"), indent=2)
print(json.dumps({k:v for k,v in stats.items() if k not in ("pitch_hist","polyphony_hist","gap_hist","onsets_by_grid16")}, indent=2))
print("polyphony:", dict(list(stats["polyphony_hist"].items())[:12]))
print("gap frac<=1,4,16,32:", stats["gap"]["frac_le_1"], stats["gap"]["frac_le_4"], stats["gap"]["frac_le_16"], stats["gap"]["frac_le_32"])
print("grid16:", pos16 / pos16.sum())
print(f"done {time.time()-t0:.1f}s")
