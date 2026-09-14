"""Figuras descriptivas del dataset -> reports/figures/."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import viz                                          # noqa: E402
from data.datasets import get_roll                  # noqa: E402

FIG = ROOT / "reports" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
out = []

out.append(viz.plot_data_overview(ROOT / "data" / "processed" / "data_stats.json",
                                  FIG / "dataset_overview.png"))

# ejemplos reales del corpus, para contrastar visualmente con lo generado
rng = np.random.default_rng(3)
rolls = []
for i in rng.choice(10604, 4, replace=False):
    r = get_roll(int(i))
    s = int(rng.integers(0, max(1, len(r) - 900)))
    rolls.append(r[s:s + 800])
out.append(viz.plot_pianoroll_grid(rolls, FIG / "corpus_examples.png",
                                   titles=["corpus: pieza %d" % i for i in range(1, 5)]))
for p in out:
    p = Path(p)
    print("%-42s %8.1f KB" % (p.name, p.stat().st_size / 1024))
