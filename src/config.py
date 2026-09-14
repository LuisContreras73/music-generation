"""Configuracion de experimentos (dataclass -> experiments/<name>/config.json)."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
EXP_DIR = ROOT / "experiments"
REPORTS = ROOT / "reports"


@dataclass
class Config:
    # --- identidad ---
    name: str = "exp"
    model: str = "music_transformer"      # music_transformer | lstm | tft | melle
    family: str = "token"                 # token | frame
    seed: int = 1234
    notes: str = ""

    # --- datos ---
    seq_len: int = 1024
    batch_size: int = 24
    grad_accum: int = 1
    num_workers: int = 6
    train_windows: int = 200000
    val_windows: int = 400

    # --- augmentacion (solo en train) -------------------------------------
    # Medido: con prefijos del CORPUS de 70 tokens el modelo puntua 58.4 y no
    # degenera (repeat8 0.223, el corpus real esta en 0.217); con melodias
    # MONOFONICAS de la misma longitud cae a 2.4 con repeat8 0.631. El problema
    # no es la longitud del contexto sino el DOMINIO: nunca vio texturas ralas.
    # thin_voices adelgaza los acordes conservando una nota, produciendo
    # versiones casi monofonicas del corpus, que es justo lo que falta.
    # Pipeline de datos enteramente en GPU (src/gpu_data.py): sin DataLoader,
    # sin workers, sin CPU. La augmentacion pasa de ~1400 ms/batch a 1.43 ms.
    gpu_data: bool = True
    augment: bool = False
    aug_p_transpose: float = 0.9
    aug_p_stretch: float = 0.3
    aug_stretch_lo: float = 0.8
    aug_stretch_hi: float = 2.0      # hasta 2x: simula melodias lentas
    aug_p_thin: float = 0.5          # mas alto que el defecto: es la clave aqui
    aug_thin_lo: float = 0.3
    aug_thin_hi: float = 1.0
    aug_p_jitter: float = 0.1

    # --- arquitectura (subset segun modelo) ---
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    attn_dropout: float = 0.0     # dropout sobre las probabilidades de atencion
    rel_attn: bool = True
    max_rel_dist: int = 512
    hidden: int = 1024
    latent_dim: int = 64
    flux_weight: float = 0.1
    kl_weight: float = 0.001
    tie_weights: bool = True
    # Perceiver AR: numero de latentes. DEBE ser >= seq_len o la mitad de las
    # posiciones no tienen prediccion valida y contaminan la perdida (medido:
    # con 512 latentes y seq_len 1024 el bpt sale 3.52 en vez de ~1.6).
    n_latents: int = 1024

    # --- optimizacion ---
    steps: int = 12000
    lr: float = 0.0003
    min_lr_frac: float = 0.05
    warmup: int = 600
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    amp: str = "bf16"                     # bf16 | fp16 | off
    compile_model: bool = False
    label_smoothing: float = 0.0

    # --- evaluacion / logging ---
    eval_every: int = 500
    log_every: int = 50
    gen_every: int = 2000
    n_gen_samples: int = 16
    gen_steps: int = 800
    gen_prime_steps: int = 200
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 0.95

    def dir(self) -> Path:
        return EXP_DIR / self.name

    def save(self) -> None:
        d = self.dir(); d.mkdir(parents=True, exist_ok=True)
        json.dump(asdict(self), open(d / "config.json", "w"), indent=2)

    @staticmethod
    def load(p) -> "Config":
        d = json.load(open(p))
        return Config(**{k: v for k, v in d.items() if k in Config.__dataclass_fields__})

    def subdirs(self) -> dict:
        d = self.dir()
        out = dict(root=d, logs=d / "logs", ckpt=d / "checkpoints",
                   figures=d / "figures", gen=d / "generations")
        for p in out.values():
            p.mkdir(parents=True, exist_ok=True)
        return out
