"""Bucle de entrenamiento unificado para las dos familias de modelos.

Uso:
    python src/train.py --name music_transformer --model music_transformer --steps 12000
    python src/train.py --config experiments/music_transformer/config.json --resume

Cada eval_every pasos se calcula la verosimilitud de validacion (bits/paso) y
cada gen_every pasos se GENERAN continuaciones reales, se puntuan con gen_score
frente al corpus y se actualizan checkpoints, CSV y figuras. De este modo
experiments/<name>/ y reports/best/ reflejan siempre el mejor estado conocido.

Nota metodologica: en la familia frame se optimiza la BCE PURA (sin pos_weight).
Solo asi los bits/paso son la verosimilitud real del piano-roll y por tanto
comparables con los modelos de eventos. El muestreo Bernoulli en generacion
recupera la densidad correcta sin necesidad de reponderar la perdida.
"""
from __future__ import annotations
import argparse, math, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config                                         # noqa: E402
from data.datasets import TokenWindows, FrameWindows              # noqa: E402
from data.tokenizer import PAD                                    # noqa: E402
from models import build_model                                    # noqa: E402
import evaluate as ev                                             # noqa: E402
import generate as gen                                            # noqa: E402
import prompts as pr                                              # noqa: E402

NAN = float("nan")


# --------------------------------------------------------------------------- utils
def set_seed(s: int):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def infinite(loader):
    while True:
        for b in loader:
            yield b


def lr_at(step: int, cfg: Config) -> float:
    """Warmup lineal + decaimiento coseno hasta min_lr_frac * lr."""
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / max(cfg.warmup, 1)
    p = (step - cfg.warmup) / max(cfg.steps - cfg.warmup, 1)
    p = min(max(p, 0.0), 1.0)
    return cfg.lr * (cfg.min_lr_frac + (1 - cfg.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * p)))


def param_groups(model, wd: float):
    """Sin weight decay en bias, normalizaciones y embeddings."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        flat = p.ndim <= 1 or n.endswith(".bias") or "emb" in n.lower()
        (no_decay if flat else decay).append(p)
    return [dict(params=decay, weight_decay=wd), dict(params=no_decay, weight_decay=0.0)]


def amp_ctx(cfg: Config):
    if cfg.amp == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if cfg.amp == "fp16":
        return torch.autocast("cuda", dtype=torch.float16)
    return torch.autocast("cuda", enabled=False)


# --------------------------------------------------------------------------- perdidas
def token_loss(model, batch, cfg, device):
    x, y = batch
    x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
    with amp_ctx(cfg):
        logits = model(x)
    loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1),
                           ignore_index=PAD, label_smoothing=cfg.label_smoothing)
    return loss, {}


def frame_loss(model, batch, cfg, device):
    x, y, mask = [t.to(device, non_blocking=True) for t in batch]
    with amp_ctx(cfg):
        out = model(x)
    if hasattr(model, "loss_terms"):
        d = model.loss_terms(out, y, mask)
        return d["loss"], {k: float(v) for k, v in d.items() if k != "loss"}
    logits = (out["logits"] if isinstance(out, dict) else out).float()
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none").sum(-1)
    loss = (bce * mask).sum() / mask.sum().clamp(min=1)
    return loss, {}


# --------------------------------------------------------------------------- generacion
def run_generation(model, cfg, device, step, dirs, save_figs=True):
    """Genera continuaciones, las puntua contra el corpus y guarda npz + PNG."""
    t0 = time.time()
    rolls, prime_steps_list, gram, gram_mass = [], [], 0.0, 0.0
    if cfg.family == "token":
        px, pst, srcs = pr.make_token_prompts(cfg.n_gen_samples, cfg.gen_prime_steps,
                                              seed=1000 + step)
        res = gen.sample_tokens(model, px, cfg.gen_steps, seq_len=cfg.seq_len,
                                temperature=cfg.temperature, top_k=cfg.top_k,
                                top_p=cfg.top_p, device=device)
        gram = res["grammar_violation_rate"]
        gram_mass = res["grammar_prob_mass"]
        for b in range(len(res["new"])):
            r = gen.tokens_to_roll(res["new"][b])
            if len(r) < cfg.gen_steps:
                r = np.pad(r, ((0, cfg.gen_steps - len(r)), (0, 0)))
            rolls.append(r[: cfg.gen_steps])
            prime_steps_list.append(int(pst[b]))
    else:
        px, srcs = pr.make_frame_prompts(cfg.n_gen_samples, cfg.gen_prime_steps,
                                         seed=1000 + step)
        res = gen.sample_frames(model, px, cfg.gen_steps, seq_len=cfg.seq_len,
                                temperature=cfg.temperature, device=device)
        rolls = [r.astype(np.uint8) for r in res["new"]]
        prime_steps_list = [cfg.gen_prime_steps] * len(rolls)

    m, agg, ref = ev.evaluate_generations(rolls)
    m["gen_grammar_violation"] = gram
    m["gen_grammar_prob_mass"] = gram_mass
    m["gen_seconds"] = time.time() - t0

    np.savez_compressed(dirs["gen"] / ("gen_step%07d.npz" % step),
                        rolls=np.stack(rolls).astype(np.uint8),
                        srcs=np.asarray(srcs), prime_steps=np.asarray(prime_steps_list))
    if save_figs:
        try:
            import viz
            viz.plot_pianoroll_grid(rolls[:4], dirs["figures"] / "generation_grid.png",
                                    titles=["muestra %d" % (i + 1) for i in range(4)])
            viz.plot_distribution_comparison(agg, ref, dirs["figures"] / "distributions.png",
                                             title="%s paso %d" % (cfg.name, step))
            viz.plot_pianoroll(rolls[0], dirs["figures"] / "generation_best.png",
                               title="%s paso %d (gen_score %.1f)" % (cfg.name, step, m["gen_score"]))
        except Exception as e:                       # viz nunca debe tumbar el entrenamiento
            print("  [viz] aviso generacion: %s: %s" % (type(e).__name__, e))
    return m


# --------------------------------------------------------------------------- main
def train(cfg: Config, resume: bool = False):
    import registry as reg
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dirs = cfg.subdirs(); cfg.save()
    logger = reg.MetricLogger(dirs["logs"] / "metrics.csv")

    model = build_model(cfg).to(device)
    n_par = model.n_params()
    print("== %s | %s | device=%s | family=%s" % (cfg.name, model.param_report(), device, cfg.family))
    print("   referencia trivial i.i.d.: %.3f bits/paso" % ev.marginal_bpt())

    # --- ruta rapida: datos y augmentacion en GPU, sin workers ----------------
    if getattr(cfg, "gpu_data", False) and cfg.family == "token":
        from gpu_data import GPUTokenStream
        aug_cfg = dict(p_transpose=cfg.aug_p_transpose, max_semitones=6,
                       p_thin=cfg.aug_p_thin, thin_lo=cfg.aug_thin_lo,
                       thin_hi=cfg.aug_thin_hi) if cfg.augment else None
        tr_stream = GPUTokenStream("train", cfg.seq_len, cfg.batch_size, device=device,
                                   seed=cfg.seed, augment=aug_cfg)
        va_stream = GPUTokenStream("val", cfg.seq_len, cfg.batch_size, device=device,
                                   max_windows=cfg.val_windows)
        te_stream = GPUTokenStream("test", cfg.seq_len, cfg.batch_size, device=device,
                                   max_windows=cfg.val_windows)
        print("   datos en GPU: %.1f M tokens residentes, %d piezas%s"
              % (tr_stream.n_tokens / 1e6, tr_stream.n_pieces,
                 ", augmentacion GPU ACTIVA" if aug_cfg else ""))
        return _train_gpu(cfg, model, n_par, tr_stream, va_stream, te_stream,
                          device, dirs, logger, reg, resume)

    DS = TokenWindows if cfg.family == "token" else FrameWindows
    aug = None
    if getattr(cfg, "augment", False) and cfg.family == "token":
        from augment import AugmentConfig
        aug = AugmentConfig(p_transpose=cfg.aug_p_transpose, p_stretch=cfg.aug_p_stretch,
                            stretch_lo=cfg.aug_stretch_lo, stretch_hi=cfg.aug_stretch_hi,
                            p_thin=cfg.aug_p_thin, thin_lo=cfg.aug_thin_lo,
                            thin_hi=cfg.aug_thin_hi, p_jitter=cfg.aug_p_jitter)
        print("   augmentacion ACTIVA: transponer %.0f%%, estirar %.0f%%, adelgazar %.0f%% "
              "(p_drop %.1f-%.1f), jitter %.0f%%"
              % (100 * aug.p_transpose, 100 * aug.p_stretch, 100 * aug.p_thin,
                 aug.thin_lo, aug.thin_hi, 100 * aug.p_jitter))
    tr = (DS("train", cfg.seq_len, cfg.train_windows, seed=cfg.seed, augment_cfg=aug)
          if aug is not None else DS("train", cfg.seq_len, cfg.train_windows, seed=cfg.seed))
    va = DS("val", cfg.seq_len, cfg.val_windows)
    dl_tr = DataLoader(tr, batch_size=cfg.batch_size, shuffle=False, drop_last=True,
                       num_workers=cfg.num_workers, pin_memory=True,
                       persistent_workers=cfg.num_workers > 0,
                       prefetch_factor=4 if cfg.num_workers > 0 else None)
    # Evaluacion SIN workers a proposito: con spawn en Windows los workers del
    # DataLoader mueren al cerrar el proceso y tumban la evaluacion final
    # (_queue.Empty, y en un caso un segmentation fault). Leer el memmap desde
    # el proceso principal es rapido y elimina esa clase entera de fallos.
    dl_va = DataLoader(va, batch_size=cfg.batch_size, shuffle=False,
                       num_workers=0, pin_memory=True)
    print("   ventanas: train=%d val=%d | seq_len=%d batch=%d" %
          (len(tr), len(va), cfg.seq_len, cfg.batch_size))

    opt = torch.optim.AdamW(param_groups(model, cfg.weight_decay), lr=cfg.lr,
                            betas=(0.9, 0.95), eps=1e-8)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp == "fp16")
    loss_fn = token_loss if cfg.family == "token" else frame_loss
    if cfg.compile_model:
        model = torch.compile(model)

    start = 0
    ck_last = dirs["ckpt"] / "last.pt"
    if resume and ck_last.exists():
        st = reg.load_checkpoint(ck_last, map_location=device)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["optimizer"])
        start = int(st["step"])
        print("   reanudado desde el paso %d" % start)

    it = infinite(dl_tr)
    t_start = time.time(); tokens_seen = 0; run_loss = []
    for step in range(start, cfg.steps):
        model.train()
        lr = lr_at(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        acc_loss = 0.0; extra = {}
        for _ in range(cfg.grad_accum):
            batch = next(it)
            loss, ex = loss_fn(model, batch, cfg, device)
            scaled = scaler.scale(loss / cfg.grad_accum) if scaler.is_enabled() else loss / cfg.grad_accum
            scaled.backward()
            acc_loss += float(loss) / cfg.grad_accum; extra = ex
            tokens_seen += cfg.batch_size * cfg.seq_len
        if scaler.is_enabled():
            scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        if scaler.is_enabled():
            scaler.step(opt); scaler.update()
        else:
            opt.step()
        run_loss.append(acc_loss)

        if (step + 1) % cfg.log_every == 0:
            el = time.time() - t_start
            sps = (step + 1 - start) / max(el, 1e-9)
            row = dict(step=step + 1, split="train",
                       train_loss=float(np.mean(run_loss[-cfg.log_every:])),
                       lr=lr, grad_norm=float(gnorm), tokens_seen=tokens_seen,
                       time_s=el, steps_per_s=sps, tokens_per_s=tokens_seen / max(el, 1e-9))
            row.update(extra)
            logger.append(row)
            print("  paso %6d/%d loss %.4f lr %.2e |g| %.2f %.2f it/s" %
                  (step + 1, cfg.steps, row["train_loss"], lr, float(gnorm), sps), flush=True)

        do_eval = (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps
        do_gen = (step + 1) % cfg.gen_every == 0 or step + 1 == cfg.steps
        if do_eval or do_gen:
            m = dict(step=step + 1, split="eval", tokens_seen=tokens_seen,
                     time_s=time.time() - t_start, lr=lr,
                     train_loss=float(np.mean(run_loss[-cfg.log_every:])))
            if do_eval:
                ef = ev.eval_token_model if cfg.family == "token" else ev.eval_frame_model
                m.update(ef(model, dl_va, device=device, amp=cfg.amp))
                m["val_loss"] = m.get("val_nll", NAN)
                print("    [eval] paso %d  val_bpt %.4f bits/paso  (trivial %.3f)" %
                      (step + 1, m["val_bpt"], ev.marginal_bpt()), flush=True)
            if do_gen:
                m.update(run_generation(model, cfg, device, step + 1, dirs))
                print("    [gen ] gen_score %.2f | densidad %.4f vs ref %.4f | prob. de nota imposible %.2e" %
                      (m["gen_score"], m.get("gen_density", NAN), m.get("ref_density", NAN),
                       m.get("gen_grammar_prob_mass", 0.0)), flush=True)
            logger.append(m)
            # best.pt / best_gen.pt van SIN estado del optimizador (100 MB en vez
            # de 300 MB): solo se usan para evaluar y generar. El estado de Adam
            # solo hace falta en last.pt, que es el checkpoint de reanudacion.
            reg.update_best(cfg, step + 1, m, model, optimizer=None)
            reg.save_checkpoint(dirs["ckpt"] / "last.pt", model, opt, cfg, step + 1, m)
            try:
                import viz
                viz.plot_training_curves(dirs["logs"] / "metrics.csv",
                                         dirs["figures"] / "training_curves.png",
                                         title="%s (%s, %.1f M par.)" % (cfg.name, cfg.model, n_par / 1e6))
            except Exception as e:
                print("  [viz] aviso curvas: %s: %s" % (type(e).__name__, e))

    # ---------------- cierre: test final + resumen + leaderboard ----------------
    total_h = (time.time() - t_start) / 3600
    # Si el run se reanudo, el bucle no se ejecuto y los contadores de esta
    # ejecucion valen ~0. El metrics.csv conserva el historial completo, asi que
    # es la fuente de verdad para el tiempo y los tokens acumulados.
    try:
        import csv as _csv
        with open(dirs["logs"] / "metrics.csv", newline="") as _f:
            _rows = list(_csv.DictReader(_f))
        _t = max((float(r["time_s"]) for r in _rows if r.get("time_s")), default=0.0)
        _tok = max((float(r["tokens_seen"]) for r in _rows if r.get("tokens_seen")), default=0.0)
        total_h = max(total_h, _t / 3600)
        tokens_seen = max(tokens_seen, int(_tok))
    except Exception as _e:
        print("  [aviso] no se pudo leer el historial del CSV: %s" % _e)
    ck_best = dirs["ckpt"] / "best.pt"
    if ck_best.exists():
        st = reg.load_checkpoint(ck_best, map_location=device)
        model.load_state_dict(st["model"])
        print("   cargado best.pt (paso %s) para la evaluacion final" % st["step"])
    te = DS("test", cfg.seq_len, cfg.val_windows)
    dl_te = DataLoader(te, batch_size=cfg.batch_size, num_workers=0, pin_memory=True)
    ef = ev.eval_token_model if cfg.family == "token" else ev.eval_frame_model
    tm = {"test_" + k.replace("val_", ""): v
          for k, v in ef(model, dl_te, device=device, amp=cfg.amp).items()}
    fm = run_generation(model, cfg, device, cfg.steps, dirs)
    row = dict(step=cfg.steps, split="test")
    row.update(tm); row.update(fm)
    logger.append(row)
    summary_extra = dict(n_params=n_par, train_hours=total_h, tokens_seen=tokens_seen)
    summary_extra.update(tm)
    summary_extra.update({"final_" + k: v for k, v in fm.items()})
    reg.write_summary(cfg, summary_extra)
    out = reg.refresh_all()
    print("== FIN %s: test_bpt %.4f  gen_score %.2f  en %.2f h" %
          (cfg.name, tm.get("test_bpt", NAN), fm["gen_score"], total_h))
    return out


def build_cfg_from_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--resume", action="store_true")
    base = Config()
    for f in Config.__dataclass_fields__:
        cur = getattr(base, f)
        if isinstance(cur, bool):
            ap.add_argument("--" + f, type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
        else:
            ap.add_argument("--" + f, type=type(cur), default=None)
    a = ap.parse_args()
    cfg = Config.load(a.config) if a.config else Config()
    for f in Config.__dataclass_fields__:
        v = getattr(a, f, None)
        if v is not None:
            setattr(cfg, f, v)
    from models import family_of
    cfg.family = family_of(cfg.model)
    return cfg, a.resume




# --------------------------------------------------------------------------- ruta GPU
def _train_gpu(cfg, model, n_par, tr, va, te, device, dirs, logger, reg, resume):
    """Bucle de entrenamiento sobre GPUTokenStream: sin DataLoader ni workers.

    Identico en metricas y checkpoints al bucle general; cambia solo de donde
    vienen los batches. Se separa para no complicar el camino frame-level, que
    sigue necesitando el DataLoader.
    """
    opt = torch.optim.AdamW(param_groups(model, cfg.weight_decay), lr=cfg.lr,
                            betas=(0.9, 0.95), eps=1e-8)
    if cfg.compile_model:
        model = torch.compile(model)

    start = 0
    ck_last = dirs["ckpt"] / "last.pt"
    if resume and ck_last.exists():
        st = reg.load_checkpoint(ck_last, map_location=device)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["optimizer"])
        start = int(st["step"])
        print("   reanudado desde el paso %d" % start)

    @torch.no_grad()
    def eval_stream(stream):
        """bits/paso, perplejidad y accuracy sobre las ventanas fijas del split."""
        model.eval()
        nll = 0.0; ntok = 0; nstep = 0; corr = 0
        for x, y in stream.eval_batches():
            with amp_ctx(cfg):
                logits = model(x)
            logits = logits.float()
            m = y != PAD
            if not bool(m.any()):
                continue
            ls = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                                 reduction="none").view_as(y)
            nll += float((ls * m).sum()); ntok += int(m.sum())
            is_sh = (y >= 91) & m
            nstep += int(((y - 91 + 1) * is_sh).sum())
            corr += int(((logits.argmax(-1) == y) & m).sum())
        if not ntok:
            return dict(val_bpt=NAN)
        nt = nll / ntok
        return dict(val_nll=nt, val_ppl=float(math.exp(min(nt, 30))),
                    val_bpt=float(nll / math.log(2.0) / max(nstep, 1)),
                    val_token_acc=corr / ntok, val_n_tokens=ntok, val_n_steps=nstep)

    t_start = time.time(); tokens_seen = 0; run_loss = []
    for step in range(start, cfg.steps):
        model.train()
        lr = lr_at(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        acc = 0.0
        for _ in range(cfg.grad_accum):
            x, y = tr.train_batch()
            with amp_ctx(cfg):
                logits = model(x)
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                   y.reshape(-1), ignore_index=PAD,
                                   label_smoothing=cfg.label_smoothing)
            (loss / cfg.grad_accum).backward()
            acc += float(loss) / cfg.grad_accum
            tokens_seen += cfg.batch_size * cfg.seq_len
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        run_loss.append(acc)

        if (step + 1) % cfg.log_every == 0:
            el = time.time() - t_start
            sps = (step + 1 - start) / max(el, 1e-9)
            logger.append(dict(step=step + 1, split="train",
                               train_loss=float(np.mean(run_loss[-cfg.log_every:])),
                               lr=lr, grad_norm=float(gn), tokens_seen=tokens_seen,
                               time_s=el, steps_per_s=sps,
                               tokens_per_s=tokens_seen / max(el, 1e-9)))
            print("  paso %6d/%d loss %.4f lr %.2e |g| %.2f %.2f it/s" %
                  (step + 1, cfg.steps, float(np.mean(run_loss[-cfg.log_every:])),
                   lr, float(gn), sps), flush=True)

        do_eval = (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps
        do_gen = (step + 1) % cfg.gen_every == 0 or step + 1 == cfg.steps
        if do_eval or do_gen:
            m = dict(step=step + 1, split="eval", tokens_seen=tokens_seen,
                     time_s=time.time() - t_start, lr=lr,
                     train_loss=float(np.mean(run_loss[-cfg.log_every:])))
            if do_eval:
                m.update(eval_stream(va))
                m["val_loss"] = m.get("val_nll", NAN)
                print("    [eval] paso %d  val_bpt %.4f bits/paso  (trivial %.3f)" %
                      (step + 1, m["val_bpt"], ev.marginal_bpt()), flush=True)
            if do_gen:
                m.update(run_generation(model, cfg, device, step + 1, dirs))
                print("    [gen ] gen_score %.2f | densidad %.4f vs ref %.4f | "
                      "prob. de nota imposible %.2e" %
                      (m["gen_score"], m.get("gen_density", NAN),
                       m.get("ref_density", NAN),
                       m.get("gen_grammar_prob_mass", 0.0)), flush=True)
            logger.append(m)
            reg.update_best(cfg, step + 1, m, model, optimizer=None)
            reg.save_checkpoint(dirs["ckpt"] / "last.pt", model, opt, cfg, step + 1, m)
            try:
                import viz
                viz.plot_training_curves(dirs["logs"] / "metrics.csv",
                                         dirs["figures"] / "training_curves.png",
                                         title="%s (%s, %.1f M par.)"
                                               % (cfg.name, cfg.model, n_par / 1e6))
            except Exception as e:
                print("  [viz] aviso curvas: %s: %s" % (type(e).__name__, e))

    total_h = (time.time() - t_start) / 3600
    ck_best = dirs["ckpt"] / "best.pt"
    if ck_best.exists():
        st = reg.load_checkpoint(ck_best, map_location=device)
        model.load_state_dict(st["model"])
        print("   cargado best.pt (paso %s) para la evaluacion final" % st["step"])
    tm = {"test_" + k.replace("val_", ""): v for k, v in eval_stream(te).items()}
    fm = run_generation(model, cfg, device, cfg.steps, dirs)
    row = dict(step=cfg.steps, split="test"); row.update(tm); row.update(fm)
    logger.append(row)
    extra = dict(n_params=n_par, train_hours=total_h, tokens_seen=tokens_seen)
    extra.update(tm); extra.update({"final_" + k: v for k, v in fm.items()})
    reg.write_summary(cfg, extra)
    out = reg.refresh_all()
    print("== FIN %s: test_bpt %.4f  gen_score %.2f  en %.2f h" %
          (cfg.name, tm.get("test_bpt", NAN), fm["gen_score"], total_h))
    return out


if __name__ == "__main__":
    _cfg, _res = build_cfg_from_args()
    train(_cfg, resume=_res)
