"""Fabrica de modelos: build_model(cfg) -> ARModel."""
from __future__ import annotations

from .base import ARModel, TokenARModel, FrameARModel, causal_mask   # noqa: F401

MODEL_FAMILY = dict(music_transformer="token", lstm="token", tft="token", melle="frame",
                    modern="token", deep_lstm="token", prefix_enc="token",
                    hierarchical="token", perceiver_ar="token")


def build_model(cfg):
    """Instancia el modelo indicado por cfg.model y valida su familia."""
    key = cfg.model
    if key == "music_transformer":
        from .music_transformer import MusicTransformer as M
    elif key == "lstm":
        from .lstm_baseline import LSTMBaseline as M
    elif key == "tft":
        from .tft_music import TFTMusic as M
    elif key == "melle":
        from .melle_music import MELLEMusic as M
    elif key == "modern":
        from .modern_transformer import ModernTransformer as M
    elif key == "deep_lstm":
        from .deep_lstm import DeepLSTM as M
    elif key == "prefix_enc":
        from .prefix_encoder import PrefixEncoderDecoder as M
    elif key == "hierarchical":
        from .hierarchical import HierarchicalMusicLM as M
    elif key == "perceiver_ar":
        from .perceiver_ar import PerceiverAR as M
    else:
        raise ValueError(f"modelo desconocido: {key!r}. Opciones: {sorted(MODEL_FAMILY)}")
    model = M(cfg)
    exp = MODEL_FAMILY[key]
    if getattr(model, "family", None) != exp:
        raise ValueError(f"{key}: family={model.family!r} pero se esperaba {exp!r}")
    if cfg.family != exp:
        cfg.family = exp
    return model


def family_of(model_key: str) -> str:
    return MODEL_FAMILY[model_key]
