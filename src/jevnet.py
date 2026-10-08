"""Jevified Gemma 4 12B: a decision model that does less work than the chat model it came from.

  exit=L   load only the first L decoder layers and read the answer through the model's own final norm and
           tied letter rows at layer L; layers L+1..48 are never loaded or run
  drop=M   from layer M+1 on, no position may attend to an image token. For the answer position this is
           identical to deleting the image tokens from the sequence (their states can no longer reach any
           text token), so accuracy can be measured here and the fast path deletes them for real.
"""
import torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from common import MID


class State:
    """Per-forward token-type info; common.score() fills is_image and valid."""
    is_image = None
    valid = None


def load_truncated(exit_layer, mid=MID, attn="eager", **kw):
    cfg = AutoConfig.from_pretrained(mid)
    t = cfg.text_config
    if exit_layer < t.num_hidden_layers:
        t.num_hidden_layers = exit_layer
        t.layer_types = t.layer_types[:exit_layer]
    proc = AutoProcessor.from_pretrained(mid, padding_side="left")
    kw.setdefault("device_map", "cuda")
    model = AutoModelForImageTextToText.from_pretrained(mid, config=cfg, dtype=torch.bfloat16,
                                                        attn_implementation=attn, **kw).eval()
    return proc, model


def install_drop(model, drop_after, state):
    """Mask image keys for every layer with index >= drop_after (0-based). Returns the hook handles."""
    handles = []
    cache = {}

    def pre(module, args, kwargs):
        m = kwargs.get("attention_mask")
        if m is None or state.is_image is None:
            return args, kwargs
        key = (m.data_ptr(), tuple(m.shape), state.is_image.data_ptr())
        if key not in cache:
            if len(cache) > 4:
                cache.clear()
            img = state.is_image[:, None, None, :].to(m.device)       # [B, 1, 1, T] over keys
            if m.dtype == torch.bool:
                m2 = m & ~img
            else:
                m2 = m.masked_fill(img, torch.finfo(m.dtype).min)
            cache[key] = m2
        kwargs["attention_mask"] = cache[key]
        return args, kwargs

    for i, layer in enumerate(model.model.language_model.layers):
        if i >= drop_after:
            handles.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
    return handles
