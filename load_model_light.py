"""
Lightweight loader for the world model. Same behavior as plan.load_model
but with minimal imports (no gym, wandb, env, planning, etc.).
Use: from load_model_light import load_model
"""
import os
import torch
from pathlib import Path
import hydra

ALL_MODEL_KEYS = [
    "encoder",
    "predictor",
    "decoder",
    "proprio_encoder",
    "action_encoder",
]


def load_ckpt(snapshot_path, device):
    with Path(snapshot_path).open("rb") as f:
        payload = torch.load(f, map_location=device)
    # Check top-level first; some checkpoints nest under "state_dict" or "model"
    source = payload
    if not any(k in payload for k in ALL_MODEL_KEYS):
        source = payload.get("state_dict") or payload.get("model") or payload
    result = {}
    for k, v in source.items():
        if k in ALL_MODEL_KEYS:
            result[k] = v.to(device) if hasattr(v, "to") else v
    # Case-insensitive fallback for key names (e.g. "Predictor" vs "predictor")
    for ckpt_key, v in source.items():
        if ckpt_key in result:
            continue
        for model_key in ALL_MODEL_KEYS:
            if model_key.lower() == ckpt_key.lower():
                result[model_key] = v.to(device) if hasattr(v, "to") else v
                break
    result["epoch"] = payload.get("epoch", source.get("epoch", 0))
    return result


def load_model(model_ckpt, train_cfg, num_action_repeat, device):
    """Load world model from checkpoint. Same signature and behavior as plan.load_model."""
    model_ckpt = Path(model_ckpt)
    result = {}
    if model_ckpt.exists():
        result = load_ckpt(model_ckpt, device)
        print(f"Resuming from epoch {result['epoch']}: {model_ckpt}")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {model_ckpt}")
    if "encoder" not in result:
        result["encoder"] = hydra.utils.instantiate(train_cfg.encoder)
    if "predictor" not in result:
        if getattr(train_cfg, "has_predictor", True):
            # Instantiate from config (checkpoint missing predictor key, e.g. older save or train_predictor=False)
            import warnings
            warnings.warn(
                "Predictor not in checkpoint; instantiating from config (untrained weights)."
            )
            encoder = result["encoder"]
            proprio_emb_dim = result["proprio_encoder"].emb_dim
            action_emb_dim = result["action_encoder"].emb_dim
            if getattr(encoder, "latent_ndim", 2) == 1:
                num_patches = 1
            else:
                decoder_scale = 16
                num_side_patches = train_cfg.img_size // decoder_scale
                num_patches = num_side_patches**2
            if getattr(train_cfg, "concat_dim", 1) == 0:
                num_patches += 2
            dim = (encoder.emb_dim +
                   (proprio_emb_dim * train_cfg.num_proprio_repeat +
                    action_emb_dim * train_cfg.num_action_repeat) *
                   getattr(train_cfg, "concat_dim", 1))
            result["predictor"] = hydra.utils.instantiate(
                train_cfg.predictor,
                num_patches=num_patches,
                num_frames=train_cfg.num_hist,
                dim=dim,
            )
        else:
            result["predictor"] = None

    if train_cfg.has_decoder and "decoder" not in result:
        base_path = os.path.dirname(os.path.abspath(__file__))
        if train_cfg.env.decoder_path is not None:
            decoder_path = os.path.join(base_path, train_cfg.env.decoder_path)
            ckpt = torch.load(decoder_path, map_location=device)
            if isinstance(ckpt, dict):
                result["decoder"] = ckpt["decoder"]
            else:
                result["decoder"] = ckpt
        else:
            raise ValueError("Decoder path not found in model checkpoint "
                             "and is not provided in config")
    elif not train_cfg.has_decoder:
        result["decoder"] = None

    model = hydra.utils.instantiate(
        train_cfg.model,
        encoder=result["encoder"],
        proprio_encoder=result["proprio_encoder"],
        action_encoder=result["action_encoder"],
        predictor=result["predictor"],
        decoder=result["decoder"],
        proprio_dim=train_cfg.proprio_emb_dim,
        action_dim=train_cfg.action_emb_dim,
        concat_dim=train_cfg.concat_dim,
        num_action_repeat=num_action_repeat,
        num_proprio_repeat=train_cfg.num_proprio_repeat,
    )
    model.to(device)
    return model
