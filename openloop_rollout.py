import argparse
import os
from pathlib import Path
import random

import hydra
import torch
import numpy as np
from einops import rearrange
from omegaconf import OmegaConf

from plan import load_model
from utils import slice_trajdict_with_t, seed


def err_eval_single(model, z_pred, z_tgt):
    logs = {}
    for k in z_pred.keys():
        loss = model.emb_criterion(z_pred[k], z_tgt[k])
        logs[k] = loss.item()
    return logs


def openloop_rollout(
    model,
    dset,
    num_rollout,
    num_hist,
    frameskip,
    min_horizon,
    rand_start_end,
    device,
):
    logs = {}
    min_horizon = min_horizon + num_hist
    num_past = [(num_hist, ""), (1, "_1framestart")]

    for _ in range(num_rollout):
        valid_traj = False
        while not valid_traj:
            traj_idx = random.randint(0, len(dset) - 1)
            obs, act, _, _ = dset[traj_idx]
            if rand_start_end:
                if obs["visual"].shape[0] > min_horizon * frameskip + 1:
                    start = random.randint(
                        0,
                        obs["visual"].shape[0] - min_horizon * frameskip - 1,
                    )
                else:
                    start = 0
                max_horizon = (obs["visual"].shape[0] - start - 1) // frameskip
                if max_horizon > min_horizon:
                    valid_traj = True
                    horizon = random.randint(min_horizon, max_horizon)
            else:
                valid_traj = True
                start = 0
                horizon = (obs["visual"].shape[0] - 1) // frameskip

        for k in obs.keys():
            obs[k] = obs[k][start:start + horizon * frameskip + 1:frameskip]
        act = act[start:start + horizon * frameskip]
        act = rearrange(act, "(h f) d -> h (f d)", f=frameskip)

        obs_g = {}
        for k in obs.keys():
            obs_g[k] = obs[k][-1].unsqueeze(0).unsqueeze(0).to(device)
        z_g = model.encode_obs(obs_g)
        actions = act.unsqueeze(0).to(device)

        for n_past, postfix in num_past:
            obs_0 = {}
            for k in obs.keys():
                obs_0[k] = obs[k][:n_past].unsqueeze(0).to(device)

            with torch.no_grad():
                z_obses, _ = model.rollout(obs_0, actions)
            z_obs_last = slice_trajdict_with_t(z_obses,
                                               start_idx=-1,
                                               end_idx=None)
            div_loss = err_eval_single(model, z_obs_last, z_g)

            for k in div_loss.keys():
                log_key = f"z_{k}_err_rollout{postfix}"
                logs.setdefault(log_key, []).append(div_loss[k])

    logs = {
        key: sum(values) / len(values) for key, values in logs.items() if values
    }
    return logs


def main():
    parser = argparse.ArgumentParser(
        description="Open-loop rollout on a dataset.")
    parser.add_argument("--model_path",
                        required=True,
                        help="Path to model output folder")
    parser.add_argument("--model_epoch",
                        default="final",
                        help="Checkpoint epoch tag")
    parser.add_argument("--num_rollout", type=int, default=10)
    parser.add_argument("--min_horizon", type=int, default=2)
    parser.add_argument("--rand_start_end", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="cuda or cpu")
    args = parser.parse_args()

    seed(args.seed)
    device = torch.device(args.device if args.device is not None else (
        "cuda:0" if torch.cuda.is_available() else "cpu"))

    model_path = Path(args.model_path)
    cfg_path = model_path / "hydra.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"hydra.yaml not found at {cfg_path}")

    model_cfg = OmegaConf.load(cfg_path)
    _, traj_dset = hydra.utils.call(
        model_cfg.env.dataset,
        num_hist=model_cfg.num_hist,
        num_pred=model_cfg.num_pred,
        frameskip=model_cfg.frameskip,
    )
    dset = traj_dset["valid"] if isinstance(traj_dset, dict) else traj_dset

    ckpt_path = model_path / "checkpoints" / f"model_{args.model_epoch}.pth"
    model = load_model(
        ckpt_path,
        model_cfg,
        model_cfg.num_action_repeat,
        device=device,
    )
    model.eval()

    logs = openloop_rollout(
        model=model,
        dset=dset,
        num_rollout=args.num_rollout,
        num_hist=model_cfg.num_hist,
        frameskip=model_cfg.frameskip,
        min_horizon=args.min_horizon,
        rand_start_end=args.rand_start_end,
        device=device,
    )

    for k, v in logs.items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()
