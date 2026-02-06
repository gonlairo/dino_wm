import torch
import numpy as np
from pathlib import Path
from typing import Callable, Optional, Dict, Any, Tuple
from einops import rearrange

from .traj_dset import TrajDataset, get_train_val_sliced


def _get_key(data: Dict[str, Any], keys: Tuple[str, ...]):
    for key in keys:
        if key in data:
            return data[key]
    return None


class CartpoleEpisodeDataset(TrajDataset):
    """
    Minimal dataset for a single long sequence.
    Expected keys in the file:
      - images / visual / obses: (T, C, H, W)
      - proprio: (T, D_p)
      - actions: (T, D_a)
      - states: optional (T, D_s)
    """

    def __init__(
        self,
        data_path: str = "data/cartpole",
        data_file: str = "episodes.pth",
        episode_len: int = 500,
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        normalize_action: bool = True,
        action_scale: float = 1.0,
    ):
        self.data_path = Path(data_path)
        self.transform = transform
        self.normalize_action = normalize_action

        data = torch.load(self.data_path / data_file)
        images = _get_key(data, ("images", "visual", "obses"))
        proprios = _get_key(data, ("proprio", "proprios"))
        actions = _get_key(data, ("actions", "acts"))
        states = _get_key(data, ("states", "state"))
        if images is None or proprios is None or actions is None:
            raise ValueError(
                "Expected keys in data file: images/visual/obses, proprio, actions."
            )

        images = torch.as_tensor(images)
        proprios = torch.as_tensor(proprios)
        actions = torch.as_tensor(actions)
        states = torch.as_tensor(
            states) if states is not None else proprios.clone()

        if images.ndim != 4:
            raise ValueError(
                f"Expected images shape (T, C, H, W), got {images.shape}")
        if images.shape[0] % episode_len != 0:
            raise ValueError(
                f"Total T {images.shape[0]} not divisible by episode_len {episode_len}"
            )
        num_episodes = images.shape[0] // episode_len
        images = images.view(num_episodes, episode_len, *images.shape[1:])
        proprios = proprios.view(num_episodes, episode_len, -1)
        actions = actions.view(num_episodes, episode_len, -1)
        states = states.view(num_episodes, episode_len, -1)
        seq_lengths = torch.full((num_episodes,), episode_len, dtype=torch.long)

        if n_rollout is not None:
            images = images[:n_rollout]
            proprios = proprios[:n_rollout]
            actions = actions[:n_rollout]
            states = states[:n_rollout]
            seq_lengths = seq_lengths[:n_rollout]

        self.images = images.float()
        self.proprios = proprios.float()
        self.actions = actions.float() / action_scale
        self.states = states.float()
        self.seq_lengths = seq_lengths

        self.action_dim = self.actions.shape[-1]
        self.state_dim = self.states.shape[-1]
        self.proprio_dim = self.proprios.shape[-1]

        if normalize_action:
            self.action_mean, self.action_std = self.get_data_mean_std(
                self.actions, self.seq_lengths)
            self.state_mean, self.state_std = self.get_data_mean_std(
                self.states, self.seq_lengths)
            self.proprio_mean, self.proprio_std = self.get_data_mean_std(
                self.proprios, self.seq_lengths)
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)

        self.actions = (self.actions - self.action_mean) / self.action_std
        self.proprios = (self.proprios - self.proprio_mean) / self.proprio_std

    def get_data_mean_std(self, data, traj_lengths):
        all_data = []
        for traj in range(len(traj_lengths)):
            traj_len = int(traj_lengths[traj])
            traj_data = data[traj, :traj_len]
            all_data.append(traj_data)
        all_data = torch.vstack(all_data)
        data_mean = torch.mean(all_data, dim=0)
        data_std = torch.std(all_data, dim=0)
        data_std = torch.where(data_std == 0, torch.ones_like(data_std),
                               data_std)
        return data_mean, data_std

    def get_seq_length(self, idx):
        return int(self.seq_lengths[idx])

    def get_frames(self, idx, frames):
        image = self.images[idx, frames]
        image = image / 255.0 if image.max() > 1.0 else image
        if image.ndim == 4 and image.shape[-1] == 3:
            image = rearrange(image, "t h w c -> t c h w")
        if self.transform:
            image = self.transform(image)
        proprio = self.proprios[idx, frames]
        act = self.actions[idx, frames]
        state = self.states[idx, frames]
        obs = {"visual": image, "proprio": proprio}
        return obs, act, state, {}

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.seq_lengths)


def load_cartpole_slice_train_val(
    transform,
    n_rollout=None,
    data_path="data/cartpole",
    data_file="episodes.pth",
    episode_len=500,
    normalize_action=True,
    split_ratio=0.9,
    num_hist=0,
    num_pred=0,
    frameskip=0,
    action_scale=1.0,
):
    dset = CartpoleEpisodeDataset(
        n_rollout=n_rollout,
        data_path=data_path,
        data_file=data_file,
        episode_len=episode_len,
        transform=transform,
        normalize_action=normalize_action,
        action_scale=action_scale,
    )
    dset_train, dset_val, train_slices, val_slices = get_train_val_sliced(
        traj_dataset=dset,
        train_fraction=split_ratio,
        num_frames=num_hist + num_pred,
        frameskip=frameskip,
    )

    datasets = {"train": train_slices, "valid": val_slices}
    traj_dset = {"train": dset_train, "valid": dset_val}
    return datasets, traj_dset
