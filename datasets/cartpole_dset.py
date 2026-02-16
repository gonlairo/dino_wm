import torch
import numpy as np
from pathlib import Path
from typing import Callable, Optional, Dict, Any, Tuple, List
from einops import rearrange

from .traj_dset import TrajDataset, get_train_val_sliced


def _get_key(data: Dict[str, Any], keys: Tuple[str, ...]):
    for key in keys:
        if key in data:
            return data[key]
    return None


def _natural_sort_key(p: Path) -> Tuple[int, ...]:
    """Sort episode_0001.pth, episode_0002.pth, ... by numeric part."""
    stem = p.stem
    if stem.startswith("episode_"):
        try:
            return (int(stem.split("_")[1]),)
        except (IndexError, ValueError):
            pass
    return (0,)


class CartpoleEpisodeDirDataset(TrajDataset):
    """
    Dataset where each episode is stored in a separate file:
      data_path/episode_0001.pth, episode_0002.pth, ...

    Each file must contain:
      - 'images': (T, C, H, W)
      - 'proprio': (T, D_p)
      - 'actions': (T, D_a)
    Optional: 'states' (T, D_s); if missing, state = proprio.
    """

    def __init__(
        self,
        data_path: str = "data/cartpole",
        episode_pattern: str = "episode_*.pth",
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        normalize_action: bool = True,
        action_scale: float = 1.0,
        use_proprio: bool = True,
    ):
        self.data_path = Path(data_path)
        self.transform = transform
        self.normalize_action = normalize_action
        self.use_proprio = use_proprio

        paths = sorted(
            self.data_path.glob(episode_pattern),
            key=_natural_sort_key,
        )
        if not paths:
            raise FileNotFoundError(
                f"No episodes found at {self.data_path / episode_pattern}")

        self.episode_paths: List[Path] = paths
        if n_rollout is not None:
            self.episode_paths = self.episode_paths[:n_rollout]
        n = len(self.episode_paths)

        # Load first episode to get dims and max length
        sample = torch.load(self.episode_paths[0])
        images = _get_key(sample, ("images", "visual", "obses"))
        proprio = _get_key(sample, ("proprio", "proprios"))
        actions = _get_key(sample, ("actions", "acts"))
        states = _get_key(sample, ("states", "state"))

        images = torch.as_tensor(images)
        state_dim = torch.as_tensor(proprio).shape[-1]
        action_dim = torch.as_tensor(actions).shape[-1]
        proprio_dim = state_dim

        seq_lengths_list: List[int] = []
        for p in self.episode_paths:
            data = torch.load(p)
            T = data["images"].shape[0]
            seq_lengths_list.append(T)
        seq_lengths = torch.tensor(seq_lengths_list, dtype=torch.long)
        max_T = int(seq_lengths.max().item())

        self.states = torch.zeros(n, max_T, state_dim, dtype=torch.float32)
        self.actions = torch.zeros(n, max_T, action_dim, dtype=torch.float32)
        self.proprios = torch.zeros(n, max_T, proprio_dim, dtype=torch.float32)

        for i, p in enumerate(self.episode_paths):
            data = torch.load(p)
            proprio = torch.as_tensor(data["proprio"]).float()
            actions = torch.as_tensor(data["actions"]).float()
            states = _get_key(data, ("states", "state"))
            if states is None:
                states = proprio
            else:
                states = torch.as_tensor(states).float()
            T = seq_lengths[i].item()
            if use_proprio:
                self.proprios[i, :T] = proprio
            self.actions[i, :T] = actions / action_scale
            self.states[i, :T] = states

        self.seq_lengths = seq_lengths
        self.action_dim = action_dim
        self.state_dim = state_dim
        if use_proprio:
            self.proprio_dim = proprio_dim
        else:
            self.proprios = torch.zeros(n, max_T, 1, dtype=torch.float32)
            self.proprio_dim = 1

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
        print(f"Loaded {n} episodes from {self.data_path} (episode_*.pth)" +
              (" [proprio disabled]" if not use_proprio else ""))

    def get_data_mean_std(self, data: torch.Tensor, traj_lengths: torch.Tensor):
        all_data = []
        for traj in range(len(traj_lengths)):
            traj_len = int(traj_lengths[traj].item())
            all_data.append(data[traj, :traj_len])
        all_data = torch.vstack(all_data)
        data_mean = torch.mean(all_data, dim=0)
        data_std = torch.std(all_data, dim=0)
        data_std = torch.where(data_std == 0, torch.ones_like(data_std),
                               data_std)
        return data_mean, data_std

    def get_seq_length(self, idx: int) -> int:
        return int(self.seq_lengths[idx].item())

    def get_frames(self, idx: int, frames):
        path = self.episode_paths[idx]
        data = torch.load(path)
        images = torch.as_tensor(data["images"])
        images = images[frames]
        images = images.float() / 255.0 if images.max() > 1.0 else images.float(
        )
        assert images.shape[
            1] == 3, f"Expected images shape (T, 3, H, W), got {images.shape}"
        if self.transform:
            images = self.transform(images)
        proprio = self.proprios[idx, frames]
        act = self.actions[idx, frames]
        state = self.states[idx, frames]
        obs = {"visual": images, "proprio": proprio}
        return obs, act, state, {}

    def __getitem__(self, idx: int):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self) -> int:
        return len(self.episode_paths)


def load_cartpole_episode_dir_slice_train_val(
    transform: Callable,
    n_rollout: Optional[int] = None,
    data_path: str = "data/cartpole",
    episode_pattern: str = "episode_*.pth",
    normalize_action: bool = True,
    split_ratio: float = 0.9,
    num_hist: int = 0,
    num_pred: int = 0,
    frameskip: int = 0,
    action_scale: float = 1.0,
    use_proprio: bool = True,
):
    """Load cartpole-style data from per-episode files (episode_0001.pth, etc.)."""
    dset = CartpoleEpisodeDirDataset(
        data_path=data_path,
        episode_pattern=episode_pattern,
        n_rollout=n_rollout,
        transform=transform,
        normalize_action=normalize_action,
        action_scale=action_scale,
        use_proprio=use_proprio,
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
