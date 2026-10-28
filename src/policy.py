"""Adapter that lets a trained PPO checkpoint play as a fixed Opponent.

Evaluation is always stochastic: actions are sampled from the policy, never
taken with argmax. In a game with mixed strategies a deterministic policy is
exploitable by construction (see the evaluation section of the project draft).
"""
import os

import numpy as np
import torch

from ppo import load_agent


class PolicyOpponent:
    """A trained policy playing one side of the match.

    Implements the Opponent protocol of opponents.py, so evaluate.py can use a
    checkpoint and a fixed opponent interchangeably. Observations are already
    side-relative, so the same object can play either p1 or p2.
    """

    def __init__(self, agent, obs_dim: int, n_actions: int,
                 device: str = "cpu", name: str = "policy"):
        self.agent = agent.to(device).eval()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.device = device
        self.name = name

    def reset(self, env_id: str = "default") -> None:
        """Nothing to clear: the network is feed-forward and keeps no state."""

    @torch.no_grad()
    def act(self, agent_id: str, obs, info: dict) -> int:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(
                f"{self.name} was trained on {self.obs_dim}-dim observations "
                f"but got {obs.shape[-1]}: the evaluation env config differs "
                "from the training one (e.g. use_special_charge_action).")
        x = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        action, _, _, _ = self.agent.get_action_and_value(x)
        return int(action.item())

    @torch.no_grad()
    def act_batch(self, obs) -> np.ndarray:
        """Sample one action per row of obs (shape (B, obs_dim)); used by the vectorized evaluation."""
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(
                f"{self.name} was trained on {self.obs_dim}-dim observations "
                f"but got {obs.shape[-1]}.")
        x = torch.as_tensor(obs, device=self.device)
        action, _, _, _ = self.agent.get_action_and_value(x)
        return action.cpu().numpy()


def load_policy_opponent(path: str, device: str = "cpu") -> PolicyOpponent:
    """Build a PolicyOpponent from a checkpoint saved by ppo.save_agent."""
    agent, ck = load_agent(path, device=device)
    opponent = PolicyOpponent(agent, ck["obs_dim"], ck["n_actions"], device=device,
                              name=os.path.basename(path))
    opponent.steps = int(ck["steps"])  # training steps at the time of the checkpoint
    return opponent
