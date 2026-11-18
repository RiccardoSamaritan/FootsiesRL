# policy.py
# Here we wrap a trained PPO checkpoint so that it can play as a fixed opponent,
# exactly like the heuristic opponents of opponents.py (random, noop, back, bot).
#
# In the evaluation we always sample the actions from the policy, we never take
# the argmax: in a game with mixed strategies a deterministic policy can be
# exploited by construction (we explain this in the evaluation section of the draft).

import os

import numpy as np
import torch

from ppo import load_agent


class PolicyOpponent:
    """A trained policy that plays one side of the match.

    It follows the same interface as the opponents in opponents.py, so in evaluate.py
    we can use a checkpoint or a fixed opponent in the same way. The observations are
    already relative to the player's side, so the same object can play as p1 or as p2.
    """

    def __init__(self, agent, obs_dim: int, n_actions: int,
                 device: str = "cpu", name: str = "policy", env_config: dict | None = None):
        self.agent = agent.to(device).eval()
        self.env_config = dict(env_config or {})  # game settings used during training
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.device = device
        self.name = name

    def reset(self, env_id: str = "default") -> None:
        """Nothing to reset: the network has no memory between steps."""

    @torch.no_grad()
    def act(self, agent_id: str, obs, info: dict) -> int:
        """Samples the action for a single observation."""
        obs = np.asarray(obs, dtype=np.float32)
        # if the sizes do not match, the evaluation env has different settings from
        # the training one, and we prefer to stop with a clear error
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(
                f"{self.name} was trained on {self.obs_dim}-dim observations "
                f"but got {obs.shape[-1]}: the evaluation env config differs "
                "from the training one (e.g. use_special_charge_action).")
        x = torch.as_tensor(obs, device=self.device).unsqueeze(0)  # add the batch dimension
        action, _, _, _ = self.agent.get_action_and_value(x)
        return int(action.item())

    @torch.no_grad()
    def act_batch(self, obs) -> np.ndarray:
        """Samples one action for each row of obs (shape (B, obs_dim)); we use it in the vectorized evaluation."""
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(
                f"{self.name} was trained on {self.obs_dim}-dim observations "
                f"but got {obs.shape[-1]}.")
        x = torch.as_tensor(obs, device=self.device)
        action, _, _, _ = self.agent.get_action_and_value(x)
        return action.cpu().numpy()


def load_policy_opponent(path: str, device: str = "cpu") -> PolicyOpponent:
    """Builds a PolicyOpponent from a checkpoint saved with ppo.save_agent."""
    agent, ck = load_agent(path, device=device)
    opponent = PolicyOpponent(agent, ck["obs_dim"], ck["n_actions"], device=device,
                              name=os.path.basename(path), env_config=ck.get("env_config", {}))
    opponent.steps = int(ck["steps"])  # training steps reached when the checkpoint was saved
    return opponent
