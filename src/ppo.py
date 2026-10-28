"""PPO (clipped surrogate) with GAE, written from scratch in PyTorch.

Algorithm only: nothing here knows about Footsies. Every place where bias or
variance enters the estimator is marked with a "BIAS/VARIANCE" comment.
"""
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


@dataclass
class PPOConfig:
    # hyperparameters of the FootsiesGym paper (Table 3)
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_coef: float = 0.3
    vf_coef: float = 1.0
    max_grad_norm: float = 0.5
    num_epochs: int = 8
    num_minibatches: int = 8
    lr: float = 1e-4  # linearly annealed to 0
    anneal_lr: bool = True
    ent_coef: float = 0.025  # fixed value, or start value of the decay
    ent_decay: bool = False  # if True: ent_coef -> 0 linearly
    norm_adv: bool = True
    # Lower bound for the std used to normalize advantages. When rewards are ~0 (a stalemate)
    # the raw advantages have std ~1e-3 and dividing by it turns noise into full-size updates.
    # 0 = classic normalization (the setting of runs r0-r2).
    adv_std_floor: float = 0.0
    clip_vloss: bool = False  # value clipping is off: returns are on a +-10 scale
    hidden: int = 256


def _layer(layer, std=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class Agent(nn.Module):
    """Shared-feature-free actor-critic: two separate MLPs 2x256 ReLU."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.critic = nn.Sequential(
            _layer(nn.Linear(obs_dim, hidden)), nn.ReLU(),
            _layer(nn.Linear(hidden, hidden)), nn.ReLU(),
            _layer(nn.Linear(hidden, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            _layer(nn.Linear(obs_dim, hidden)), nn.ReLU(),
            _layer(nn.Linear(hidden, hidden)), nn.ReLU(),
            _layer(nn.Linear(hidden, n_actions), std=0.01),
        )

    def get_value(self, obs):
        return self.critic(obs).squeeze(-1)

    def get_action_and_value(self, obs, action=None):
        dist = Categorical(logits=self.actor(obs))
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(obs).squeeze(-1)


@torch.no_grad()
def compute_gae(rewards, values, terminated, truncated, gamma, lam):
    """Generalized Advantage Estimation on a rollout of shape (T, S).

    values has shape (T+1, S): V(s_0..s_T), where V(s_T) is the bootstrap value
    of the observation after the last step.

    Episode ends are handled in two different ways:
      * terminated: true end of the game, next value = 0 (no bias);
      * truncated (max_t): the game is NOT over, the correct target bootstraps
        on V(final obs). The vectorized env overwrites the final obs with the
        reset obs, so we approximate it with V(s_t) of the previous obs.
        BIAS/VARIANCE: this is a (small) bias, but far smaller than treating
        the truncation as a terminal state with value 0.
    In both cases the advantage recursion is cut (the next obs belongs to a
    new episode).

    lam interpolates between TD(0) (lam=0: low variance, biased by the critic)
    and Monte Carlo (lam=1: unbiased w.r.t. the critic, high variance).
    """
    T = rewards.shape[0]
    done = (terminated | truncated).float()
    next_values = values[1:].clone()
    next_values = torch.where(terminated, torch.zeros_like(next_values), next_values)
    next_values = torch.where(truncated & ~terminated, values[:-1], next_values)

    adv = torch.zeros_like(rewards)
    last = torch.zeros_like(rewards[0])
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_values[t] - values[t]
        last = delta + gamma * lam * (1.0 - done[t]) * last
        adv[t] = last
    returns = adv + values[:-1]
    return adv, returns


def ppo_update(agent, optimizer, batch, cfg: PPOConfig, ent_coef: float):
    """Several epochs of minibatch SGD on one rollout. Returns diagnostics.

    batch: dict of flat tensors obs, actions, logprobs, advantages, returns, values.
    """
    n = batch["obs"].shape[0]
    mb_size = n // cfg.num_minibatches
    stats = {k: [] for k in ["pg_loss", "v_loss", "entropy", "approx_kl", "clipfrac"]}

    for _ in range(cfg.num_epochs):
        # BIAS/VARIANCE: reusing the same batch for several epochs makes the
        # samples off-policy; the importance ratio corrects the expectation,
        # the clipping bounds its variance at the cost of a biased surrogate.
        perm = torch.randperm(n, device=batch["obs"].device)
        for start in range(0, n - mb_size + 1, mb_size):
            idx = perm[start:start + mb_size]
            _, new_logp, entropy, new_v = agent.get_action_and_value(
                batch["obs"][idx], batch["actions"][idx])
            log_ratio = new_logp - batch["logprobs"][idx]
            ratio = log_ratio.exp()

            adv = batch["advantages"][idx]
            if cfg.norm_adv:
                adv = (adv - adv.mean()) / (adv.std().clamp(min=cfg.adv_std_floor) + 1e-8)

            pg1 = -adv * ratio
            pg2 = -adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
            pg_loss = torch.max(pg1, pg2).mean()

            ret = batch["returns"][idx]
            if cfg.clip_vloss:
                v_old = batch["values"][idx]
                v_clip = v_old + torch.clamp(new_v - v_old, -cfg.clip_coef, cfg.clip_coef)
                v_loss = 0.5 * torch.max((new_v - ret) ** 2, (v_clip - ret) ** 2).mean()
            else:
                v_loss = 0.5 * ((new_v - ret) ** 2).mean()

            # BIAS/VARIANCE: the entropy bonus deliberately biases the policy
            # gradient towards exploration; it vanishes if ent_decay is on.
            ent = entropy.mean()
            loss = pg_loss + cfg.vf_coef * v_loss - ent_coef * ent

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                stats["pg_loss"].append(pg_loss.item())
                stats["v_loss"].append(v_loss.item())
                stats["entropy"].append(ent.item())
                stats["approx_kl"].append(((ratio - 1) - log_ratio).mean().item())
                stats["clipfrac"].append(((ratio - 1).abs() > cfg.clip_coef).float().mean().item())

    out = {k: float(np.mean(v)) for k, v in stats.items()}
    # explained variance of the critic on the whole batch: 1 = perfect,
    # <=0 = useless. Diagnostic for the critic bias.
    y_pred, y_true = batch["values"].cpu().numpy(), batch["returns"].cpu().numpy()
    var_y = np.var(y_true)
    out["explained_var"] = float("nan") if var_y == 0 else float(1 - np.var(y_true - y_pred) / var_y)
    out["adv_std"] = float(batch["advantages"].std().item())
    out["return_std"] = float(np.sqrt(var_y))
    return out


def save_agent(path, agent, cfg: PPOConfig, obs_dim, n_actions, steps):
    torch.save({"agent": agent.state_dict(), "cfg": asdict(cfg), "obs_dim": obs_dim,
                "n_actions": n_actions, "steps": steps}, path)


def load_agent(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    agent = Agent(ck["obs_dim"], ck["n_actions"], ck["cfg"]["hidden"]).to(device)
    agent.load_state_dict(ck["agent"])
    return agent, ck
