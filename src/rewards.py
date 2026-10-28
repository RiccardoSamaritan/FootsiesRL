"""Training-reward variants (R0-R3). Evaluation always uses the terminal win/loss.

make_reward(name) returns (reward_fn, env_overrides):
  * env_overrides: extra FootsiesEnv config (R1 uses the library's own guard reward);
  * reward_fn: optional transformation applied by FootsiesVecEnv.step (see env.py),
    called as reward_fn(env, prev_obs, obs, rewards, terminated, truncated) with
    arrays of shape (2N, ...) and returning the training rewards.

Observation features (p1-centric obs, footsiesgym feature_indices.json): own
guard health one-hot at [6:10], opponent's at [47:51] (class k = guard health k,
0..3). The environment's own reward is zero-sum and terminal: +-W (W = 10) at the
end of the game, 0 elsewhere and 0 on timeout.

  r0  terminal reward only (environment default, as in the paper)
  r1  dense guard-break reward taken out of the win reward (library option)
  r2  potential-based shaping on the guard difference (Ng et al., 1999)
  r3  win/loss scaled by game length (faster wins are worth more)
"""
import numpy as np

REWARDS = ("r0", "r1", "r2", "r3")

WIN = 10.0
GUARD_TOTAL = 6.0   # total shaping mass over the 3 guard points (must be < WIN)
OWN_GUARD = slice(6, 10)
OPP_GUARD = slice(47, 51)


def guard_potential(obs: np.ndarray) -> np.ndarray:
    """Phi(s) = GUARD_TOTAL * (own_guard - opp_guard) / 3, range [-6, 6].

    Zero at the start of a game (both guards full) and antisymmetric between
    the two players, so the shaping stays zero-sum.
    """
    own = obs[:, OWN_GUARD].argmax(axis=1)
    opp = obs[:, OPP_GUARD].argmax(axis=1)
    return GUARD_TOTAL * (own - opp) / 3.0


def make_reward(name: str, gamma: float = 0.995, max_t: int = 4000,
                min_time_factor: float = 0.5):
    if name == "r0":
        return None, {}

    if name == "r1":
        # 2 per guard point, deducted from the win reward (reward budget): the
        # total return of a win stays WIN.
        return None, {"guard_break_reward": GUARD_TOTAL / 3.0, "use_reward_budget": True}

    if name == "r2":
        def reward_fn(env, prev_obs, obs, rewards, terminated, truncated):
            phi = guard_potential(prev_obs)
            phi_next = guard_potential(obs)
            # Terminal state: Phi = 0. Truncation: the true next obs is lost
            # (auto-reset), so Phi(s') is approximated with Phi(s).
            phi_next = np.where(terminated, 0.0, np.where(truncated, phi, phi_next))
            return rewards + gamma * phi_next - phi
        return reward_fn, {}

    if name == "r3":
        def reward_fn(env, prev_obs, obs, rewards, terminated, truncated):
            t = np.tile(env.t, 2).astype(np.float32)  # step counter before this step
            factor = 1.0 - (1.0 - min_time_factor) * np.clip(t / max_t, 0.0, 1.0)
            return np.where(terminated, rewards * factor, rewards)
        return reward_fn, {}

    raise ValueError(f"unknown reward variant {name!r}; available: {REWARDS}")
