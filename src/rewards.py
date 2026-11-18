# rewards.py
# The different training rewards we compare (R0, R1, R2, R3).
#
# The default reward of the env is zero-sum and only at the end of the game:
# +10 / -10 when someone is knocked out, 0 during the game and 0 on timeout.
#
# The variants:
#   r0  only the final reward (default of the env, same as the paper)
#   r1  reward every time a guard point is broken, taken from the win reward
#   r2  potential-based shaping on the guard difference
#   r3  win/loss multiplied by a factor that gets smaller the longer the game lasts

import numpy as np

REWARDS = ("r0", "r1", "r2", "r3")

WIN = 10.0
GUARD_TOTAL = 6.0   # total reward given for the 3 guard points (has to be < WIN)
OWN_GUARD = slice(6, 10)
OPP_GUARD = slice(47, 51)


def guard_potential(obs: np.ndarray) -> np.ndarray:
    """Potential used by R2: Phi(s) = GUARD_TOTAL * (own_guard - opp_guard) / 3, between -6 and 6.

    At the beginning of a game both guards are full, so Phi = 0. It is also
    antisymmetric between the two players (what one gains the other loses),
    so the shaping keeps the game zero-sum.
    """
    own = obs[:, OWN_GUARD].argmax(axis=1)  # one-hot -> guard value 0..3
    opp = obs[:, OPP_GUARD].argmax(axis=1)
    return GUARD_TOTAL * (own - opp) / 3.0


def make_reward(name: str, gamma: float = 0.995, max_t: int = 4000,
                min_time_factor: float = 0.5):
    """Returns (reward_fn, env_overrides) for the reward variant called name (r0, r1, r2 or r3)."""
    if name == "r0":
        # nothing to change: the env already gives the final reward
        return None, {}

    if name == "r1":
        # 2 points for each guard point broken, taken away from the win reward
        # (reward budget), so a win is still worth WIN in total
        return None, {"guard_break_reward": GUARD_TOTAL / 3.0, "use_reward_budget": True}

    if name == "r2":
        def reward_fn(env, prev_obs, obs, rewards, terminated, truncated):
            """R2: adds gamma * Phi(s') - Phi(s) to the reward of the env."""
            phi = guard_potential(prev_obs)
            phi_next = guard_potential(obs)
            # if the game is over Phi = 0. On timeout we don't have the real next obs
            # (the env already reset the game), so I use Phi(s) instead of Phi(s')
            phi_next = np.where(terminated, 0.0, np.where(truncated, phi, phi_next))
            return rewards + gamma * phi_next - phi
        return reward_fn, {}

    if name == "r3":
        def reward_fn(env, prev_obs, obs, rewards, terminated, truncated):
            """R3: the final reward is multiplied by a factor that goes from 1 to min_time_factor."""
            t = np.tile(env.t, 2).astype(np.float32)  # steps of each game before this step (p1 and p2)
            factor = 1.0 - (1.0 - min_time_factor) * np.clip(t / max_t, 0.0, 1.0)
            return np.where(terminated, rewards * factor, rewards)
        return reward_fn, {}

    raise ValueError(f"unknown reward variant {name!r}; available: {REWARDS}")
