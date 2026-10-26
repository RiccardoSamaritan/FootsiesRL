"""Vectorized FootsiesGym environment for self-play with a shared network.

N game instances run in a single Unity server (``num_envs=N``). Each game has
two players, so the learner sees 2N "slots": slots [0, N) are the p1 side of
games 0..N-1, slots [N, 2N) are the p2 side. All slots use the same policy
(self-play), each one from its own point of view.

Notes on the underlying env (footsiesgym vectorized mode):
  * finished games are auto-reset and the returned observation of a finished
    game is already the *reset* observation, not the final one;
  * ``reset(seed=...)`` ignores the seed.
"""
import socket
import sys
import time
from typing import Callable, Optional

import numpy as np

import footsiesgym

ENV_CONFIG = {
    "frame_skip": 4,
    "action_delay": 8,
    "max_t": 4000,
    "win_reward_scaling_coeff": 10.0,
    "guard_break_reward": 0.0,
    "use_reward_budget": False,
    "headless": True,
}


def default_platform() -> str:
    return "mac" if sys.platform == "darwin" else "linux"


class FootsiesVecEnv:
    """2N-slot view of a vectorized FootsiesEnv.

    reward_fn (optional) lets a future rewards module replace the training reward:
        reward_fn(env, prev_obs, obs, rewards, terminated, truncated) -> rewards
    where all arrays have shape (2N, ...). ``obs`` of a finished game is the
    reset observation (see above), so a shaper must treat done slots separately.
    ``env.t`` holds the per-game step counter *before* the step.
    """

    def __init__(
        self,
        num_envs: int,
        port: int = 45001,
        platform: Optional[str] = None,
        env_config: Optional[dict] = None,
        reward_fn: Optional[Callable] = None,
    ):
        cfg = dict(ENV_CONFIG)
        if env_config:
            cfg.update(env_config)
        cfg["num_envs"] = max(num_envs, 1)
        cfg["port"] = port
        self.port = port
        self.num_envs = num_envs
        self.num_slots = 2 * num_envs
        self.max_t = cfg["max_t"]
        self.win_reward = cfg["win_reward_scaling_coeff"]
        self.reward_fn = reward_fn
        self.env = footsiesgym.make(config=cfg, platform=platform or default_platform())
        self.obs_dim = self.env.observation_space("p1").shape[0]
        self.n_actions = self.env.action_space("p1").n

        self.t = np.zeros(num_envs, dtype=np.int64)
        self._ep_ret = np.zeros(num_envs)  # raw p1 return of the running episode
        self.finished = []  # per-episode stats, drained with pop_finished()
        self._obs = None

    # ------------------------------------------------------------------
    @staticmethod
    def _stack(d):
        return np.concatenate([d["p1"], d["p2"]], axis=0)

    def _wait_for_server(self, timeout: float = 120.0):
        # The Unity server can start slower than the library waits for it
        # (seen on macOS): wait until it accepts connections before reset().
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                socket.create_connection(("127.0.0.1", self.port), timeout=0.5).close()
                return
            except OSError:
                time.sleep(0.5)
        raise RuntimeError(f"Footsies server did not start on port {self.port}")

    def reset(self) -> np.ndarray:
        self._wait_for_server()
        obs, _ = self.env.reset()
        self.t[:] = 0
        self._ep_ret[:] = 0
        self._obs = self._stack(obs).astype(np.float32)
        return self._obs

    def step(self, actions: np.ndarray):
        N = self.num_envs
        actions = np.asarray(actions, dtype=np.int64)
        obs, rew, term, trunc, _ = self.env.step({"p1": actions[:N], "p2": actions[N:]})
        obs = self._stack(obs).astype(np.float32)
        rew = self._stack(rew).astype(np.float32)
        term = self._stack(term).astype(bool)
        trunc = self._stack(trunc).astype(bool)

        # episode statistics from the raw (unshaped) p1 reward
        self._ep_ret += rew[:N]
        done = term[:N] | trunc[:N]
        for i in np.flatnonzero(done):
            p1_reward = rew[:N][i]
            if term[i]:
                outcome = 1 if p1_reward > 0 else (-1 if p1_reward < 0 else 0)
            else:
                outcome = 0
            self.finished.append(
                {"len": int(self.t[i]) + 1, "truncated": bool(trunc[i] and not term[i]),
                 "p1_outcome": outcome, "p1_return": float(self._ep_ret[i])}
            )
            self._ep_ret[i] = 0

        train_rew = rew
        if self.reward_fn is not None:
            train_rew = self.reward_fn(self, self._obs, obs, rew, term, trunc).astype(np.float32)

        self.t += 1
        self.t[done] = 0
        self._obs = obs
        return obs, train_rew, term, trunc

    def pop_finished(self):
        out, self.finished = self.finished, []
        return out

    def close(self):
        self.env.close()
