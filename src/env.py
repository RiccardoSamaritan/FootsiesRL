# env.py
# Wrapper around the vectorized FootsiesGym environment, used for self-play.
#
# Idea: we run N games inside one Unity server (num_envs = N). Every game has
# two players, so in total there are 2N "slots":
#   - slots 0 .. N-1   -> player p1 of games 0 .. N-1
#   - slots N .. 2N-1  -> player p2 of games 0 .. N-1
# All the slots are controlled by the same network (self-play), and each slot
# sees the game from its own side.
#
# Things I found out about footsiesgym in vectorized mode:
#   - when a game ends it is reset automatically, and the observation we get
#     back is already the one of the NEW game (not the last one of the old game)
#   - reset(seed=...) does not use the seed

import socket
import sys
import time
from typing import Callable, Optional

import numpy as np

import footsiesgym

# default game settings
ENV_CONFIG = {
    "frame_skip": 4,
    "action_delay": 8,
    "max_t": 4000,
    "win_reward_scaling_coeff": 10.0,  # +10 for a win, -10 for a loss
    "guard_break_reward": 0.0,
    "use_reward_budget": False,
    "headless": True,
}


def default_platform() -> str:
    return "mac" if sys.platform == "darwin" else "linux"


class FootsiesVecEnv:
    """2N-slot view of a vectorized FootsiesEnv.

    reward_fn (optional) lets rewards.py replace the training reward:
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
        # with num_envs = 1 footsiesgym goes into single-env mode and the outputs
        # have another shape, so this wrapper only works with 2 or more games
        assert num_envs >= 2, "FootsiesVecEnv needs num_envs >= 2"
        cfg = dict(ENV_CONFIG)
        if env_config:
            cfg.update(env_config)
        cfg["num_envs"] = num_envs
        cfg["port"] = port
        self.port = port
        self.num_envs = num_envs
        self.num_slots = 2 * num_envs
        self.max_t = cfg["max_t"]
        self.reward_fn = reward_fn
        self.env = footsiesgym.make(config=cfg, platform=platform or default_platform())
        self.obs_dim = self.env.observation_space("p1").shape[0]
        # cast to a normal int, otherwise torch.load(weights_only=True) complains about numpy types
        self.n_actions = int(self.env.action_space("p1").n)

        self.t = np.zeros(num_envs, dtype=np.int64)  # steps done in each game
        self._ep_ret = np.zeros(num_envs)  # return of p1 in the current game (raw reward)
        self.finished = []  # stats of the finished games, emptied by pop_finished()
        self._obs = None

    @staticmethod
    def _stack(d):
        # put p1 and p2 one after the other -> array of length 2N
        return np.concatenate([d["p1"], d["p2"]], axis=0)

    def _wait_for_server(self, timeout: float = 120.0):
        # on my Mac the Unity server sometimes starts slower than the library
        # expects, so before reset() I wait until it accepts connections
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
        # first N actions are for p1, the other N for p2
        obs, rew, term, trunc, _ = self.env.step({"p1": actions[:N], "p2": actions[N:]})
        obs = self._stack(obs).astype(np.float32)
        rew = self._stack(rew).astype(np.float32)
        term = self._stack(term).astype(bool)
        trunc = self._stack(trunc).astype(bool)

        # statistics of the games, computed on the raw reward of p1
        # (not the shaped one, so they are the same for every reward variant)
        self._ep_ret += rew[:N]
        done = term[:N] | trunc[:N]
        for i in np.flatnonzero(done):
            p1_reward = rew[:N][i]
            if term[i]:
                # someone was knocked out: the sign of the reward says who won
                outcome = 1 if p1_reward > 0 else (-1 if p1_reward < 0 else 0)
            else:
                # timeout -> draw
                outcome = 0
            self.finished.append(
                {"len": int(self.t[i]) + 1, "truncated": bool(trunc[i] and not term[i]),
                 "p1_outcome": outcome, "p1_return": float(self._ep_ret[i])}
            )
            self._ep_ret[i] = 0

        # reward used for training (changed only if a reward_fn is given)
        train_rew = rew
        if self.reward_fn is not None:
            train_rew = self.reward_fn(self, self._obs, obs, rew, term, trunc).astype(np.float32)

        self.t += 1
        self.t[done] = 0  # the finished games start again from 0
        self._obs = obs
        return obs, train_rew, term, trunc
    
    # returns the stats of the games finished since the last call and clears them
    def pop_finished(self):
        out, self.finished = self.finished, []
        return out

    def close(self):
        self.env.close()
