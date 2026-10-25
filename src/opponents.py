"""Fixed opponents for evaluating agents on footsiesgym.

Each opponent implements the Opponent protocol and is built by name
through make_opponent(), so evaluation can sweep the difficulty scale.
"""
from typing import Callable, Protocol

import numpy as np


class Opponent(Protocol):
    """A fixed policy that plays one side of the match."""

    def reset(self, env_id: str = "default") -> None:
        """Clear per-episode state for the given env."""
        ...

    def act(self, agent_id: str, obs, info: dict) -> int:
        """Return a valid integer action for agent_id."""
        ...


class RandomOpponent:
    """Uniform random action over range(num_actions).

    Uses its own seeded numpy Generator. reset() does not reseed it:
    the random sequence continues across episodes.
    """

    def __init__(self, num_actions: int, seed: int | None = None):
        self.num_actions = num_actions
        self.rng = np.random.default_rng(seed)

    def reset(self, env_id: str = "default") -> None:
        pass

    def act(self, agent_id: str, obs, info: dict) -> int:
        return int(self.rng.integers(self.num_actions))


OPPONENTS: dict[str, Callable[..., Opponent]] = {
    "random": RandomOpponent,
}


def make_opponent(name: str, **kwargs) -> Opponent:
    """Build the opponent registered under name, passing kwargs to it."""
    if name not in OPPONENTS:
        raise ValueError(
            f"Unknown opponent {name!r}; available: {sorted(OPPONENTS)}")
    return OPPONENTS[name](**kwargs)
