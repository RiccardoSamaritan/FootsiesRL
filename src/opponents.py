"""Fixed opponents for evaluating agents on footsiesgym.

Each opponent implements the Opponent protocol and is built by name
through make_opponent(), so evaluation can sweep the difficulty scale.
"""
from typing import Callable, Protocol

import numpy as np
from footsiesgym.footsies.game.constants import ACTION_TO_BITS, EnvActions
from footsiesgym.footsies.game.footsies_bot import FootsiesBot
from footsiesgym.footsies.game.footsies_game import FootsiesGame


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

    Uses its own seeded numpy Generator.
    """

    def __init__(self, num_actions: int, seed: int | None = None):
        self.num_actions = num_actions
        self.rng = np.random.default_rng(seed)

    def reset(self, env_id: str = "default") -> None:
        pass

    def act(self, agent_id: str, obs, info: dict) -> int:
        return int(self.rng.integers(self.num_actions))


class ConstantActionOpponent:
    """Always plays the same action."""

    def __init__(self, action: int):
        self.action = action

    def reset(self, env_id: str = "default") -> None:
        pass

    def act(self, agent_id: str, obs, info: dict) -> int:
        return self.action


def game_action_to_env_action(action: int, is_player_1: bool) -> int:
    """Convert an absolute GameAction to the player-relative EnvAction.

    Exact inverse of FootsiesGame.action_to_bits. Covers only the six base
    actions.
    """
    bits = ACTION_TO_BITS[action]
    for env_action in range(EnvActions.FORWARD_ATTACK + 1):
        if FootsiesGame.action_to_bits(env_action, is_player_1) == bits:
            return env_action
    raise ValueError(f"No EnvAction maps to GameAction {action}")


_FIGHT_STATE_KEYS = (
    "distance_x", "is_opponent_damage", "is_opponent_guard_break",
    "is_opponent_blocking", "is_opponent_normal_attack",
    "is_opponent_special_attack", "is_facing_right",
)


class FootsiesBotOpponent:
    """Rule-based bot: footsiesgym's rewrite of the original game's BattleAI.

    Reads the fight state from info, frame_skip must match the env's.
    The bot uses numpy's global legacy RNG; with a seed, a private
    RandomState is swapped in only around the bot call, so the caller's
    global state is untouched.
    """

    def __init__(self, frame_skip: int = 4, seed: int | None = None):
        self._bot = FootsiesBot(frame_skip=frame_skip)
        self._rng_state = (None if seed is None
                           else np.random.RandomState(seed).get_state())
        self.env_id = "default"

    def reset(self, env_id: str = "default") -> None:
        self.env_id = env_id
        self._bot.move_queues.pop(env_id, None)
        self._bot.attack_queues.pop(env_id, None)
        self._bot.override_active = False

    def act(self, agent_id: str, obs, info: dict) -> int:
        if not info or any(k not in info for k in _FIGHT_STATE_KEYS):
            raise ValueError(
                "FootsiesBotOpponent needs the fight state in info: create "
                "the env with return_fight_state_in_infos=True and "
                "num_envs=1 (vectorized and RLlib-wrapped envs return "
                "empty infos).")
        if self._rng_state is None:
            action = self._bot.get_next_input(self.env_id, agent_id, info)
        else:
            saved = np.random.get_state()
            np.random.set_state(self._rng_state)
            try:
                action = self._bot.get_next_input(self.env_id, agent_id, info)
            finally:
                self._rng_state = np.random.get_state()
                np.random.set_state(saved)
        is_player_1 = agent_id.split("_")[0] == "p1"
        return game_action_to_env_action(action, is_player_1)


OPPONENTS: dict[str, Callable[..., Opponent]] = {
    # Every factory takes num_actions, seed and frame_skip by name and
    # explicitly ignores the ones it does not use.
    "random": lambda num_actions, seed, frame_skip: RandomOpponent(
        num_actions, seed),
    "noop": lambda num_actions, seed, frame_skip: ConstantActionOpponent(
        EnvActions.NONE),
    "back": lambda num_actions, seed, frame_skip: ConstantActionOpponent(
        EnvActions.BACK),
    "footsies_bot": lambda num_actions, seed, frame_skip: FootsiesBotOpponent(
        frame_skip, seed),
}


def make_opponent(name: str, *, num_actions: int = 6,
                  seed: int | None = None, frame_skip: int = 4) -> Opponent:
    """Build the opponent registered under name.

    Parameters are explicit so that a misspelled keyword raises a
    TypeError instead of silently falling back to a default seed.
    """
    if name not in OPPONENTS:
        raise ValueError(
            f"Unknown opponent {name!r}; available: {sorted(OPPONENTS)}")
    return OPPONENTS[name](num_actions=num_actions, seed=seed,
                           frame_skip=frame_skip)
