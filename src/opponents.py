# opponents.py
# The fixed opponents we use to evaluate the agents: random, noop, back and the
# rule-based bot of the game (FootsiesBot).

from typing import Callable, Protocol

import numpy as np
from footsiesgym.footsies.game.constants import ACTION_TO_BITS, EnvActions
from footsiesgym.footsies.game.footsies_bot import FootsiesBot
from footsiesgym.footsies.game.footsies_game import FootsiesGame


class Opponent(Protocol):
    """
    Interface of a fixed policy that plays one side of the match.
    """

    def reset(self, env_id: str = "default") -> None:
        """
        Clears the state of the episode for the given env.
        """
        ...

    def act(self, agent_id: str, obs, info: dict) -> int:
        """
        Returns a valid action (an integer) for agent_id.
        """
        ...


class RandomOpponent:
    """
    Chooses an action uniformly at random in range(num_actions).

    It has its own numpy Generator with its own seed, so it does not depend on
    the global random state.
    """

    def __init__(self, num_actions: int, seed: int | None = None):
        self.num_actions = num_actions
        self.rng = np.random.default_rng(seed)

    def reset(self, env_id: str = "default") -> None:
        pass

    def act(self, agent_id: str, obs, info: dict) -> int:
        return int(self.rng.integers(self.num_actions))


class ConstantActionOpponent:
    """
    Always plays the same action (we use it for noop and back).
    """

    def __init__(self, action: int):
        self.action = action

    def reset(self, env_id: str = "default") -> None:
        pass

    def act(self, agent_id: str, obs, info: dict) -> int:
        return self.action


def game_action_to_env_action(action: int, is_player_1: bool) -> int:
    """
    Converts an absolute GameAction (what the bot returns) into the EnvAction of the player.

    The game actions are absolute (left/right), while the env actions are relative to the
    player (forward/back), so the same game action means different things for p1 and p2.
    This is exactly the inverse of FootsiesGame.action_to_bits, and covers only the six
    base actions.
    """
    bits = ACTION_TO_BITS[action]
    # we try every env action and keep the one that gives the same bits
    for env_action in range(EnvActions.FORWARD_ATTACK + 1):
        if FootsiesGame.action_to_bits(env_action, is_player_1) == bits:
            return env_action
    raise ValueError(f"No EnvAction maps to GameAction {action}")


# information about the fight that the bot needs to find in info
_FIGHT_STATE_KEYS = (
    "distance_x", "is_opponent_damage", "is_opponent_guard_break",
    "is_opponent_blocking", "is_opponent_normal_attack",
    "is_opponent_special_attack", "is_facing_right",
)


class FootsiesBotOpponent:
    """Rule-based bot: the version in footsiesgym of the BattleAI of the original game.

    It reads the state of the fight from info, and its frame_skip has to be the same as
    the env's. The bot uses the global (legacy) random generator of numpy: when we give
    a seed, we put a private RandomState in place only during the call to the bot, so the
    global random state of the rest of the program is not changed.
    """

    def __init__(self, frame_skip: int = 4, seed: int | None = None):
        self._bot = FootsiesBot(frame_skip=frame_skip)
        self._rng_state = (None if seed is None
                           else np.random.RandomState(seed).get_state())
        self.env_id = "default"

    def reset(self, env_id: str = "default") -> None:
        # we empty the queues of planned moves of the bot for this env
        self.env_id = env_id
        self._bot.move_queues.pop(env_id, None)
        self._bot.attack_queues.pop(env_id, None)
        self._bot.override_active = False

    def act(self, agent_id: str, obs, info: dict) -> int:
        # the bot only works with the single env, where info contains the fight state
        if not info or any(k not in info for k in _FIGHT_STATE_KEYS):
            raise ValueError(
                "FootsiesBotOpponent needs the fight state in info: create "
                "the env with return_fight_state_in_infos=True and "
                "num_envs=1 (vectorized and RLlib-wrapped envs return "
                "empty infos).")
        if self._rng_state is None:
            action = self._bot.get_next_input(self.env_id, agent_id, info)
        else:
            # swap in our random state, call the bot, then put back the global one
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
    # every function receives num_actions, seed and frame_skip by name
    # and simply ignores the ones it does not need
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
    """Creates the opponent registered with the given name.

    The parameters must be passed by name: this way a typo in a keyword gives a
    TypeError, instead of silently using the default seed.
    """
    if name not in OPPONENTS:
        raise ValueError(
            f"Unknown opponent {name!r}; available: {sorted(OPPONENTS)}")
    return OPPONENTS[name](num_actions=num_actions, seed=seed,
                           frame_skip=frame_skip)
