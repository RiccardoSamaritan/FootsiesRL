# src/

Code for training a Footsies agent with our own PPO in self-play.
Run everything from the repository root with `uv run python ...`.

## Files

- `env.py`: `FootsiesVecEnv`, a vectorized wrapper around FootsiesGym. N game
  instances run in one Unity server and are exposed as 2N "slots" (first N are
  the p1 side, last N the p2 side). All slots share one network, so this is
  self-play. It returns observations, rewards and `terminated`/`truncated`
  flags, collects per-episode statistics and waits for the game server to be
  ready before the first `reset()`. It also has an optional `reward_fn` hook
  for replacing the training reward (not used yet).
- `ppo.py`: the algorithm, with no dependency on Footsies. `Agent` (actor and
  critic MLPs, 2x256 ReLU), `compute_gae` (handles termination and truncation
  differently), `ppo_update` (clipped surrogate, value loss, entropy bonus,
  diagnostics), `PPOConfig` (paper hyperparameters), checkpoint save/load.
  Places where bias or variance enter the estimator are marked with
  `BIAS/VARIANCE` comments.
- `train.py`: the training loop. Collects rollouts from `env.py`, computes
  advantages with GAE, updates the network with `ppo.py`, writes
  `runs/<run-name>/log.csv` and checkpoints (`ckpt_<steps>.pt`, `final.pt`).
- `opponents.py`: fixed opponents (random, no-op, back, rule-based
  FootsiesBot) for evaluation.

## Usage

```
# quick smoke test
uv run python src/train.py --total-steps 20000 --num-envs 4 --run-name smoke

# full run with the paper settings (48 games, rollout 64, 25M steps)
uv run python src/train.py --run-name lam0.95_s1
```

Useful options: `--gae-lambda`, `--gamma`, `--lr`, `--ent-coef`, `--ent-decay`,
`--seed`, `--port`, `--checkpoint-every`, `--wandb`.

## Notes

- One "step" is one game step of one game instance. An update uses
  `num_envs * rollout` steps, i.e. twice as many agent samples (two players).
- The vectorized environment resets finished games automatically and returns
  the reset observation, so the final observation of a game is lost. For
  timeouts (`max_t`) `compute_gae` approximates the bootstrap value with the
  value of the previous observation.
- Training reward is the terminal win/loss reward (+-10). Evaluation scripts
  are not written yet.
