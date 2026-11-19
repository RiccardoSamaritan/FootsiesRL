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
  for replacing the training reward (used by `rewards.py` for R2 and R3).
- `rewards.py`: the training rewards R0–R3, chosen with `train.py --reward`
  (R0 terminal ±10, R1 guard break with budget, R2 potential-based shaping on
  the guard, R3 win scaled by game length).
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
- `evaluate.py`: plays N games against each fixed opponent, sides alternating,
  win/draw/loss with a 95% confidence interval. The agent can be a fixed
  opponent or a trained checkpoint.
- `policy.py`: wraps a trained checkpoint as an Opponent, sampling actions
  (never argmax).
- `head_to_head.py`: checkpoints play each other, in shards; writes
  `results/h2h_<pool>*.csv` and the score matrix.
- `best_response.py`: trains a PPO learner against a frozen checkpoint to
  estimate its exploitability; writes `runs_br/` and `results/br_<target>.csv`.
- `diagnostics.py`: action distribution and game statistics of a checkpoint;
  writes `results/diag_<run>.csv`.
- `make_tables.py`: Markdown tables of `results/summary_heuristics.md` from
  `results/summary_final.csv`.

## Usage

```
# quick smoke test
uv run python src/train.py --total-steps 20000 --num-envs 4 --run-name smoke

# full run as in the experiments (48 games, rollout 64, 20M steps)
uv run python src/train.py --run-name r0_s1 --reward r0 --seed 1 --total-steps 20000000
```

Useful options: `--reward`, `--gae-lambda`, `--gamma`, `--lr`, `--ent-coef`,
`--ent-decay`, `--adv-std-floor`, `--seed`, `--port`, `--checkpoint-every`,
`--resume`, `--wandb`.

## Notes

- One "step" is one game step of one game instance. An update uses
  `num_envs * rollout` steps, i.e. twice as many agent samples (two players).
- The vectorized environment resets finished games automatically and returns
  the reset observation, so the final observation of a game is lost. For
  timeouts (`max_t`) `compute_gae` approximates the bootstrap value with the
  value of the previous observation.
- The training reward is chosen with `--reward` (default `r0`, terminal
  win/loss reward ±10). Evaluation always uses the game outcome; see the file
  list above.