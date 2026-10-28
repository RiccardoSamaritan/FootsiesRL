# results/

Evaluation results, small text files that are tracked by git (unlike `runs/`,
which holds checkpoints and training logs and is synced with rsync, see
`RSYNC.md`).

- `eval.csv`: written by `src/evaluate.py`, one line per agent-opponent pair
  (appended at every evaluation, nothing is overwritten). Columns: `timestamp`,
  `agent` (fixed opponent name or checkpoint path), `opponent`, `games`, `wins`,
  `draws`, `losses`, `win_rate`, `ci_low`, `ci_high` (95% Wilson interval), `seed`.
  `win_rate` counts wins only, so draws (timeouts) are in neither wins nor
  losses: report the draw count as well.

Use `--out <file>` to write elsewhere, or `--no-save` to only print the table.
