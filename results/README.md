# results/

Evaluation results: small text files tracked by git (unlike `runs/`, which holds
checkpoints and training logs and is synced with rsync, see `RSYNC.md`).

`eval.csv` is written by `src/evaluate.py`: one line per agent-opponent pair,
appended at every evaluation (nothing is overwritten).

| Column | Meaning |
|---|---|
| `eval_id` | identifier of one evaluation (date-time + 4 random characters); all lines of one `evaluate.py` call share it |
| `timestamp` | when the evaluation was run |
| `label` | free text given with `--label` (empty if not given) |
| `agent` | fixed opponent name or checkpoint path, as passed to `--agent` |
| `run` | run folder of the checkpoint (e.g. `r0_s1`), empty for fixed opponents |
| `steps` | training steps of the checkpoint, empty for fixed opponents |
| `opponent` | fixed opponent name or checkpoint path |
| `games`, `wins`, `draws`, `losses` | outcomes of the agent; draws are timeouts |
| `win_rate`, `ci_low`, `ci_high` | wins / games and its 95% Wilson interval; draws count as non-wins, so also report the draw count |
| `seed` | seed of the evaluation |

Example:

```
uv run python src/evaluate.py --agent runs/r1_s1/final.pt --games 200 --label "r1 final"
```

Use `--out <file>` to write elsewhere and `--no-save` to only print the table.
An older CSV with fewer columns is upgraded automatically on the next write.
