# Moving results between Orfeo and your computer (rsync)

Training runs are produced on the Orfeo cluster, while analysis, evaluation and
the windowed game are done on a personal computer. Results are **not** moved
with git: `runs/` and `slurm/logs/` are in `.gitignore` on purpose, so the
repository stays code-only. Use `rsync` instead.

## What rsync does

`rsync` copies files between two machines over SSH, like `scp`, but it compares
the two folders first and transfers **only what is new or changed**. The first
run copies everything; later runs copy only the new runs or the files that grew.
Running it twice is safe: files that already match are skipped.

## Basic usage

Run these commands **on your computer**, from the root of the repository.
Replace `<user>` and `<orfeo-host>` with the account and address you use for
`ssh` to log in to Orfeo.

```
# everything: logs, configs, checkpoints
rsync -av --progress <user>@<orfeo-host>:~/FootsiesRL/runs/ ./runs/

# only small files (no intermediate checkpoints): enough for plots
rsync -av --exclude='ckpt_*.pt' <user>@<orfeo-host>:~/FootsiesRL/runs/ ./runs/

# a single run
rsync -av <user>@<orfeo-host>:~/FootsiesRL/runs/r0_s1/ ./runs/r0_s1/

# SLURM job logs
rsync -av <user>@<orfeo-host>:~/FootsiesRL/slurm/logs/ ./slurm/logs/
```

Useful flags: `-a` keeps the folder structure and timestamps, `-v` lists the
files, `--progress` shows the transfer, `-n` (dry run) shows what *would* be
copied without copying anything.

## Rules to avoid mistakes

- **Trailing slash matters.** `runs/` (with slash) means "the contents of
  runs"; `runs` (without) would create `runs/runs/`. Keep the slash on both
  paths, as in the examples.
- **Direction.** The examples copy *from* Orfeo *to* the computer. Swapping the
  two arguments would copy the other way; check with `-n` first if unsure.
- **Do not add `--delete`** unless you want rsync to remove local files that no
  longer exist on Orfeo.
- **Size.** A checkpoint is about 0.7 MB; a 25M-step run keeps about 25 of
  them, so roughly 20 MB per run. Use the `--exclude='ckpt_*.pt'` form when you
  only need `log.csv` and `config.json`.

## If Orfeo needs a jump host or VPN

Add the same option you use with `ssh`, for example:

```
rsync -av -e "ssh -J <user>@<jump-host>" <user>@<orfeo-host>:~/FootsiesRL/runs/ ./runs/
```

or define the host once in `~/.ssh/config` and use its alias in place of
`<user>@<orfeo-host>`.

## Cleaning up on Orfeo

Old test runs and job logs only take space. From the repository folder on
Orfeo (check with `ls` first, `rm` cannot be undone):

```
ls runs slurm/logs
rm -r runs/<test-run-name>
rm slurm/logs/<old-log>.out
```

Do not use `git clean -fdX`: it removes every ignored file, including `.venv`.
