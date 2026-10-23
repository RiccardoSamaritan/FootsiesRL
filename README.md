# FootsiesRL
A Reinforcement Learning approach to Footsies game

How to install the training environment, using [uv](https://docs.astral.sh/uv/):

### macOS/Linux

```
# Install uv (once). On macOS:
brew install uv
# or, on macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create .venv and install the locked dependencies (pyproject.toml + uv.lock)
uv sync

# Run any script inside the environment, without activating it
uv run python test.py
uv run python src/train_ppo.py
```

> **Note**: the project now uses uv only; `requirements.txt` and pip are gone. After pulling, install uv (see above) and run `uv sync` to recreate `.venv` (delete your old `.venv` first if it was created with `python3 -m venv`).
> `uv.lock` pins the exact versions of all packages so everyone gets the same environment: do not edit it by hand and always commit it.

To add a dependency use `uv add <package>`; commit both `pyproject.toml` and `uv.lock`.

### Windows

FootsiesGym only ships macOS and Linux game binaries, so the project does not run natively on Windows. Use WSL2 (Windows Subsystem for Linux), which gives you a real Linux environment:

```
# In PowerShell (as administrator), once; then reboot
wsl --install -d Ubuntu
```

Then open the Ubuntu terminal and work **inside Linux**, not in `/mnt/c/...` (the Windows filesystem is slow and breaks file permissions):

```
curl -LsSf https://astral.sh/uv/install.sh | sh   # then reopen the terminal
git clone <repo-url> ~/FootsiesRL
cd ~/FootsiesRL
uv sync
uv run python test.py
```

The headless environment (`test.py` and training) works this way. `visual_test.py` is written for the macOS windowed build and is not expected to work on WSL.

**IMPORTANT**: both in `test.py` and `visual_test.py` you have to set the correct platform depending on your OS. Use `"mac"` if you are on MacOS, or `"linux"` if you are on Linux. **Native Windows is not supported** (use WSL2 and `"linux"`, see above).
~~This disclaimer is for Giovanni, who will for sure text me and Alessio on Whatsapp because the code is not working on his machine (skill issue).~~
I don't need to be spoonfed.
