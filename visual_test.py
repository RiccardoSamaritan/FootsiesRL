import socket
import sys
import time

import footsiesgym

PLATFORM = "mac" if sys.platform == "darwin" else "linux"

# The library launches the windowed game itself (and handles code signing on mac).
env = footsiesgym.make(config={"headless": False}, platform=PLATFORM)

# The windowed build can take longer to start than the library waits:
# wait until the game server accepts connections before calling reset().
for _ in range(120):
    try:
        socket.create_connection(("127.0.0.1", env.game.port), timeout=0.5).close()
        break
    except OSError:
        time.sleep(0.5)
else:
    env.close()
    raise RuntimeError("Footsies server did not start")

obs, infos = env.reset()

STEP_DT = 4 / 60
episodes = 0
try:
    while episodes < 5:
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rewards, terminateds, truncateds, infos = env.step(actions)
        time.sleep(STEP_DT)
        if terminateds["p1"] or truncateds["p1"]:
            episodes += 1
            obs, infos = env.reset()
finally:
    env.close()