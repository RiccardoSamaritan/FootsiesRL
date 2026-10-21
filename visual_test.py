import time
import footsiesgym
import sys
PLATFORM = "mac" if sys.platform == "darwin" else "linux"

env = footsiesgym.make(
    config={"headless": False, "port": 50051},
    platform=PLATFORM, # IMPORTANT: "mac" or "linux" depending on your platform
    launch_binaries=False,
)
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
