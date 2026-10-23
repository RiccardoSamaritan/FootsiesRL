import os
import socket
import subprocess
import sys
import time
import footsiesgym

PORT = 50051
BINARY = os.path.join(
    os.path.dirname(footsiesgym.__file__),
    "binaries", "footsies_binaries_windowed", "Contents", "MacOS", "FOOTSIES",
)
cmd = [BINARY, "-batchmode", "--grpc", "--port", str(PORT)]
if sys.platform == "darwin":
    cmd = ["arch", "-x86_64"] + cmd
# The windowed build needs ~10s to open its gRPC port (more than the library waits).
server = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for _ in range(120):
    try:
        socket.create_connection(("127.0.0.1", PORT), timeout=0.5).close()
        break
    except OSError:
        time.sleep(0.5)
else:
    server.terminate()
    raise RuntimeError("Footsies server did not start")

env = footsiesgym.make(
    config={"headless": False, "port": PORT},
    platform="mac",
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
    server.terminate()
