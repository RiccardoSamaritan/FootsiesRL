import footsiesgym

env = footsiesgym.make(platform="mac") # IMPORTANT: "mac" or "linux" depending on your platform
obs, infos = env.reset() 

for _ in range(1000):
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rewards, terminateds, truncateds, infos = env.step(actions)
    if terminateds["p1"] or truncateds["p1"]:
        obs, infos = env.reset()

print({a: o.shape for a, o in obs.items()})
env.close()
