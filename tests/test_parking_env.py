import imageio

from tl_search.envs.parking import AdversarialParkingEnv

animation_save_path: str = "out/plots/animation/parking_sample.gif"

env = AdversarialParkingEnv()

obs, _ = env.reset()
print(obs)

frames = []

for _ in range(50):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(obs)
    frames.append(env.render())
    if terminated:
        obs = env.reset()
        print(obs)

env.close()

imageio.mimsave(animation_save_path, frames, fps=4, loop=0)
