from luxai_s2.env import LuxAI_S2

from agent import Agent
from helpers import interact, animate

if __name__ == "__main__":
    env = LuxAI_S2()  # create the environment object
    seed = 128
    obs = env.reset(seed=seed)  # resets an environment with a seed

    agents = {player: Agent(player, env.state.env_cfg) for player in env.agents}
    imgs = interact(env, agents, 30, seed)

    animate(imgs, fps=2, name="test")

