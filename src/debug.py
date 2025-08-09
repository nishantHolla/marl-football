import numpy as np
from football_env import FootballEnv


def debug(step_by_step=True):
    env = FootballEnv(n_agents=3, field_width=700, render_mode="human")
    observations, _ = env.reset()

    while True:
        actions = {
            agent: np.random.randint(len(env.action_list)) for agent in env.agents
        }
        observations, rewards, terminations, truncations, infos = env.step(actions)
        env.render(actions)

        if all(terminations.values()):
            break

        while step_by_step:
            n = input("q: quit, n: next > ")
            if n == "q":
                return
            elif n == "n":
                break
