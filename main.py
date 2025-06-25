from envs.abstract_football_env import AbstractFootballEnv
import numpy as np

env = AbstractFootballEnv(num_of_agents=2, field_width=800, render_mode="human")
observations, _ = env.reset()
bp = env.ball

while True:
	actions = {agent: np.random.randint(10) for agent in env.agents}
	observations, rewards, terminations, truncations, infos = env.step(actions)
	env.render(actions)

	if all(terminations.values()):
		break
