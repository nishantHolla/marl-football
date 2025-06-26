from envs.abstract_football_env_v1 import AbstractFootballEnv_V1
import numpy as np

env = AbstractFootballEnv_V1(n_agents=5, field_width=800, render_mode="human")
observations, _ = env.reset()

while True:
	actions = {agent: np.random.randint(len(env.action_list)) for agent in env.agents}
	observations, rewards, terminations, truncations, infos = env.step(actions)
	env.render(actions)

	if all(terminations.values()):
		break
