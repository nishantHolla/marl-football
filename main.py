from envs.abstract_football_env import AbstractFootballEnv
import numpy as np

env = AbstractFootballEnv(num_of_agents=2, field_size=(120, 80))
observations = env.reset()
bp = env.ball

while True:
	actions = {agent: np.random.randint(10) for agent in env.agents}
	observations, rewards, terminations, truncations, infos = env.step(actions)
	env.render()

	if all(terminations.values()) or env.ball[0] != bp[0] or env.ball[1] != bp[1]:
		break
