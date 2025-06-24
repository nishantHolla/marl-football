from pettingzoo import ParallelEnv
from gymnasium.spaces import Box, Discrete
import numpy as np

class AbstractFootballEnv(ParallelEnv):
	metadata = {
		"render_modes": ["human"],
		"name": "abstract_footbal_env_v0"
	}

	def __init__(self, max_cycles=5, num_of_agents=2, field_size=(300, 200)):
		self.max_cycles = max_cycles
		self.num_of_agents = num_of_agents
		self.field_width, self.field_height = field_size
		self.agent_width = 0

		self.agents = [f"robot_{i}" for i in range(num_of_agents)]
		self.positions = {}
		self.ball = np.array([field_size[0] // 2, field_size[1] // 2], dtype=np.float32)
		self.ball_velocity = np.array([0.0, 0.0], dtype=np.float32)

		self.goal_1 = [0, self.field_width // 2]
		self.action_list = [
			{ "name": "MOVE_NORTH",      "motion": np.array([ 0,  -1], dtype=np.float32)   },
			{ "name": "MOVE_NORTH_EAST", "motion": np.array([ 1,  -1], dtype=np.float32)   },
			{ "name": "MOVE_EAST",       "motion": np.array([ 1,   0], dtype=np.float32)   },
			{ "name": "MOVE_SOUTH_EAST", "motion": np.array([ 1,   1], dtype=np.float32)   },
			{ "name": "MOVE_SOUTH",      "motion": np.array([ 0,   1], dtype=np.float32)   },
			{ "name": "MOVE_SOUTH_WEST", "motion": np.array([-1,   1], dtype=np.float32)   },
			{ "name": "MOVE_WEST",       "motion": np.array([-1,   0], dtype=np.float32)   },
			{ "name": "MOVE_NORTH_WEST", "motion": np.array([-1,  -1], dtype=np.float32)   },
			{ "name": "STAY",            "motion": np.array([ 0,   0], dtype=np.float32)   },
			{ "name": "KICK",            "motion": np.array([ 0,   0], dtype=np.float32)   }
		]

		self.action_space = Discrete(10)
		self.observation_space = Box(low=0, high=max(self.field_width, self.field_height), shape=(4,), dtype=np.float32)
		self.cycle_count = 0

		self.reset()


	def reset(self, seed=None, options=None):
		self.agents = [f"robot_{i}" for i in range(self.num_of_agents)]
		self.positions = { agent: self._get_random_postion() for agent in self.agents }
		self.ball = np.array([self.field_width // 2, self.field_height // 2], dtype=np.float32)
		self.ball_velocity = np.array([0.0, 0.0], dtype=np.float32)
		observations = {agent: self._observe(agent) for agent in self.agents}

		self.cycle_count = 0

		return observations


	def step(self, actions):
		self.rewards = {agent: 0.0 for agent in self.agents}
		self.terminations = {agent: False for agent in self.agents}
		self.truncations = {agent: False for agent in self.agents}
		self.infos = {agent: {} for agent in self.agents}

		for agent, action in actions.items():
			if action < 9:
				self.positions[agent] += self.action_list[action]["motion"]
			elif action == 9 and self._has_kicked_ball(agent):
				kick_dir = self.ball - self.positions[agent]
				if np.linalg.norm(kick_dir) > 0:
					self.ball_velocity = kick_dir / np.linalg.norm(kick_dir) * 5.0
					self.rewards[agent] += 1.0

		self.ball += self.ball_velocity
		self.ball_velocity *= 0.9

		self._clip_positions()

		if self.ball[0] < 0:
			for agent in self.agents:
				self.rewards[agent] += 10
				self.terminations[agent] = True

		self.observations = {agent: self._observe(agent) for agent in self.agents}
		self.cycle_count += 1
		return self.observations, self.rewards, self.terminations, self.truncations, self.infos


	def render(self):
		print(f"\n-- Frame {self.cycle_count}--")
		for agent in self.agents:
			print(f"{agent} =>\n\tposition: {self.positions[agent].astype(int)}\n\treward: {self.rewards[agent]}")
		print(f"ball: {self.ball.round(1)}")


	def close(self):
		pass


	def _clip_positions(self):
		for agent in self.agents:
			self.positions[agent] = np.clip(self.positions[agent], [0, 0], [self.field_width, self.field_height])
		self.ball = np.clip(self.ball, [0, 0], [self.field_width, self.field_height])


	def _observe(self, agent):
		return np.concatenate([self.positions[agent], self.ball])


	def _get_random_postion(self):
		return np.array([
			np.random.randint(self.agent_width, self.field_width - self.agent_width),
			np.random.randint(self.agent_width, self.field_width - self.agent_width),
		], dtype=np.float32)


	def _has_kicked_ball(self, agent):
		return (np.linalg.norm(self.positions[agent] - self.ball) < 5.0)


def env():
	return AbstractFootballEnv()
