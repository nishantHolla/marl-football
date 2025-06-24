from pettingzoo import ParallelEnv
from gymnasium.spaces import Box, Discrete
from render.viewer import Viewer
import numpy as np

class AbstractFootballEnv(ParallelEnv):
	metadata = {
		"render_modes": ["human"],
		"name": "abstract_footbal_env_v0"
	}

	def __init__(self, max_cycles=5, num_of_agents=2, field_width=1200):
		self.max_cycles = max_cycles
		self.num_of_agents = num_of_agents

		self.field_width = field_width
		self.field_height = round(field_width * 2 / 3)
		self.agent_size = 30
		self.agent_speed = 3
		self.ball_size = 20

		self.viewer = Viewer({
			"field_width": self.field_width,
			"field_height": self.field_height,
			"agent_size": self.agent_size,
			"ball_size": self.ball_size
		})

		self.agents = [f"robot_{i}" for i in range(num_of_agents)]
		self.positions = {}
		self.ball = np.array([self.field_width // 2, self.field_height // 2], dtype=np.float32)
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
				self.positions[agent] += self.action_list[action]["motion"] * self.agent_speed
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
		self._log_frame_info()
		self.viewer.render(self.positions, self.ball)


	def close(self):
		self.viewer.close()


	def _log_frame_info(self):
		print(f"\n-- Frame {self.cycle_count}--")
		for agent in self.agents:
			print(f"{agent} =>\n\
		position: {self.positions[agent].astype(int)}\n\
		reward: {self.rewards[agent]}")

		print(f"ball: {self.ball.round(1)}")


	def _clip_positions(self):
		for agent in self.agents:
			self.positions[agent] = self._clip_value(self.positions[agent], self.agent_size)
		self.ball = self._clip_value(self.ball, self.ball_size)


	def _clip_value(self, position, size):
		return np.clip(position, [size, size], [self.field_width - size, self.field_height - size])


	def _observe(self, agent):
		return np.concatenate([self.positions[agent], self.ball])


	def _get_random_postion(self):
		return np.array([
			np.random.randint(self.agent_size, self.field_width - self.agent_size),
			np.random.randint(self.agent_size, self.field_width - self.agent_size),
		], dtype=np.float32)


	def _has_kicked_ball(self, agent):
		return (np.linalg.norm(self.positions[agent] - self.ball) < (self.ball_size + self.agent_size))


def env():
	return AbstractFootballEnv()
