from pettingzoo import ParallelEnv
from gymnasium.spaces import Box, Discrete
from render.viewer import Viewer
import numpy as np
import itertools

class AbstractFootballEnv_V1(ParallelEnv):
	metadata = {
		"render_modes": ["human", "none"],
		"name": "abstract_football_env_v1"
	}

	def __init__(self, render_mode="none", n_agents=2, field_width=800):
		self.render_mode = render_mode
		self.n_agents = n_agents
		self.field_width = field_width

		self.field_height = round(field_width * 2 / 3)
		self.agent_size = 20
		self.agent_speed = 3
		self.ball_size = 15
		self.repulsion_strength = 1.5
		self.ball_push_strength = 1.0
		self.ball_friction = 0.95

		self.viewer = None
		self._init_viewer()

		self.possible_agents = [f"agent_{i}" for i in range(self.n_agents)]
		self.agents = self.possible_agents[:]
		self.agent_pos = {}
		self.ball_pos = np.array([self.field_width // 2, self.field_height // 2], dtype=np.float32)
		self.ball_vel = np.array([0.0, 0.0], dtype=np.float32)

		self.goal_size = np.array([20, self.field_height // 4])
		self.goal_center = np.array([30, self.field_height // 2])
		top = (self.field_height // 2) - (self.goal_size[1] // 2)
		bottom = (self.field_height // 2) + (self.goal_size[1] // 2)
		left = 20
		right = 20 + self.goal_size[0]
		self.goal_zone = np.array([
			np.array([left, top]),
			np.array([right, top]),
			np.array([left, bottom]),
			np.array([right, bottom])
		])

		self.action_list = [
			{ "name": "MOVE_NORTH",      "motion": np.array([ 0, -1], dtype=np.float32) },
			{ "name": "MOVE_NORTH_EAST", "motion": np.array([ 1, -1], dtype=np.float32) },
			{ "name": "MOVE_EAST",       "motion": np.array([ 1,  0], dtype=np.float32) },
			{ "name": "MOVE_SOUTH_EAST", "motion": np.array([ 1,  1], dtype=np.float32) },
			{ "name": "MOVE_SOUTH",      "motion": np.array([ 0,  1], dtype=np.float32) },
			{ "name": "MOVE_SOUTH_WEST", "motion": np.array([-1,  1], dtype=np.float32) },
			{ "name": "MOVE_WEST",       "motion": np.array([-1,  0], dtype=np.float32) },
			{ "name": "MOVE_NORTH_WEST", "motion": np.array([-1, -1], dtype=np.float32) },
			{ "name": "STAY",            "motion": np.array([ 0,  0], dtype=np.float32) }
		]

		self.action_spaces = {
			agent: Discrete(len(self.action_list)) for agent in self.agents
		}

		self.obsertvation_spaces = {
			agent: Box(low=0, high=max(self.field_width, self.field_height), shape=(8 + 4 * (self.n_agents - 1),), dtype=np.float32)
			for agent in self.agents
		}

		self.reset()

	def action_space(self, agent):
		return self.action_spaces[agent]

	def observation_space(self, agent):
		return self.observation_spaces[agent]

	def reset(self, seed=None, options=None):
		self.agents = self.possible_agents[:]
		self.agent_pos = {}
		placed_pos = []
		for agent in self.agents:
			while True:
				pos = self._get_random_pos(self.agent_size)
				collides_with_agent = self._check_agent_agent_collision(pos, placed_pos)
				collides_with_ball = self._check_agent_ball_collision(pos)
				if not collides_with_ball and not collides_with_agent:
					self.agent_pos[agent] = pos
					placed_pos.append(pos)
					break

		self.ball_pos = np.array([self.field_width // 2, self.field_height // 2], dtype=np.float32)
		self.ball_vel = np.array([0.0, 0.0], dtype=np.float32)

		self.frame_count = 0
		self.observations = { agent: self._observe(agent) for agent in self.agents }
		self.infos = { agent: {} for agent in self.agents }

		return self.observations, self.infos

	def step(self, actions):
		self.frame_actions = actions
		self.rewards = { agent: 0.0 for agent in self.agents }
		self.terminations = { agent: False for agent in self.agents }
		self.truncations = { agent: False for agent in self.agents }
		self.infos = { agent: {} for agent in self.agents }
		self.prev_agent_pos = { agent: self.agent_pos[agent].copy() for agent in self.agents }
		self.prev_ball_pos = self.ball_pos[:]

		for agent, action_idx in actions.items():
			motion = self.action_list[action_idx]["motion"]
			norm = np.linalg.norm(motion)
			if norm > 0:
				motion = (motion / norm) * self.agent_speed
			else:
				motion = np.zeros(2, dtype=np.float32)

			self.agent_pos[agent] += motion

		for a1, a2 in itertools.combinations(self.agents, 2):
			pos1, pos2 = self.agent_pos[a1], self.agent_pos[a2]
			diff = pos1 - pos2
			dist = np.linalg.norm(diff)
			min_dist = 2 * self.agent_size

			if dist < min_dist and dist > 1e-5:
				repulse = self.repulsion_strength * (min_dist - dist) * (diff / dist)
				self.agent_pos[a1] += repulse / 2
				self.agent_pos[a2] -= repulse / 2

		for agent in self.agents:
			agent_to_ball = self.ball_pos - self.agent_pos[agent]
			dist = np.linalg.norm(agent_to_ball)
			min_dist = self.agent_size + self.ball_size

			if dist < min_dist > 1e-5:
				push_dir = agent_to_ball / dist
				force = self.ball_push_strength * (min_dist - dist)
				self.ball_vel += push_dir * force

		self.ball_pos += self.ball_vel
		self.ball_vel *= self.ball_friction

		self._clip_pos()

		# TODO: Calculate reward for the agent

		self.observations = { agent: self._observe(agent) for agent in self.agents }
		self.frame_count += 1

		return self.observations, self.rewards, self.terminations, self.truncations, self.infos

	def render(self, actions=None):
		if actions is None:
			actions = self.frame_actions

		if self.render_mode != "human":
			return

		if self.viewer is None:
			self._init_viewer()

		self._log_frame_info()
		self.viewer.render(self.agent_pos, self.ball_pos, self.goal_zone[0], self.goal_size)

	def close(self):
		if self.viewer is not None:
			self.viewer.close()

	def _observe(self, agent):
		field_size = np.array([self.field_width, self.field_height], dtype=np.float32)
		goal_dist = np.linalg.norm(self.goal_center - self.ball_pos)
		ball_dist = np.linalg.norm(self.agent_pos[agent] - self.ball_pos)
		teammates_pos = [ self.agent_pos[other] for other in self.agents if other != agent ]

		norm_self_pos = self.agent_pos[agent] / field_size
		norm_ball_pos = self.ball_pos / field_size
		norm_ball_vel = self.ball_vel / field_size
		norm_goal_dist = goal_dist / field_size
		norm_ball_dist = ball_dist / field_size
		norm_teammates = [ teammates_pos / field_size for teammates_pos in teammates_pos ]
		norm_dist_teammates = [
			np.linalg.norm(self.agent_pos[agent] - pos) / field_size
			for pos in teammates_pos
		]

		return np.concatenate([
			norm_self_pos,
			norm_ball_pos,
			norm_ball_vel,
			norm_goal_dist,
			norm_ball_dist,
			*norm_teammates,
			*norm_dist_teammates
		])

	def _check_agent_agent_collision(self, pos, other_positions, min_dist=None):
		if min_dist is None:
			min_dist = 2 * self.agent_size

		for other_pos in other_positions:
			if np.linalg.norm(pos - other_pos) < min_dist:
				return True

		return False

	def _check_agent_ball_collision(self, pos, min_dist=None):
		if min_dist is None:
			min_dist = self.agent_size + self.ball_size

		return np.linalg.norm(pos - self.ball_pos) < min_dist

	def _get_random_pos(self, size):
		x = np.random.uniform(size, self.field_width - size)
		y = np.random.uniform(size, self.field_height - size)
		return np.array([x, y], dtype=np.float32)

	def _clip_pos(self):
		for agent in self.agents:
			self.agent_pos[agent] = np.clip(self.agent_pos[agent], self.agent_size, [self.field_width - self.agent_size, self.field_height - self.agent_size])

		self.ball_pos = np.clip(self.ball_pos, self.ball_size, [self.field_width - self.ball_size, self.field_height - self.ball_size])


	def _init_viewer(self):
		if self.render_mode != "human":
			return

		if self.viewer is not None:
			return

		self.viewer = Viewer({
			"field_width": self.field_width,
			"field_height": self.field_height,
			"agent_size": self.agent_size,
			"ball_size": self.ball_size
		})

	def _log_frame_info(self):
		print(f"\n-- Frame {self.frame_count} --")
		for agent in self.agents:
			action = self.frame_actions[agent]

			print(f"{agent}:")
			print(f"\tprevious position: {self.prev_agent_pos[agent].round(2)}")
			print(f"\tcurrent position : {self.agent_pos[agent].round(2)}")
			print(f"\taction           : {self.action_list[action]["name"]}")
			print()
			# TODO: Log reward info
			print()
			print(f"\tobservations     : {self.observations[agent].round(2)}")

		print(f"ball position : {self.ball_pos.round(2)}")
		print(f"ball velocity : {self.ball_vel.round(2)}")

