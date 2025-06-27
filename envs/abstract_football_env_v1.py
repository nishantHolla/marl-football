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
		self.goal_zone = {
			"top_left": np.array([left, top]),
			"top_right": np.array([right, top]),
			"bottom_left": np.array([left, bottom]),
			"bottom_right": np.array([right, bottom])
		}

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

		self.observation_spaces = {
			agent: Box(low=0, high=max(self.field_width, self.field_height), shape=(10 + 4 * (self.n_agents - 1),), dtype=np.float32)
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
		self.prev_goal_scored = getattr(self, "goal_scored", False)

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

		self._calculate_rewards()
		self.goal_scored = self._check_goal_scored()

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
		self.viewer.render(self.agent_pos, self.ball_pos, self.goal_zone["top_left"], self.goal_size)

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

	def _calculate_rewards(self):
		self.proximity_reward = {}
		self.ball_control_reward = {}
		self.positioning_reward = {}
		self.team_coordination_reward = {}
		self.movement_penalty = {}
		self.goal_proximity_bonus = {}

		goal_scored_reward = 100.0
		ball_progress_reward_scale = 10.0
		ball_proximity_reward_scale = 2.0
		team_coordination_reward_scale =1.0
		ball_control_reward_scale = 3.0
		positioning_reward_scale =1.5

		goal_scored = self._check_goal_scored()
		prev_goal_scored = self._was_goal_scored_prev_step()

		prev_ball_to_goal_dist = np.linalg.norm(self.goal_center - self.prev_ball_pos)
		curr_ball_to_goal_dist = np.linalg.norm(self.goal_center - self.ball_pos)
		ball_progress = prev_ball_to_goal_dist - curr_ball_to_goal_dist

		team_reward = 0.0
		team_reward += ball_progress_reward_scale * ball_progress

		# Individual agent rewards
		for agent in self.agents:
			agent_reward = team_reward  # Start with shared team reward

			curr_pos = self.agent_pos[agent]
			prev_pos = self.prev_agent_pos[agent]

			# 1. Ball proximity reward (encourages staying near ball)
			ball_dist = np.linalg.norm(curr_pos - self.ball_pos)
			prev_ball_dist = np.linalg.norm(prev_pos - self.prev_ball_pos)
			max_dist = np.sqrt(self.field_width**2 + self.field_height**2)

			# Normalized proximity reward (closer = higher reward)
			proximity_reward = ball_proximity_reward_scale * (1.0 - ball_dist / max_dist)
			agent_reward += proximity_reward
			self.proximity_reward[agent] = proximity_reward

			# 2. Ball control reward (for agents who influenced ball movement)
			ball_movement = np.linalg.norm(self.ball_vel)
			if self._agent_influenced_ball(agent):  # Check if agent pushed ball this step
				ball_control_reward = ball_control_reward_scale * ball_movement
			else:
				ball_control_reward = 0.0
			agent_reward += ball_control_reward
			self.ball_control_reward[agent] = ball_control_reward

			# 3. Positioning reward (encourage good field positioning)
			# Reward agents for being between ball and goal when ball is far from goal
			if curr_ball_to_goal_dist > self.field_width * 0.3:  # Ball is far from goal
				ball_to_goal_vec = self.goal_center - self.ball_pos
				ball_to_agent_vec = curr_pos - self.ball_pos

				# Check if agent is positioned between ball and goal
				if np.dot(ball_to_goal_vec, ball_to_agent_vec) > 0:
					alignment = np.dot(ball_to_goal_vec, ball_to_agent_vec) / (
						np.linalg.norm(ball_to_goal_vec) * np.linalg.norm(ball_to_agent_vec) + 1e-6
					)
					positioning_reward = positioning_reward_scale * max(0, alignment)
				else:
					positioning_reward = 0.0
			else:
				positioning_reward = 0.0

			agent_reward += positioning_reward
			self.positioning_reward[agent] = positioning_reward

			# 4. Team coordination reward (reward for maintaining good spacing)
			teammates = [other for other in self.agents if other != agent]
			if len(teammates) > 0:
				avg_teammate_dist = np.mean([
					np.linalg.norm(curr_pos - self.agent_pos[teammate])
					for teammate in teammates
				])

				# Optimal spacing (not too close, not too far)
				optimal_spacing = self.field_width * 0.2
				spacing_factor = 1.0 - abs(avg_teammate_dist - optimal_spacing) / optimal_spacing
				spacing_factor = max(0, spacing_factor)
				team_coordination_reward = team_coordination_reward_scale * spacing_factor
			else:
				team_coordination_reward = 0.0

			agent_reward += team_coordination_reward
			self.team_coordination_reward[agent] = team_coordination_reward

			# 5. Movement efficiency reward (small penalty for unnecessary movement)
			movement_dist = np.linalg.norm(curr_pos - prev_pos)
			if movement_dist < 1e-3:  # Agent didn't move
				movement_penalty = -0.1  # Small penalty for staying still
			else:
				movement_penalty = 0.0

			agent_reward += movement_penalty
			self.movement_penalty[agent] = movement_penalty

			# 6. Bonus for agents who contributed to goal
			if goal_scored and not prev_goal_scored:
				# Give extra reward to agent who last touched the ball
				if self._agent_influenced_ball(agent):
					agent_reward += 20.0

				# Give bonus to all agents based on proximity to ball when goal was scored
				goal_proximity_bonus = 10.0 * (1.0 - ball_dist / max_dist)
			else:
				goal_proximity_bonus = 0.0

			agent_reward += goal_proximity_bonus
			self.goal_proximity_bonus[agent] = goal_proximity_bonus

			self.rewards[agent] = agent_reward

	def _agent_influenced_ball(self, agent):
		agent_pos = self.agent_pos[agent]
		prev_agent_pos = self.prev_agent_pos[agent]

		# Check if agent was close enough to ball to influence it
		dist_to_ball = np.linalg.norm(agent_pos - self.ball_pos)
		prev_dist_to_ball = np.linalg.norm(prev_agent_pos - self.prev_ball_pos)

		influence_threshold = self.agent_size + self.ball_size + 5  # Slightly larger than collision

		return min(dist_to_ball, prev_dist_to_ball) < influence_threshold

	def _was_goal_scored_prev_step(self):
		return getattr(self, 'prev_goal_scored', False)

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

	def _check_goal_scored(self):
		left = self.goal_zone["top_left"][0].round()
		right = self.goal_zone["top_right"][0].round()
		top = self.goal_zone["top_left"][1].round()
		bottom = (self.goal_zone["bottom_right"][1] - self.ball_size).round()
		return (left <= self.ball_pos[0].round() <= right) and (top <= self.ball_pos[1].round() <= bottom)

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
			print(f"\trewards          :")
			print(f"\t\tproximity reward        : {self.proximity_reward[agent]}")
			print(f"\t\tball control reward     : {self.ball_control_reward[agent]}")
			print(f"\t\tpositioning reward      : {self.positioning_reward[agent]}")
			print(f"\t\tteam coordination reward: {self.team_coordination_reward[agent]}")
			print(f"\t\tmovement penalty        : {self.movement_penalty[agent]}")
			print(f"\t\tgoal proximity bonus    : {self.goal_proximity_bonus[agent]}")
			print(f"\t\ttotal                   : {self.rewards[agent]}")
			print()
			print(f"\tobservations     : {self.observations[agent].round(2)}")
			print(f"\tlength of obs    : {len(self.observations[agent])}")
			print(f"\tball inside goal : {self._check_goal_scored()}")

		print(f"ball position : {self.ball_pos.round(2)}")
		print(f"ball velocity : {self.ball_vel.round(2)}")

