from pettingzoo import ParallelEnv
from gymnasium.spaces import Box, Discrete
from render.viewer import Viewer
import numpy as np


class AbstractFootballEnv_V1(ParallelEnv):
    metadata = {"render_modes": ["human", "none"], "name": "abstract_football_env_v1"}

    def __init__(self, render_mode="none", n_agents=2, field_width=800):
        self.render_mode = render_mode
        self.n_agents = n_agents
        self.field_width = field_width

        self.field_height = round(field_width * 2 / 3)
        self.agent_size = 20
        self.agent_speed = 3
        self.ball_size = 15
        self.ball_push_strength = 1.0
        self.ball_friction = 0.95
        self.kick_range = self.agent_size + self.ball_size + 3

        self.possible_agents = [f"agent_{i}" for i in range(self.n_agents)]
        self.agents = self.possible_agents[:]
        self.agent_pos = {}
        self.ball_pos = np.array(
            [self.field_width // 2, self.field_height // 2], dtype=np.float32
        )
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
            "bottom_right": np.array([right, bottom]),
        }

        self.action_list = [
            {"name": "MOVE_NORTH", "motion": np.array([0, -1], dtype=np.float32)},
            {"name": "MOVE_NORTH_EAST", "motion": np.array([1, -1], dtype=np.float32)},
            {"name": "MOVE_EAST", "motion": np.array([1, 0], dtype=np.float32)},
            {"name": "MOVE_SOUTH_EAST", "motion": np.array([1, 1], dtype=np.float32)},
            {"name": "MOVE_SOUTH", "motion": np.array([0, 1], dtype=np.float32)},
            {"name": "MOVE_SOUTH_WEST", "motion": np.array([-1, 1], dtype=np.float32)},
            {"name": "MOVE_WEST", "motion": np.array([-1, 0], dtype=np.float32)},
            {"name": "MOVE_NORTH_WEST", "motion": np.array([-1, -1], dtype=np.float32)},
            {"name": "STAY", "motion": np.array([0, 0], dtype=np.float32)},
            {"name": "KICK", "motion": np.array([0, 0], dtype=np.float32)},
        ]

        self.action_spaces = {
            agent: Discrete(len(self.action_list)) for agent in self.agents
        }

        self.observation_spaces = {
            agent: Box(
                low=0,
                high=max(self.field_width, self.field_height),
                shape=(10 + 4 * (self.n_agents - 1),),
                dtype=np.float32,
            )
            for agent in self.agents
        }

        self.viewer = None
        self._init_viewer()

        self.reset()

    def action_space(self, agent):
        return self.action_spaces[agent]

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def reset(self, seed=None, options=None):
        self.last_ball_contact_agent = None
        self.last_contact_rewarded_agent = None
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

        self.ball_pos = np.array(
            [self.field_width // 2, self.field_height // 2], dtype=np.float32
        )
        self.ball_vel = np.array([0.0, 0.0], dtype=np.float32)

        self.frame_count = 0
        self.observations = {agent: self._observe(agent) for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        return self.observations, self.infos

    def step(self, actions):
        self.frame_actions = actions
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.prev_agent_pos = {
            agent: self.agent_pos[agent].copy() for agent in self.agents
        }
        self.prev_ball_pos = self.ball_pos[:]

        for agent in self.agents:
            if agent not in actions:
                continue

            action_idx = actions[agent]
            motion = self.action_list[action_idx]["motion"]

            # Calculate the intended new position
            intended_pos = self.agent_pos[agent] + motion * self.agent_speed

            # Get positions of other agents (excluding current agent)
            other_agent_positions = [
                self.agent_pos[other] for other in self.agents if other != agent
            ]

            # Check if intended position would cause collisions
            agent_collision = self._check_agent_agent_collision(
                intended_pos, other_agent_positions
            )
            ball_collision = self._check_agent_ball_collision(intended_pos)

            if not agent_collision and not ball_collision:
                # No collision - move to intended position
                self.agent_pos[agent] = intended_pos
            else:
                # Collision detected - try to move as much as possible
                self.agent_pos[agent] = self._resolve_collision_movement(
                    agent, motion * self.agent_speed, other_agent_positions
                )

            if self.action_list[action_idx]["name"] == "KICK":
                self._handle_kick_action(agent)

        goal_y_min = self.goal_zone["top_left"][1]
        goal_y_max = self.goal_zone["bottom_left"][1]
        goal_x_start = self.goal_zone["top_left"][0]

        if (
            goal_y_min <= self.ball_pos[1] <= goal_y_max
            and self.ball_pos[0] <= goal_x_start
        ):
            self.ball_pos[0] = self.goal_zone["top_left"][0]
            self.ball_vel = np.zeros(2, dtype=np.float32)
        else:
            self.ball_pos += self.ball_vel

        self.ball_vel *= self.ball_friction
        self._clip_pos()

        self.goal_scored = self._check_goal_scored()
        self._calculate_rewards()

        self.observations = {agent: self._observe(agent) for agent in self.agents}
        self.frame_count += 1

        return (
            self.observations,
            self.rewards,
            self.terminations,
            self.truncations,
            self.infos,
        )

    def _resolve_collision_movement(
        self, agent, intended_movement, other_agent_positions
    ):
        """
        Try to move the agent as much as possible in the intended direction
        without causing collisions with other agents or the ball.
        """
        current_pos = self.agent_pos[agent]

        # Try to move step by step (10 steps) to see how far we can go
        max_steps = 10
        best_pos = current_pos.copy()

        for step in range(1, max_steps + 1):
            # Calculate partial movement
            partial_movement = intended_movement * (step / max_steps)
            test_pos = current_pos + partial_movement

            # Check for collisions at this test position
            agent_collision = self._check_agent_agent_collision(
                test_pos, other_agent_positions
            )
            ball_collision = self._check_agent_ball_collision(test_pos)

            if not agent_collision and not ball_collision:
                # This position is valid, update best position
                best_pos = test_pos.copy()
            else:
                # Collision detected, stop here and return the last valid position
                break

        return best_pos

    def _handle_kick_action(self, agent):
        """
        Handle the KICK action by pushing the ball if the agent is close enough.
        The ball is pushed in the direction from the agent to the ball.
        """
        agent_pos = self.agent_pos[agent]

        # Check if agent is close enough to kick the ball
        distance_to_ball = np.linalg.norm(agent_pos - self.ball_pos)
        kick_range = self.agent_size + self.ball_size + 3

        if distance_to_ball <= self.kick_range:
            # Calculate direction from agent to ball
            direction = self.ball_pos - agent_pos

            # Normalize the direction (avoid division by zero)
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)

                # Apply kick force to ball velocity in the direction from agent to ball
                kick_force = direction * self.ball_push_strength
                self.ball_vel += kick_force

    def render(self, actions=None):
        if actions is None:
            actions = self.frame_actions

        if self.render_mode != "human":
            return

        if self.viewer is None:
            self._init_viewer()

        self._log_frame_info()
        self.viewer.render(self.agent_pos, self.ball_pos)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    def _observe(self, agent):
        field_size = np.array([self.field_width, self.field_height], dtype=np.float32)
        goal_dist = np.linalg.norm(self.goal_center - self.ball_pos)
        ball_dist = np.linalg.norm(self.agent_pos[agent] - self.ball_pos)
        teammates_pos = [
            self.agent_pos[other] for other in self.agents if other != agent
        ]

        norm_self_pos = self.agent_pos[agent] / field_size
        norm_ball_pos = self.ball_pos / field_size
        norm_ball_vel = self.ball_vel / field_size
        norm_goal_dist = goal_dist / field_size
        norm_ball_dist = ball_dist / field_size
        norm_teammates = [teammates_pos / field_size for teammates_pos in teammates_pos]
        norm_dist_teammates = [
            np.linalg.norm(self.agent_pos[agent] - pos) / field_size
            for pos in teammates_pos
        ]

        return np.concatenate(
            [
                norm_self_pos,
                norm_ball_pos,
                norm_ball_vel,
                norm_goal_dist,
                norm_ball_dist,
                *norm_teammates,
                *norm_dist_teammates,
            ]
        )

    def _calculate_rewards(self):
        self.move_towards_ball_reward = {}

        for agent in self.agents:
            # 1. Reward for moving towards the ball
            prev_dist_to_ball = np.linalg.norm(
                self.prev_ball_pos - self.prev_agent_pos[agent]
            )
            curr_dist_to_ball = np.linalg.norm(
                self.prev_ball_pos - self.agent_pos[agent]
            )
            move_towards_ball_reward = (prev_dist_to_ball - curr_dist_to_ball) * 0.1
            self.move_towards_ball_reward[agent] = move_towards_ball_reward

            self.rewards[agent] = move_towards_ball_reward

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
        return (left <= self.ball_pos[0].round() <= right) and (
            top <= self.ball_pos[1].round() <= bottom
        )

    def _get_random_pos(self, size):
        x = np.random.uniform(size, self.field_width - size)
        y = np.random.uniform(size, self.field_height - size)
        return np.array([x, y], dtype=np.float32)

    def _clip_pos(self):
        for agent in self.agents:
            self.agent_pos[agent] = np.clip(
                self.agent_pos[agent],
                self.agent_size,
                [
                    self.field_width - self.agent_size,
                    self.field_height - self.agent_size,
                ],
            )

        self.ball_pos = np.clip(
            self.ball_pos,
            self.ball_size,
            [self.field_width - self.ball_size, self.field_height - self.ball_size],
        )

    def _init_viewer(self):
        if self.render_mode != "human":
            return

        if self.viewer is not None:
            return

        self.viewer = Viewer(
            {
                "field_width": self.field_width,
                "field_height": self.field_height,
                "agent_size": self.agent_size,
                "ball_size": self.ball_size,
                "goal_zone": self.goal_zone,
                "goal_size": self.goal_size,
            }
        )

    def _log_frame_info(self):
        print(f"\n-- Frame {self.frame_count} --")
        for agent in self.agents:
            action = self.frame_actions[agent]

            print(f"{agent}:")
            print(f"\tprevious position: {self.prev_agent_pos[agent].round(2)}")
            print(f"\tcurrent position : {self.agent_pos[agent].round(2)}")
            print(f"\taction           : {self.action_list[action]['name']}")
            print()
            print("\trewards          :")
            print(f"\t\tmove towards ball: {self.move_towards_ball_reward[agent]}")
            print(f"\t\ttotal            : {self.rewards[agent]}")
            print()
            print(f"\tobservations     : {self.observations[agent].round(2)}")
            print(f"\tlength of obs    : {len(self.observations[agent])}")
            print(f"\tball inside goal : {self._check_goal_scored()}")

        print(f"ball position : {self.ball_pos.round(2)}")
        print(f"ball velocity : {self.ball_vel.round(2)}")
