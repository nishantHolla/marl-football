from pettingzoo import ParallelEnv
from gymnasium.spaces import Box, Discrete
from render.viewer import Viewer
import numpy as np


class AbstractFootballEnv_V1(ParallelEnv):
    """Abstract Footbal Environment with N agents, a ball and a Goalpost"""

    metadata = {"render_modes": ["human", "none"], "name": "abstract_football_env_v1"}

    def __init__(self, render_mode="none", n_agents=2, field_width=800):
        """
        Initialize the football env

        Params:
            (str) render_mode: "none" for no rendering or "human" for rendering with pygame
            (int) n_agents   : number of agents to add to the env
            (int) field_width: Width of the filed in pixels
        """
        ## Env params
        self.render_mode = render_mode
        self.n_agents = n_agents
        self.field_width = field_width
        self.field_height = round(field_width * 2 / 3)

        ## Agent params
        self.agent_size = 20
        self.agent_speed = 3
        self.possible_agents = [f"agent_{i}" for i in range(self.n_agents)]
        self.agents = self.possible_agents[:]

        ## Ball params
        self.ball_size = 15
        self.ball_push_strength = 3.0
        self.ball_friction = 0.95

        self.kick_range = self.agent_size + self.ball_size + 3

        ## Agent and ball positions
        self.agent_pos = {}
        self.ball_pos = np.array(
            [self.field_width // 2, self.field_height // 2], dtype=np.float32
        )
        self.ball_vel = np.array([0.0, 0.0], dtype=np.float32)

        ## Goalpost params
        self.goal_size = np.array([20, self.field_height // 4])
        self.goal_center = np.array([30, self.field_height // 2])

        ## Goalpost position
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

        ## Agent action list
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

        ## Action space and Observation space of all agents
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

        ## Initialize viewer if needed
        self.viewer = None
        self._init_viewer()

        ## Reset the env
        self.reset()

    def action_space(self, agent):
        """
        Get the actions space of the given agent

        Params:
            (str) agent: Agents whose action space is needed

        Return:
            action space of the agent
        """

        return self.action_spaces[agent]

    def observation_space(self, agent):
        """
        Get the observation space of the given agent

        Params:
            (str) agent: Agents whose observation space is needed

        Return:
            observation space of the agent
        """

        return self.observation_spaces[agent]

    def reset(self, seed=None, options=None):
        """
        Reset the env to starting values

        Params:
            (int)  seed   : Seed number of randomization
            (dict) options: Overrides for reset

        Return:
            observations of all agents and infos
        """
        ## Reset agents and their positions
        self.agents = self.possible_agents[:]
        self.agent_pos = {}

        ## Place agents at random positions in the filed
        placed_pos = []
        for agent in self.agents:
            ## Generate position and check if it collides with other agents or the ball
            while True:
                pos = self._get_random_pos(self.agent_size)
                collides_with_agent = self._check_agent_agent_collision(pos, placed_pos)
                collides_with_ball = self._check_agent_ball_collision(pos)
                if not collides_with_ball and not collides_with_agent:
                    self.agent_pos[agent] = pos
                    placed_pos.append(pos)
                    break

        ## Reset ball position and velocity
        self.ball_pos = np.array(
            [self.field_width // 2, self.field_height // 2], dtype=np.float32
        )
        self.ball_vel = np.array([0.0, 0.0], dtype=np.float32)

        ## Reset observations and env metadata
        self.frame_count = 0
        self.observations = {agent: self._observe(agent) for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        return self.observations, self.infos

    def step(self, actions):
        """
        Perform actions for the agents

        Params:
            (dict) actions: Key value pair of agent and the action index they need to perform

        Return:
            New observations of agents, Rewards earned by agents in this step, Termination of agents,
            Truncations of agents and infos
        """
        ## Define return values
        self.frame_actions = actions
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        ## Copy current agent and ball pos as previous positions
        self.prev_agent_pos = {
            agent: self.agent_pos[agent].copy() for agent in self.agents
        }
        self.prev_ball_pos = self.ball_pos[:]

        ## Perform the actions
        for agent in self.agents:
            if agent not in actions:
                continue

            ## Get the motion to perform from the action index
            action_idx = actions[agent]
            motion = self.action_list[action_idx]["motion"]

            ## Calculate the intended new position
            intended_pos = self.agent_pos[agent] + motion * self.agent_speed

            ## Get positions of other agents (excluding current agent)
            other_agent_positions = [
                self.agent_pos[other] for other in self.agents if other != agent
            ]

            ## Check if intended position would cause collisions
            agent_collision = self._check_agent_agent_collision(
                intended_pos, other_agent_positions
            )
            ball_collision = self._check_agent_ball_collision(intended_pos)

            if not agent_collision and not ball_collision:
                ## No collision - move to intended position
                self.agent_pos[agent] = intended_pos
            else:
                ## Collision detected - try to move as much as possible
                self.agent_pos[agent] = self._resolve_collision_movement(
                    agent, motion * self.agent_speed, other_agent_positions
                )

            ## Handle kick action
            if self.action_list[action_idx]["name"] == "KICK":
                self._handle_kick_action(agent)

        ## Restrict ball movement if it is inside the goal area
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
            self._move_ball_with_collision_detection()

        self.ball_vel *= self.ball_friction

        ## Clip the positions of agents and the ball to prevent them from leaving the field
        self._clip_pos()

        ## Check if goal is scored and calculate agent rewards
        self.goal_scored = self._check_goal_scored()
        self._calculate_rewards()

        ## Get the new observations of the agents
        self.observations = {agent: self._observe(agent) for agent in self.agents}
        self.frame_count += 1

        return (
            self.observations,
            self.rewards,
            self.terminations,
            self.truncations,
            self.infos,
        )

    def _move_ball_with_collision_detection(self):
        """
        Move the ball while checking for collisions with agents.
        If a collision is detected, stop the ball at the collision point.
        """
        if np.linalg.norm(self.ball_vel) == 0:
            return  # Ball is not moving

        # Calculate intended new ball position
        intended_ball_pos = self.ball_pos + self.ball_vel

        # Get all current agent positions
        all_agent_positions = list(self.agent_pos.values())

        # Check if the intended position would cause collision with any agent
        collision_detected = False
        for agent_pos in all_agent_positions:
            distance = np.linalg.norm(intended_ball_pos - agent_pos)
            collision_distance = self.ball_size + self.agent_size

            if distance < collision_distance:
                collision_detected = True
                break

        if not collision_detected:
            # No collision - move ball to intended position
            self.ball_pos = intended_ball_pos
        else:
            # Collision detected - find the furthest safe position along the trajectory
            safe_ball_pos = self._find_safe_ball_position(all_agent_positions)
            self.ball_pos = safe_ball_pos
            # Stop the ball when it collides
            self.ball_vel = np.zeros(2, dtype=np.float32)

    def _find_safe_ball_position(self, agent_positions):
        """
        Find the furthest position the ball can move along its trajectory
        without colliding with any agent.
        """
        # Try moving the ball in small steps along its velocity vector
        max_steps = 20
        current_pos = self.ball_pos.copy()
        step_vel = self.ball_vel / max_steps
        safe_pos = current_pos.copy()

        for step in range(1, max_steps + 1):
            test_pos = current_pos + step_vel * step

            # Check collision with all agents at this test position
            collision_detected = False
            for agent_pos in agent_positions:
                distance = np.linalg.norm(test_pos - agent_pos)
                collision_distance = self.ball_size + self.agent_size

                if distance < collision_distance:
                    collision_detected = True
                    break

            if not collision_detected:
                # This position is safe
                safe_pos = test_pos.copy()
            else:
                # Collision detected, stop here
                break

        return safe_pos

    def _resolve_collision_movement(
        self, agent, intended_movement, other_agent_positions
    ):
        """
        Try to move the agent as much as possible in the intended direction
        without causing collisions with other agents or the ball.

        Params:
            (str)        agent                : Agent to resolve the collision for
            (float[2])   intended_movement    : Velocity vector of intended motion
            (float[][2]) other_agent_positions: List of position vector of other agents

        Return:
            The best position the agent can move without collision
        """
        ## Get the current position of the agent
        current_pos = self.agent_pos[agent]

        ## Try to move step by step (10 steps) to see how far we can go
        max_steps = 10
        best_pos = current_pos.copy()

        for step in range(1, max_steps + 1):
            ## Calculate partial movement
            partial_movement = intended_movement * (step / max_steps)
            test_pos = current_pos + partial_movement

            ## Check for collisions at this test position
            agent_collision = self._check_agent_agent_collision(
                test_pos, other_agent_positions
            )
            ball_collision = self._check_agent_ball_collision(test_pos)

            if not agent_collision and not ball_collision:
                ## This position is valid, update best position
                best_pos = test_pos.copy()
            else:
                ## Collision detected, stop here and return the last valid position
                break

        return best_pos

    def _handle_kick_action(self, agent):
        """
        Handle the KICK action by pushing the ball if the agent is close enough.
        The ball is pushed in the direction from the agent to the ball.

        Params:
            (str) agent: Agent who performed the kick
        """
        ## Get the position of the agent
        agent_pos = self.agent_pos[agent]

        ## Check if agent is close enough to kick the ball
        distance_to_ball = np.linalg.norm(agent_pos - self.ball_pos)

        if distance_to_ball <= self.kick_range:
            ## Calculate direction from agent to ball
            direction = self.ball_pos - agent_pos

            ## Normalize the direction (avoid division by zero)
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)

                ## Apply kick force to ball velocity in the direction from agent to ball
                kick_force = direction * self.ball_push_strength
                self.ball_vel += kick_force

    def render(self, actions=None):
        """
        Render the env using pygame

        Params:
            (dict) actions: Actions that the agents performed in the current frame
        """
        ## Get the actions from the env if not passed
        if actions is None:
            actions = self.frame_actions

        ## If render mode is not human then stop
        if self.render_mode != "human":
            return

        ## Initialize the viewer if not yet initialized
        if self.viewer is None:
            self._init_viewer()

        ## Log frame info and render the view using the viewer
        self._log_frame_info()
        self.viewer.render(self.agent_pos, self.ball_pos)

    def close(self):
        """
        Clean up the env before shutdown
        """
        ## Call the close function of the viewer if it is initialized
        if self.viewer is not None:
            self.viewer.close()

    def _observe(self, agent):
        """
        Get the observations of the given agent

        Params:
            (str) agent: Name of the agent to get the observations for

        Return:
            Observation array of the given agent
        """
        ## NOTE: If you modify the return array of this function don't forget to update the shape
        ##       parameter in observation_spaces of the __init__ function

        ## Calculate field size, distance between goal and ball, and distance between agent and ball
        field_size = np.array([self.field_width, self.field_height], dtype=np.float32)
        goal_dist = np.linalg.norm(self.goal_center - self.ball_pos)
        ball_dist = np.linalg.norm(self.agent_pos[agent] - self.ball_pos)
        teammates_pos = [
            self.agent_pos[other] for other in self.agents if other != agent
        ]

        ## Normalize the values using filed size
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

        ## Concatenate observations into a single array and return it
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
        """
        Calculate the rewards of all agents for this frame

        Return:
            Reward dict of agents and the rewards they earned
        """
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
        """
        Check if the given position of an agent collides with other positions

        Params:
            (float[2])   pos                  : Position of the agent to check for collision
            (float[][2]) other_agent_positions: Position array of all other agents
            (float)      min_dist             : Minimum distance possible between two agents without collision

        Return:
            True if pos collides with other_positions else False
        """
        ## Calculate min_dist if not given
        if min_dist is None:
            min_dist = 2 * self.agent_size

        ## Check for collision
        for other_pos in other_positions:
            if np.linalg.norm(pos - other_pos) < min_dist:
                return True

        return False

    def _check_agent_ball_collision(self, pos, min_dist=None):
        """
        Check if the given position of the agent collides with the ball position

        Params:
            (float[2]) pos     : Position of the agent to check for collision
            (float)    min_dist: Minimum distance possible between agent and ball without collision

        Return:
            True if the agent collides with the ball else False
        """
        ## Calculate min_dist if not given
        if min_dist is None:
            min_dist = self.agent_size + self.ball_size

        ## Check for collision
        return np.linalg.norm(pos - self.ball_pos) < min_dist

    def _check_goal_scored(self):
        """
        Check if the ball is in the goal zone

        Return:
            True if ball is in the goal zone else False
        """
        ## Define the boundaries of the goal zone
        left = self.goal_zone["top_left"][0].round()
        right = self.goal_zone["top_right"][0].round()
        top = self.goal_zone["top_left"][1].round()
        bottom = (self.goal_zone["bottom_right"][1] - self.ball_size).round()

        ## Check if the ball is inside the goal zone
        return (left <= self.ball_pos[0].round() <= right) and (
            top <= self.ball_pos[1].round() <= bottom
        )

    def _get_random_pos(self, size):
        """
        Generate a ranomd position in the field

        Params:
            (float) size: Size of the entity for which the position will be used for

        Return:
            Random position vector
        """
        x = np.random.uniform(size, self.field_width - size)
        y = np.random.uniform(size, self.field_height - size)
        return np.array([x, y], dtype=np.float32)

    def _clip_pos(self):
        """
        Clip the positions of the agents and the ball so that they do not leave the filed
        """
        ## Clip the positions of the agents
        for agent in self.agents:
            self.agent_pos[agent] = np.clip(
                self.agent_pos[agent],
                self.agent_size,
                [
                    self.field_width - self.agent_size,
                    self.field_height - self.agent_size,
                ],
            )

        ## Clip the position of the ball
        self.ball_pos = np.clip(
            self.ball_pos,
            self.ball_size,
            [self.field_width - self.ball_size, self.field_height - self.ball_size],
        )

    def _init_viewer(self):
        """
        Initialize the viewer
        """
        ## No need to initialize if render_mode is not human
        if self.render_mode != "human":
            return

        ## Return if viewer is already initialized
        if self.viewer is not None:
            return

        ## Initialize the viewer
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
        """
        Log the information of the current frame to stdout
        """
        print(f"\n-- Frame {self.frame_count} --")
        ## Print info about each agent
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

        ## Print info about ball
        print(f"ball position : {self.ball_pos.round(2)}")
        print(f"ball velocity : {self.ball_vel.round(2)}")
