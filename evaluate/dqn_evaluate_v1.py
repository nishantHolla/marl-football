from models.dqn_v1 import DQNAgent_V1
from envs.abstract_football_env_v1 import AbstractFootballEnv_V1
import numpy as np
import torch
from pathlib import Path


class MADQNEvaluator_V1:
    """
    Multi-Agent DQN Evaluator
    """

    def __init__(
        self,
        env,
        lr=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        memory_size=10000,
        batch_size=32,
    ):
        """
        Initialize the evaluator with the given parameters

        Params:
            (env)   env          : Environment to evaluate
            (float) lr           : Learning rate
            (float) gamma        : Discount factor
            (float) epsilon      : Randomization for learning
            (float) epsilon_min  : Minimum possible epsilon value
            (float) epsilon_decay: Decay factor for epsilon
            (int)   memory_size  : Size of the memory
            (int)   batch_size   : Number of sessions in one batch
        """
        self.env = env

        ## Get environment info
        self.agents = env.possible_agents
        self.num_agents = len(self.agents)

        ## Create agents
        self.dqn_agents = {}
        for agent in self.agents:
            state_size = env.observation_space(agent).shape[0]
            action_size = env.action_space(agent).n
            self.dqn_agents[agent] = DQNAgent_V1(
                state_size,
                action_size,
                lr,
                gamma,
                epsilon,
                epsilon_min,
                epsilon_decay,
                memory_size,
                batch_size,
            )

    def load_models(self, prefix):
        """
        Load trained models

        Params:
            (str) prefix: prefix string for the saved models
        """
        ## For each agent load the model
        for agent in self.agents:
            self.dqn_agents[agent].q_network.load_state_dict(
                torch.load(f"{prefix}_{agent}.pth")
            )

        print(f"Models loaded from {prefix}_*.pth")

    def evaluate(self, num_episodes=10, episode_length=1000, render=True, debug=False):
        """
        Evaluate trained agents

        Params:
            (int)  num_episodes   : Number of episodes to evaluate for
            (int)  episode_lengths: Length of each episode
            (bool) render         : Whether to render or not

        Return:
            average rewards earned per episode and total goals scored
        """
        print(f"Evaluating for {num_episodes} episodes...")

        total_rewards = []
        goals_scored = 0

        for episode in range(num_episodes):
            ## Reset the env
            observations, infos = self.env.reset()
            episode_reward = {agent: 0 for agent in self.agents}

            ## Run the env for given episode lengths
            for frame_number in range(episode_length):
                ## Get next set of actions using the agent's model
                actions = {}
                for agent in self.agents:
                    if agent in observations:
                        actions[agent] = self.dqn_agents[agent].act(
                            observations[agent], training=False
                        )

                ## Perform the action
                next_observations, rewards, terminations, truncations, infos = (
                    self.env.step(actions)
                )

                ## Get the rewards of each agent
                for agent in self.agents:
                    if agent in rewards:
                        episode_reward[agent] += rewards[agent]

                ## Render the env if required
                if render:
                    self.env.render(actions)

                to_stop = False
                if debug:
                    while True:
                        n = input("q: quit, n: next, c: continue > ")
                        if n == "c":
                            to_stop = True
                            break
                        elif n == "q":
                            return
                        elif n == "n":
                            break

                if to_stop:
                    break

                ## Copy the observations
                observations = next_observations.copy()

                ## Check if goals were scored
                if any(terminations.values()) or any(truncations.values()):
                    if any(terminations.values()):
                        goals_scored += 1
                    break

            ## Calculate total rewards in this episode
            total_reward = sum(episode_reward.values())
            total_rewards.append(total_reward)
            print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")

        ## Calculate average reward
        avg_reward = np.mean(total_rewards)
        print("\nEvaluation Results:")
        print(f"Average Reard: {avg_reward:.2f}")
        print(f"Goals Scored: {goals_scored}/{num_episodes}")
        print(f"Goal Rate: {goals_scored / num_episodes * 100:.1f}%")

        return avg_reward, goals_scored


def evaluate(model_prefix, num_episodes, debug=False):
    """
    Function to call for evaluation

    Params:
        (str) model_prefix: Path to the models
        (int) num_episodes: Number of episodes to evaluate for
    """
    ## Initialize the env
    Path(f"saves/{model_prefix}").mkdir(parents=True, exist_ok=True)
    env = AbstractFootballEnv_V1(n_agents=2, render_mode="human")

    ## Create the evaluator
    evaluator = MADQNEvaluator_V1(
        env=env,
        lr=0.0005,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.9995,
        memory_size=50000,
        batch_size=64,
    )

    ## Load the models and evaluate
    evaluator.load_models(f"saves/{model_prefix}/{model_prefix}")
    evaluator.evaluate(num_episodes=num_episodes, debug=debug)

    ## Close the env
    env.close()
