from models.dqn_v1 import DQNAgent_V1
from envs.abstract_football_env_v1 import AbstractFootballEnv_V1
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path


class MADQNTrainer_V1:
    """
    Multi-Agent DQN Trainer

    """

    def __init__(
        self,
        env,
        model_prefix,
        lr,
        gamma,
        epsilon,
        epsilon_min,
        epsilon_decay,
        memory_size,
        batch_size,
        target_update,
    ):
        """
        Initialize the trainer with the given parameters

        Params:
            (env)   env          : Environment to evaluate
            (str)   model_prefix : Path to save checkpoints
            (float) lr           : Learning rate
            (float) gamma        : Discount factor
            (float) epsilon      : Randomization for learning
            (float) epsilon_min  : Minimum possible epsilon value
            (float) epsilon_decay: Decay factor for epsilon
            (int)   memory_size  : Size of the memory
            (int)   batch_size   : Number of sessions in one batch
            (int)   target_update: Update after every n episodes
        """
        ## Initialize the env
        self.env = env
        self.model_prefix = model_prefix
        self.target_update = target_update

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

        ## Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.goal_count = 0

    def train(self, num_episodes, max_steps, render_every, save_every):
        """
        Train the multi-agent system

        Params:
            (int) num_episodes: Number of episodes to train for
            (int) max_steps   : Number of steps to perform in each episode
            (int) render_every: Render every n episodes
            (int) save_every  : Save as checkpoint after every n episodes
        """
        print(f"Starting training for {num_episodes} episodes...")
        print(f"Device: {self.dqn_agents[self.agents[0]].device}")
        print(f"Initial epsilon: {self.dqn_agents[self.agents[0]].epsilon:.3f}")

        for episode in range(num_episodes):
            observations, infos = self.env.reset()
            episode_reward = {agent: 0 for agent in self.agents}
            episode_length = 0

            for step in range(max_steps):
                ## Get actions from all agents
                actions = {}
                for agent in self.agents:
                    if agent in observations:
                        actions[agent] = self.dqn_agents[agent].act(observations[agent])

                ## Step environment
                next_observations, rewards, terminations, truncations, infos = (
                    self.env.step(actions)
                )

                ## Store experiences and update episode rewards
                for agent in self.agents:
                    if agent in observations and agent in next_observations:
                        self.dqn_agents[agent].remember(
                            observations[agent],
                            actions[agent],
                            rewards[agent],
                            next_observations[agent],
                            terminations[agent] or truncations[agent],
                        )
                        episode_reward[agent] += rewards[agent]

                ## Train agents
                for agent in self.agents:
                    if (
                        len(self.dqn_agents[agent].memory)
                        >= self.dqn_agents[agent].batch_size
                    ):
                        self.dqn_agents[agent].replay_without_epsilon_decay()

                ## Update target networks periodically
                if episode % self.target_update == 0:
                    for agent in self.agents:
                        self.dqn_agents[agent].update_target_network()

                ## Render if needed
                if episode % render_every == 0 and episode > 0:
                    self.env.render(actions)

                observations = next_observations
                episode_length += 1

                ## Check if episode is done
                if any(terminations.values()) or any(truncations.values()):
                    if any(terminations.values()):
                        self.goal_count += 1
                    break

            ## Decay epsilon once per episode for all agents
            for agent in self.agents:
                if self.dqn_agents[agent].epsilon > self.dqn_agents[agent].epsilon_min:
                    self.dqn_agents[agent].epsilon *= self.dqn_agents[
                        agent
                    ].epsilon_decay

            ## Store episode metrics
            total_reward = sum(episode_reward.values())
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(episode_length)

            ## Print progress
            if episode % 50 == 0:
                avg_reward = (
                    np.mean(self.episode_rewards[-50:]) if self.episode_rewards else 0
                )
                avg_length = (
                    np.mean(self.episode_lengths[-50:]) if self.episode_lengths else 0
                )
                epsilon = self.dqn_agents[self.agents[0]].epsilon

                # Calculate reward statistics
                recent_rewards = (
                    self.episode_rewards[-50:]
                    if len(self.episode_rewards) >= 50
                    else self.episode_rewards
                )
                min_reward = np.min(recent_rewards) if recent_rewards else 0
                max_reward = np.max(recent_rewards) if recent_rewards else 0

                print(
                    f"Episode {episode:4d} | "
                    f"Avg Reward: {avg_reward:7.2f} | "
                    f"Min/Max: {min_reward:6.1f}/{max_reward:6.1f} | "
                    f"Avg Length: {avg_length:6.2f} | "
                    f"Goals: {self.goal_count:3d} | "
                    f"Epsilon: {epsilon:.4f}"
                )

            ## Save models periodically
            if episode % save_every == 0 and episode > 0:
                self.save_models(f"checkpoint_{episode}_{self.model_prefix}")

        print(f"Training completed! Total goals scored: {self.goal_count}")

    def save_models(self, prefix):
        """
        Save trained models

        Params:
            (str) prefix: Prefix path to save the models in
        """
        for agent in self.agents:
            torch.save(
                self.dqn_agents[agent].q_network.state_dict(), f"{prefix}_{agent}.pth"
            )
        print(f"Models saved as {prefix}_*.pth")

    def plot_training_progress(self):
        """
        Plot training metrics
        """
        if not self.episode_rewards:
            print("No training data to plot")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ## Plot rewards
        ax1.plot(self.episode_rewards, alpha=0.3, color="blue")

        ## Moving average
        window = 50
        if len(self.episode_rewards) >= window:
            moving_avg = [
                np.mean(self.episode_rewards[i : i + window])
                for i in range(len(self.episode_rewards) - window + 1)
            ]
            ax1.plot(
                range(window - 1, len(self.episode_rewards)), moving_avg, color="red"
            )

        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Total Reward")
        ax1.set_title("Training Rewards")
        ax1.grid(True)

        ## Plot episode lengths
        ax2.plot(self.episode_lengths, alpha=0.3, color="green")
        if len(self.episode_lengths) >= window:
            moving_avg = [
                np.mean(self.episode_lengths[i : i + window])
                for i in range(len(self.episode_lengths) - window + 1)
            ]
            ax2.plot(
                range(window - 1, len(self.episode_lengths)), moving_avg, color="red"
            )

        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Episode Length")
        ax2.set_title("Episode Lengths")
        ax2.grid(True)

        plt.tight_layout()
        plt.show()


def train(model_prefix, num_episodes):
    """
    Function to call for training

    Params:
        (str) model_prefix: Path to the models
        (int) num_episodes: Number of episodes to evaluate for
    """
    ## Initialize the env
    Path(f"saves/{model_prefix}_{num_episodes}").mkdir(parents=True, exist_ok=True)
    env = AbstractFootballEnv_V1(n_agents=2, render_mode="human")

    ## Create the trainer
    trainer = MADQNTrainer_V1(
        env=env,
        model_prefix=model_prefix,
        lr=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        memory_size=50000,
        batch_size=32,
        target_update=100,
    )

    ## Train
    trainer.train(
        num_episodes=num_episodes, max_steps=1000, render_every=1, save_every=500
    )

    ## Plot training and save models
    trainer.plot_training_progress()
    trainer.save_models(
        f"saves/{model_prefix}_{num_episodes}/{model_prefix}_{num_episodes}"
    )

    env.close()
