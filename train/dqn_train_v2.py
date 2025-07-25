import numpy as np
from envs.abstract_football_env_v1 import AbstractFootballEnv_V1
from models.dqn_v1 import DQN_MultiAgent
import threading

render_toggle = threading.Event()
running_flag = threading.Event()


def train_work():
    env = AbstractFootballEnv_V1(n_agents=2, render_mode="human")
    obs_size = env.observation_spaces[env.agents[0]].shape[0]
    action_size = env.action_spaces[env.agents[0]].n
    min_steps = 500
    target_update_freq = 10
    num_episodes = 100000

    agent = DQN_MultiAgent(
        agent_names=env.agents,
        state_size=obs_size,
        action_size=action_size,
    )

    episode_rewards = []
    goals_scored = 0

    for episode in range(num_episodes):
        if not running_flag.is_set():
            break

        env.reset()
        total_episode_reward = 0

        # Get initial observations
        observations = {
            agent_name: env._observe(agent_name) for agent_name in env.agents
        }

        for step in range(min(min_steps + 2 * episode, 2000)):
            if not running_flag.is_set():
                break
            actions = {}

            # Select actions for all agents
            for agent_name in env.agents:
                obs = observations[agent_name]
                action = agent.act(obs, training=True)
                actions[agent_name] = action

            # Step the environment
            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            if render_toggle.is_set():
                env.render(actions)

            # Store experiences for all agents
            for agent_name in env.agents:
                state = observations[agent_name]
                action = actions[agent_name]
                reward = rewards[agent_name]
                next_state = next_obs[agent_name]
                done = terminations[agent_name] or truncations[agent_name]

                agent.remember(state, action, reward, next_state, done)

                total_episode_reward += reward

            # Update observations
            observations = next_obs

            # Train the agent
            agent.replay()

            if all(terminations.values()) or all(truncations.values()):
                goals_scored += 1
                break

        # Update target network every few episodes
        if episode % target_update_freq == 0:
            agent.update_target()

        # Decay epsilon
        agent.decay_epsilon()

        # Logging
        episode_rewards.append(total_episode_reward)
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(
                f"Episode {episode + 1}/{num_episodes}, "
                f"Avg Reward (last 10): {avg_reward:.2f}, "
                f"Epsilon: {agent.epsilon:.3f}, "
                f"Goals scored: {goals_scored}"
            )

    return episode_rewards


def train():
    render_toggle.clear()
    running_flag.set()

    train_thread = threading.Thread(
        target=train_work,
        daemon=True,
    )
    train_thread.start()

    while running_flag.is_set():
        try:
            input()
        except KeyboardInterrupt:
            running_flag.clear()
            break

        if render_toggle.is_set():
            render_toggle.clear()
        else:
            render_toggle.set()

    train_thread.join()
