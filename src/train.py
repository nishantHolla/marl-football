import threading
import time
from pathlib import Path
import numpy as np
import pandas as pd
from football_env import FootballEnv
from dqn import DQN_MultiAgent
import matplotlib.pyplot as plt

render_toggle = threading.Event()
running_flag = threading.Event()
paused_flag = threading.Event()
rewards = []
goals = []
epsilon = []


def train_work(episodes, hyperparameters):
    env = FootballEnv(n_agents=2, render_mode="human")
    obs_size = env.observation_spaces[env.agents[0]].shape[0]
    action_size = env.action_spaces[env.agents[0]].n
    min_steps = 500
    target_update_freq = 10
    num_episodes = episodes

    agent = DQN_MultiAgent(
        agent_names=env.agents,
        state_size=obs_size,
        action_size=action_size,
        hyperparameters=hyperparameters,
    )

    episode_rewards = []
    goals = []
    goals_scored = 0
    epsilon = []

    for episode in range(num_episodes):
        if not running_flag.is_set():
            break

        env.reset()
        total_episode_reward = 0
        has_scored_goal = False

        ## Get initial observations
        observations = {
            agent_name: env._observe(agent_name) for agent_name in env.agents
        }

        for step in range(min(min_steps + 2 * episode, 2000)):
            if not running_flag.is_set():
                break

            while paused_flag.is_set():
                time.sleep(1)

            actions = {}

            ## Select actions for all agents
            for agent_name in env.agents:
                obs = observations[agent_name]
                action = agent.act(obs, training=True)
                actions[agent_name] = action

            ## Step the environment
            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            if render_toggle.is_set():
                env.render(actions=actions, episode_number=episode)

            ## Store experiences for all agents
            for agent_name in env.agents:
                state = observations[agent_name]
                action = actions[agent_name]
                reward = rewards[agent_name]
                next_state = next_obs[agent_name]
                done = terminations[agent_name] or truncations[agent_name]

                agent.remember(state, action, reward, next_state, done)

                total_episode_reward += reward

            ## Update observations
            observations = next_obs

            ## Train the agent
            agent.replay()

            if all(terminations.values()) or all(truncations.values()):
                has_scored_goal = True
                break

        ## Update target network every few episodes
        if episode % target_update_freq == 0:
            agent.update_target()

        ## Decay epsilon
        agent.decay_epsilon()

        ## Logging
        episode_rewards.append(total_episode_reward)
        epsilon.append(agent.epsilon)

        if has_scored_goal:
            goals_scored += 1
            goals.append(1)
        else:
            goals.append(0)

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(
                f"({agent.lr}, {agent.epsilon_decay}, {agent.gamma}) "
                f"Episode {episode + 1}/{num_episodes}, "
                f"Avg Reward (last 10): {avg_reward:.2f}, "
                f"Epsilon: {agent.epsilon:.3f}, "
                f"Goals scored: {goals_scored}"
            )

    return episode_rewards, goals, epsilon


def train(episodes, hyperparameters):
    render_toggle.clear()
    running_flag.set()
    paused_flag.clear()

    def train_work_wrapper(episodes, hyperparameters):
        global rewards, goals, epsilon
        rewards, goals, epsilon = train_work(episodes, hyperparameters)

    train_thread = threading.Thread(
        target=train_work_wrapper,
        args=(episodes, hyperparameters),
        daemon=True,
    )
    train_thread.start()

    while running_flag.is_set():
        try:
            n = input()
        except KeyboardInterrupt:
            running_flag.clear()
            break

        if n == "i":
            if render_toggle.is_set():
                render_toggle.clear()
            else:
                render_toggle.set()
        elif n == "p":
            if paused_flag.is_set():
                paused_flag.clear()
            else:
                paused_flag.set()

    train_thread.join()
    running_avg = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
    plt.plot(running_avg, label="Running Average")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Episode Rewards Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.clf()
    window = 10
    rolling_avg = pd.Series(goals).rolling(window).mean()
    plt.plot(rolling_avg, label="Goals scored")
    plt.xlabel("Episode")
    plt.ylabel("Goals scored")
    plt.title("Goals scored Over Episodes")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.clf()
    plt.plot(epsilon, label="Epsilon Decay")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Epsilon value Over Episodes")
    plt.legend()
    plt.grid(True)
    plt.show()

    results_dir = Path("../results")
    results_dir.mkdir(parents=True, exist_ok=True)

    x = list(range(len(running_avg)))
    df = pd.DataFrame({"x": x, "y": running_avg})
    df.to_csv(results_dir / "reward_running_avg.csv", index=False)

    df = pd.DataFrame({"x": x, "y": goals})
    df.to_csv(results_dir / "goals_dist.csv", index=False)

    df = pd.DataFrame({"x": x, "y": epsilon})
    df.to_csv(results_dir / "epsilon.csv", index=False)
