import sys
import numpy as np
from envs.abstract_football_env_v1 import AbstractFootballEnv_V1
from train.dqn_train_v1 import train
from evaluate.dqn_evaluate_v1 import evaluate

if len(sys.argv) != 2:
    print("Usage: python main.py [run|train|eval]")
    sys.exit(1)

if sys.argv[1] == "run":
    env = AbstractFootballEnv_V1(n_agents=5, field_width=800, render_mode="human")
    observations, _ = env.reset()

    while True:
        actions = {
            agent: np.random.randint(len(env.action_list)) for agent in env.agents
        }
        observations, rewards, terminations, truncations, infos = env.step(actions)
        env.render(actions)

        if all(terminations.values()):
            break

elif sys.argv[1] == "train":
    train("test", 1000)

elif sys.argv[1] == "eval":
    evaluate("saves/test_1000/test_1000", 10)
