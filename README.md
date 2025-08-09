# MARL Football

Testing Multi-Agent reinforcement learning using Deep Q-Network in a football environment.<br />
The aim of this project was to:
- Build a simulation of a football environment using PettingZoo parallel
API with a flexible number of agents interacting with a football against a stationary obstacle.
- Train a shared Deep Q-Network that allows the agents to collaborate and explore viable
strategies that allow them to score goals against a stationary obstacle.

## Training highlights

https://github.com/user-attachments/assets/5dd6acce-2cb3-4775-97b8-6e58a886f234

- Episode 0: Agents move around the football ground in random motion due to a high value of epsilon,
resulting in high exploration and low exploitation.
- Episode 11: Agents kick the ball for the first time, which gives them a reward, encouraging them
to kick the ball again and forming relevant connections between observations.
- Episode 23: Agents score a goal for the first time, which gives them a high reward, encouraging them
to score again and forming relevant connections between observations.
- Episode 95: Agents score a goal by collaborating, which gives them a higher reward compared to a solo
goal, encouraging them to collaborate to score goals.
- Episode 548: Agent motion is less random due to the decaying value of epsilon, which results in high
exploitation and low exploration.

## Hyperparameters

Many hyperparameters like neural network architecture, dropout value, batch size, learning rate,
epsilon decay, and gamma can be changed in order to improve the performance of the model.<br /><br />
Three hyperparameters: learning rate, gamma, and epsilon decay were changed to observe how they
impacted the performance of the model.<br /><br />
A baseline of `learning rate = 0.001`, `epsilon-decay = 0.995`, and `gamma = 0.95` was taken, and the
impact of changing them was observed. The results of these can be found in the results directory
of the project.

## Usage

- Clone the repository
```bash
git clone https://github.com/nishantHolla/marl-football.git
cd marl-football
```

- Install [uv](https://github.com/astral-sh/uv?tab=readme-ov-file)

- Set up the environment and install dependencies
```bash
uv venv
uv sync
```

- Enter the environment
```bash
# For linux and macOS
source .venv/bin/activate

# For windows
.venv\Scripts\Activate.ps1
```

- Run the main.py file
```bash
cd src

# Run the environment step by step
python main.py debug

# Train the model
python main.py train
```

- Update the main.py file for changing the number of episodes to train and the hyperparameters.

- Once training is completed, the results will be stored in the results directory.
