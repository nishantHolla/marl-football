import torch
import torch.nn as nn
import random
from collections import deque


class DQN_V1(nn.Module):
    """
    Deep Q-Network for individual agents
    """

    def __init__(self, state_size, action_size, hidden_size=256):
        """
        Initialize the DQN

        (int) state_size : Size of the input state
        (int) action_size: Size of the otuput state
        (int) hidden_size: Size of the hidden layers
        """
        super(DQN_V1, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc5 = nn.Linear(hidden_size // 2, action_size)
        self.dropout = nn.Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        """
        Initialize network weights using Xavier initialization
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forwarding function
        """
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.relu(self.fc4(x))

        return self.fc5(x)


class DQN_MultiAgent:
    def __init__(self, agent_names, state_size, action_size, hyperparameters):
        """
        Initialize a multi-agent DQN trainer with a shared Q-network.

        Params:
            (list) agent_names      : list of agent identifiers (strings)
            (int) state_size        : dimensionality of the observation/state space
            (int) action_size       : number of possible discrete actions
            (dict) hyperparameters  : dictionary containing training hyperparameters
                                      {
                                          "lr": learning rate (default=0.001),
                                          "epsilon_decay": decay rate for epsilon (default=0.995),
                                          "gamma": discount factor (default=0.95),
                                          "epsilon_min": minimum exploration epsilon (default=0.1),
                                          "memory_size": size of the replay buffer (default=100000)
                                      }
        """
        ## Parameters
        self.agent_names = agent_names
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon_start = 1.0
        self.epsilon = self.epsilon_start
        self.batch_size = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ## Hyperparameters
        self.lr = hyperparameters.get("lr", 0.001)
        self.epsilon_decay = hyperparameters.get("epsilon_decay", 0.995)
        self.gamma = hyperparameters.get("gamma", 0.95)
        self.epsilon_min = hyperparameters.get("epsilon_min", 0.1)
        self.memory_size = hyperparameters.get("memory_size", 100000)

        ## Shared model
        self.memory = deque(maxlen=self.memory_size)
        self.q_network = DQN_V1(state_size, action_size).to(self.device)
        self.target_network = DQN_V1(state_size, action_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)

    def act(self, agent_obs, training=True):
        """
        Select an action for a given agent based on its observation.

        Params:
            (array-like) agent_obs : observation vector for the agent
            (bool) training        : if True, applies epsilon-greedy exploration

        Returns:
            (int) action index
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        obs = torch.FloatTensor(agent_obs).unsqueeze(0).to(self.device)
        q_values = self.q_network(obs)

        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        """
        Initialize a multi-agent DQN trainer with a shared Q-network.

        Params:
            (list) agent_names      : list of agent identifiers (strings)
            (int) state_size        : dimensionality of the observation/state space
            (int) action_size       : number of possible discrete actions
            (dict) hyperparameters  : dictionary containing training hyperparameters
                                      {
                                          "lr": learning rate (default=0.001),
                                          "epsilon_decay": decay rate for epsilon (default=0.995),
                                          "gamma": discount factor (default=0.95),
                                          "epsilon_min": minimum exploration epsilon (default=0.1),
                                          "memory_size": size of the replay buffer (default=100000)
                                      }
        """
        self.memory.append((state, action, reward, next_state, done))

    def update_target(self):
        """
        Copy weights from the Q-network to the target network.
        """
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        """
        Decay the exploration rate epsilon after each episode.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def replay(self):
        """
        Perform a single step of training using experience replay.

        Returns:
            (float) loss value from the Q-network update (or 0.0 if not enough memory)
        """
        if len(self.memory) < self.batch_size:
            return 0.0

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        ## Compute target Q-values using Bellman equation
        current_q = self.q_network(states).gather(1, actions).squeeze()
        max_next_q = self.target_network(next_states).max(1)[0]
        target_q = rewards + self.gamma * max_next_q * (~dones)

        ## Compute loss
        loss = torch.nn.MSELoss()(current_q, target_q)

        ## Optimize the Q-network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        return loss
