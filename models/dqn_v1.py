import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
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

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using Xavier initialization"""
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
    def __init__(self, agent_names, state_size, action_size, shared=True):
        self.agent_names = agent_names
        self.state_size = state_size
        self.action_size = action_size
        self.shared = shared
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Shared model
        self.q_network = DQN_V1(state_size, action_size).to(self.device)
        self.target_network = DQN_V1(state_size, action_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.001)

        self.memory = deque(maxlen=100000)
        self.batch_size = 64
        self.gamma = 0.95

    def act(self, agent_obs, training=True):
        """
        Act on behalf of an agent given its observation
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        obs = torch.FloatTensor(agent_obs).unsqueeze(0).to(self.device)
        q_values = self.q_network(obs)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        current_q = self.q_network(states).gather(1, actions).squeeze()
        max_next_q = self.target_network(next_states).max(1)[0]
        target_q = rewards + self.gamma * max_next_q * (~dones)

        loss = torch.nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
