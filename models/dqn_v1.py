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

    def __init__(self, state_size, action_size, hidden_size=128):
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
        self.fc4 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        """
        Forwarding function
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


class DQNAgent_V1:
    """
    DQN Agent for multi-agent learning
    """

    def __init__(
        self,
        state_size,
        action_size,
        lr=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        memory_size=10000,
        batch_size=32,
    ):
        """
        Initialize the DQN for multiple agents

        Params:
            (int)   state_size   : Size of the input state
            (int)   action_size  : Size of the output state
            (float) lr           : Learning rate
            (float) gamma        : Discount factor
            (float) epsilon      : Randomization for learning
            (float) epsilon_min  : Minimum possible epsilon value
            (float) epsilon_decay: Decay factor for epsilon
            (int)   memory_size  : Size of the memory
            (int)   batch_size   : Number of sessions in one batch
        """
        ## Initialize the parameters
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        ## Define the Neural networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQN_V1(state_size, action_size).to(self.device)
        self.target_network = DQN_V1(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        ## Experience replay
        self.memory = deque(maxlen=memory_size)

        ## Update target network
        self.update_target_network()

    def update_target_network(self):
        """
        Copy weights from main network to target network
        """
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay buffer
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training=True):
        """
        Choose action using epsilon-greedy policy
        """
        if training and random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())

    def replay(self):
        """
        Train the agent on a batch of experiences
        """
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def replay_without_epsilon_decay(self):
        """
        Train the agent on a batch of experiences without decaying epsilon
        """
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.array([e[0] for e in batch])).to(self.device)
        actions = torch.LongTensor(np.array([e[1] for e in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([e[2] for e in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([e[3] for e in batch])).to(self.device)
        dones = torch.BoolTensor(np.array([e[4] for e in batch])).to(self.device)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
