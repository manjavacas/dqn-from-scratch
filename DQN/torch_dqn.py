import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.nn import functional as F
from collections import deque


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class DQN_agent():
    ''' Deep Q-learning agent implementation. '''

    def __init__(
        self,
        observation_space,
        action_space,
        alpha=1e-4,
        gamma=.99,
        epsilon=1.,
        min_epsilon=.05,
        epsilon_decay=.995,
        tau=1.,
        grad_steps=3,
        buffer_size=1_000_000,
        batch_size=32,
        train_steps=4,
        samples_to_learn=1_000,
        target_update_steps=1_000
    ):
        ''' DQN Agent class constructor. It includes the set of parameters that characterise the agent's training and behaviour. '''

        self.observation_space = observation_space
        self.action_space = action_space

        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.tau = tau  # Polyak update coefficient

        self.epsilon = epsilon  # exploration probability
        self.min_epsilon = min_epsilon  # minimum exploration probability
        self.epsilon_decay = epsilon_decay  # exploration decay

        self.grad_steps = grad_steps  # optimization steps

        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.train_steps = train_steps
        self.samples_to_learn = samples_to_learn

        self.target_update_steps = target_update_steps
        self.steps = 0

        self.replay_buffer = deque(maxlen=self.buffer_size)

        # Q-network
        self.q_network = nn.Sequential(
            nn.Linear(self.observation_space, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_space)
        ).to(device)

        # Target-network
        self.target_network = nn.Sequential(
            nn.Linear(self.observation_space, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_space)
        ).to(device)

        # Q-network Xavier initialization
        for layer in self.q_network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.alpha)

    def pred_action(self, state):
        ''' Epsilon-greedy action selection. The following action may be random or based on the maximum estimated action value. '''
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            input_tensor = torch.from_numpy(
                state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = self.q_network(input_tensor)
            action = np.argmax(q_values.cpu().data.numpy())

        self.steps += 1

        # Train Q-network periodically
        if len(self.replay_buffer) >= self.samples_to_learn and len(self.replay_buffer) >= self.batch_size and self.steps % self.train_steps == 0:
            self.learn()

        # Update Target-network periodically
        if self.steps % self.target_update_steps == 0:
            self.update_target_net()

        # Reduce exploration
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        return action

    def save_experience(self, state, action, reward, state_next, done):
        ''' Adds a tuple (s, a, r, s', done) to the replay buffer. '''
        self.replay_buffer.append([state, action, reward, state_next, done])

    def learn(self):
        ''' Updates the Q-network based on sampled past experience tuples. '''
        # Apply N optimization steps
        for _ in range(self.grad_steps):

            # Sample data from experience
            batch_data = random.sample(self.replay_buffer, self.batch_size)

            # Get experience components
            states, actions, rewards, states_next, dones = zip(*batch_data)

            states = torch.tensor(np.asarray(states)).to(device)
            states_next = torch.tensor(np.asarray(states_next)).to(device)
            rewards = torch.tensor(np.asarray(rewards)).to(device)
            dones = torch.tensor(np.asarray(dones)).long().to(device)

            # Get next Q-values using target network
            with torch.no_grad():
                q_values_next = self.target_network(states_next)

            # Get the maximum Q-values
            max_q_values_next = q_values_next.max(dim=1)[0]

            # Compute Q_targets = R(s,a,s') + gamma * Q_max(s', a') for non-terminal states, and R(s,a,s') for terminal states
            q_targets = rewards + (1 - dones) * self.gamma * max_q_values_next

            # Compute current estimated Q_values
            q_values = self.q_network(states)
            q_values_pred = q_values[range(self.batch_size), actions]

            # Compute loss (Huber loss, which is less sensitive to outliers)
            loss = F.smooth_l1_loss(q_values_pred, q_targets)

            # Policy optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def update_target_net(self):
        ''' Applies a soft update from the Q-network to the Target-network by using the Polyak update coefficient (tau). '''
        for target_net_param, q_net_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_net_param.data.copy_(
                self.tau * q_net_param.data + (1. - self.tau) * target_net_param.data)
