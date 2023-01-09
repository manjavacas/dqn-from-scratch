import random
import numpy as np

from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from torch.nn import functional as F


class DQN_agent():
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

        self.q_network = nn.Sequential(
            nn.Linear(self.observation_space, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_space)
        )

        self.target_network = nn.Sequential(
            nn.Linear(self.observation_space, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_space)
        )

        for layer in self.q_network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.alpha)

    def pred_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            input_tensor = torch.tensor(state)
            with torch.no_grad():
                q_values = self.q_network(input_tensor).numpy()[0]
            action = np.argmax(q_values)

        self.steps += 1

        # Train and update target net periodically
        if len(self.replay_buffer) > self.samples_to_learn and len(self.replay_buffer) >= self.batch_size and self.steps % self.train_steps == 0:
            self.train_q_net()

        if self.steps % self.target_update_steps == 0:
            self.update_target_net()

        # Reduce exploration
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        return action

    def save_experience(self, state, action, reward, state_next, done):
        self.replay_buffer.append([state, action, reward, state_next, done])

    def train_q_net(self):
        # Apply N optimization steps
        for _ in range(self.grad_steps):

            # Sample data from experience
            batch_data = random.sample(self.replay_buffer, self.batch_size)

            # Get experience components
            states, actions, rewards, states_next, dones = zip(*batch_data)

            states = torch.tensor(np.asarray(states))
            states_next = torch.tensor(np.asarray(states_next))
            dones = np.asarray(dones).astype(int)

            with torch.no_grad():
                # Compute next Q-values using target network
                q_values_next = self.target_network(states_next).numpy()
                # Get the maximum Q-value
                max_q_values_next = np.amax(q_values_next, axis=1)
                # Compute Q_targets = R(s,a,s') + gamma * Q_max(s', a')
                q_targets = rewards + (1 - dones) * \
                    self.gamma * max_q_values_next
                q_targets = torch.tensor(q_targets)

            # Compute current estimated Q_values
            q_values = self.q_network(states)
            q_values_pred = q_values[range(self.batch_size), actions]

            # Compute loss
            loss = F.smooth_l1_loss(q_values_pred, q_targets)

            # Policy optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def update_target_net(self):
        for target_net_param, q_net_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_net_param.data.copy_(
                self.tau * q_net_param.data + (1. - self.tau) * target_net_param.data)
