import torch

import numpy as np

import torch.nn as nn
import torch.optim as optim

from collections import deque

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class A2C_agent:
    def __init__(self, observation_space, action_space, batch_size=100, gamma=.99, alpha_actor=1e-3, alpha_critic=1e-3):

        self.observation_space = observation_space
        self.action_space = action_space

        self.alpha_actor = alpha_actor
        self.alpha_critic = alpha_critic

        self.gamma = gamma

        self.batch_size = batch_size
        self.experience_batch = deque(maxlen=self.batch_size)

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(self.observation_space, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_space),
            nn.Softmax(dim=1)
        ).to(device)

        # Actor weight initialization
        for layer in self.actor:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(self.observation_space, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(device)

        # Critic weight initialization
        for layer in self.critic:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

        self.optim_actor = optim.Adam(
            self.actor.parameters(), lr=self.alpha_actor)
        self.optim_critic = optim.Adam(
            self.critic.parameters(), lr=self.alpha_critic)

    def pred_action(self, state):
        state_tensor = torch.Tensor(state).unsqueeze(0).to(device)
        action_probs = self.actor(state_tensor).detach()
        action = torch.multinomial(action_probs, 1).item()
        return action

    def save_experience(self, state, action, reward, state_next, done):
        self.experience_batch.append(
            (state, action, reward, state_next, done))

        # Learn when the experience batch is full or when the episode ends
        if len(self.experience_batch) % self.batch_size == 0 or done:
            self.learn()
            # Reset experience
            self.experience_batch.clear()

    def learn(self):
        # Initialize the loss
        actor_loss = 0.
        critic_loss = 0.

        # Loop through the experience batch
        for state, _, reward, state_next, done in self.experience_batch:
            state_tensor = torch.tensor(state).unsqueeze(0).to(device)
            state_next_tensor = torch.tensor(state_next).to(device)

            advantage = reward + (1 - done) * self.gamma * \
                self.critic(state_next_tensor) - \
                self.critic(state_tensor).detach()

            # Compute action prob distribution and sample action
            action_probs = self.actor(state_tensor).detach()
            action_distrib = torch.distributions.Categorical(
                probs=action_probs)
            action = action_distrib.sample().unsqueeze(0)
            action_tensor = torch.tensor(action, dtype=torch.int64)

            # Compute and accumulate loss
            critic_loss += advantage.pow(2)
            actor_loss += -action_distrib.log_prob(action_tensor) * advantage

        # Average the loss
        batch_len = len(self.experience_batch)
        actor_loss /= batch_len
        critic_loss /= batch_len

        # Update Actor and Critic networks
        self.optim_actor.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.optim_actor.step()

        self.optim_critic.zero_grad()
        critic_loss.backward()
        self.optim_critic.step()


