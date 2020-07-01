import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.distributions import Categorical
import numpy as np


class Actor(nn.Module):
    def __init__(self, lr, input_size=4, hidden_size_1=256, hidden_size_2=256, output_size=2):
        super(Actor, self).__init__()
        "Actor decides and updates the policy based on the feedback from the Critic"
        self.learning_rate = lr
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, output_size)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.loss_func = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, data):
        action_probs = func.relu(self.fc1(data))
        action_probs = func.relu(self.fc2(action_probs))
        action_probs = self.fc3(action_probs)

        return action_probs


class Critic(nn.Module):
    def __init__(self, lr, input_size=4, hidden_size_1=256, hidden_size_2=256, output_size=1):
        super(Critic, self).__init__()
        """Critic takes in state (s), and outputs a value for that state"""
        self.learning_rate = lr
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, output_size)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.loss_func = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation):
        value = func.relu(self.fc1(observation))
        value = func.relu(self.fc2(value))
        value = self.fc3(value)

        return value


class Agent:
    def __init__(self, lr=.001, num_states=4, num_actions=2):
        self.gamma = .99
        self.value_network = Critic(lr, num_states, output_size=num_actions)
        self.policy_network = Actor(lr, num_states)

        self.log_probs = []
        self.values = []

    def step(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32)

        probs = func.softmax(self.policy_network.forward(state_tensor))
        act_probs = Categorical(probs)
        chosen_action = act_probs.sample()
        log_probs = act_probs.log_prob(chosen_action)
        self.log_probs.append(log_probs)

        value = self.value_network.forward(state_tensor)
        self.values.append(value)

        return chosen_action.item()

    def sample_trajectory(self, rewards, dones):
        self.value_network.optimizer.zero_grad()
        self.policy_network.optimizer.zero_grad()

        q_vals = np.zeros((len(rewards), 1))
        q_val = 0.
        for i, reward in enumerate(reversed(rewards)):
            q_val = reward + self.gamma * q_val * (1.0 - int(dones[i]))
            q_vals[-1 - i] = q_val

        advantage = torch.from_numpy(q_vals) - torch.stack(self.values)

        critic_loss = advantage.pow(2).mean()
        critic_loss.backward()
        self.value_network.optimizer.step()

        actor_loss = (-torch.stack(self.log_probs) * advantage.detach()).mean()
        actor_loss.backward()
        self.policy_network.optimizer.step()

        self.values = []
        self.log_probs = []
