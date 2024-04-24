import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class RLAgent:
    def __init__(self, state_size, action_size):
        # Initialize RL agent parameters
        self.state_size = state_size
        self.action_size = action_size
        self.policy = PolicyNetwork(state_size, action_size)
        self.value_net = ValueNetwork(state_size)
        self.optimizer = optim.Adam(list(self.policy.parameters()) + list(self.value_net.parameters()), lr=0.001)

    def select_action(self, state):
        # Select action based on current state (using PPO policy)
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.policy(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        return action.item()

    def train(self, state, action, reward, next_state, done):
        # Train RL agent using PPO algorithm
        state = torch.FloatTensor(state).unsqueeze(0)
        action = torch.tensor([action])
        reward = torch.tensor([reward], dtype=torch.float32)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)

        # Calculate advantage
        advantage = self.calculate_advantage(state, reward, next_state, done)

        # Policy loss
        action_probs = self.policy(state).gather(1, action.unsqueeze(1))
        action_probs_next = self.policy(next_state).gather(1, action.unsqueeze(1))
        ratio = torch.exp(torch.log(action_probs) - torch.log(action_probs_next))
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * advantage
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value function loss
        value = self.value_net(state)
        next_value = self.value_net(next_state)
        value_target = reward + (1 - done) * next_value
        value_loss = F.smooth_l1_loss(value, value_target.detach())

        # Total loss
        loss = policy_loss + 0.5 * value_loss

        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def calculate_advantage(self, state, reward, next_state, done):
        # Placeholder for advantage calculation
        # Replace with actual advantage calculation based on reward and value function
        return reward

    def save_model(self, filename):
        # Save trained model
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filename)

    def load_model(self, filename):
        # Load trained model
        checkpoint = torch.load(filename)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_probs = F.softmax(self.fc2(x), dim=-1)
        return action_probs

class ValueNetwork(nn.Module):
    def __init__(self, state_size, hidden_size=64):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.fc2(x)
        return value
