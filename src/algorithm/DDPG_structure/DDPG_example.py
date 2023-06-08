# Lillicrap, Timothy P., et al. "Continuous control with deep reinforcement learning." arXiv preprint arXiv:1509.02971 (2015).
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import gym
import random
from collections import deque


class ReplayBuffer():
    def __init__(self, max_size=100000):
        super(ReplayBuffer, self).__init__()
        self.max_size = max_size
        self.memory = deque(maxlen=self.max_size)

    # Add the replay memory
    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Sample the replay memory
    def sample(self, batch_size):
        batch = random.sample(self.memory, min(batch_size, len(self.memory)))
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return states, actions, rewards, next_states, dones


class ActorNet(nn.Module):
    def __init__(self, state_num, action_num, min_action, max_action, bn=False):
        super(ActorNet, self).__init__()
        self.input = nn.Linear(state_num, 256)
        self.fc = nn.Linear(256, 512)
        self.output = nn.Linear(512, action_num)

        # Batch normalization
        self.bn = bn
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(512)

        # Get the action interval for clipping
        self.min_action = min_action
        self.max_action = max_action

    def forward(self, x):
        if self.bn:
            x = F.relu(self.bn1(self.input(x)))
            x = F.relu(self.bn2(self.fc(x)))
        else:
            x = F.relu(self.input(x))
            x = F.relu(self.fc(x))

        action = self.output(x)
        action = torch.clamp(action, self.min_action, self.max_action)
        return action


class CriticNet(nn.Module):
    def __init__(self, state_num, action_num, bn=False):
        super(CriticNet, self).__init__()
        self.input = nn.Linear(state_num + action_num, 256)
        self.fc = nn.Linear(256, 512)
        self.output = nn.Linear(512, 1)

        # Batch normalization
        self.bn = bn
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(512)

    def forward(self, x, u):
        x = torch.cat([x, u], 1)

        if self.bn:
            x = F.relu(self.bn1(self.input(x)))
            x = F.relu(self.bn2(self.fc(x)))
        else:
            x = F.relu(self.input(x))
            x = F.relu(self.fc(x))

        value = self.output(x)
        return value


class DDPG():
    def __init__(self, env, memory_size=10000000, batch_size=64, tau=0.01, gamma=0.95, learning_rate=1e-3, eps_min=0.05,
                 eps_period=10000, bn=False):
        super(DDPG, self).__init__()
        self.env = env
        self.state_num = self.env.observation_space.shape[0]
        self.action_num = self.env.action_space.shape[0]
        self.action_max = float(env.action_space.high[0])
        self.action_min = float(env.action_space.low[0])

        # Torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Actor
        self.actor_net = ActorNet(self.state_num, self.action_num, self.action_min, self.action_max, bn).to(self.device)
        self.actor_opt = optim.Adam(self.actor_net.parameters(), lr=learning_rate)

        # Target Actor
        self.actor_target_net = ActorNet(self.state_num, self.action_num, self.action_min, self.action_max, bn).to(
            self.device)
        self.actor_target_net.load_state_dict(self.actor_net.state_dict())

        # Critic
        self.critic_net = CriticNet(self.state_num, self.action_num, bn).to(self.device)
        self.critic_opt = optim.Adam(self.critic_net.parameters(), lr=learning_rate)

        # Target Critic
        self.critic_target_net = CriticNet(self.state_num, self.action_num, bn).to(self.device)
        self.critic_target_net.load_state_dict(self.critic_net.state_dict())

        # Replay buffer
        self.replay_buffer = ReplayBuffer(memory_size)
        self.batch_size = batch_size

        # Learning setting
        self.gamma = gamma
        self.tau = tau

        # Noise setting
        self.epsilon = 1
        self.eps_min = eps_min
        self.eps_period = eps_period

    # Get the action
    def get_action(self, state, exploration=True):
        self.actor_net.eval()
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor_net(state).cpu().detach().numpy().flatten()
        self.actor_net.train()

        if exploration:
            # Get noise (gaussian distribution with epsilon greedy)
            action_mean = (self.action_max + self.action_min) / 2
            action_std = (self.action_max - self.action_min) / 2
            action_noise = np.random.normal(action_mean, action_std, 1)[0]
            action_noise *= self.epsilon
            self.epsilon = self.epsilon - (
                        1 - self.eps_min) / self.eps_period if self.epsilon > self.eps_min else self.eps_min

            # Final action
            action = action + action_noise
            action = np.clip(action, self.action_min, self.action_max)
            return action

        else:
            return action

    # Soft update a target network
    def soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    # Learn the policy
    def learn(self):
        # Replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Target Q values
        next_actions = self.actor_target_net(next_states)
        target_q = self.critic_target_net(next_states, next_actions).view(1, -1)
        target_q = (rewards + self.gamma * target_q * (1 - dones))

        # Current Q values
        values = self.critic_net(states, actions).view(1, -1)

        # Calculate the critic loss and optimize the critic network
        critic_loss = F.mse_loss(values, target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Calculate the actor loss and optimize the actor network
        actor_loss = -self.critic_net(states, self.actor_net(states)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Soft update the target networks
        self.soft_update(self.critic_net, self.critic_target_net)
        self.soft_update(self.actor_net, self.actor_target_net)


def main():
    env = gym.make("Pendulum-v1")
    agent = DDPG(env, memory_size=100000, batch_size=64, tau=0.01, gamma=0.95, learning_rate=1e-3, eps_min=0.00001,
                 eps_period=100000, bn=True)
    ep_rewards = deque(maxlen=1)
    total_episode = 10000

    for i in range(total_episode):
        state = env.reset()[0]
        ep_reward = 0
        while True:
            action = agent.get_action(state, True)
            back_info = env.step(action)
            next_state, reward, done, _ = back_info[0], back_info[1], back_info[2], back_info[3]
            ep_reward += reward

            agent.replay_buffer.add(state, action, reward, next_state, done)
            if i > 2:
                agent.learn()

            if done:
                ep_rewards.append(ep_reward)
                if i % 1 == 0:
                    print("episode: {}\treward: {}".format(i, round(np.mean(ep_rewards), 3)))
                break

            state = next_state


if __name__ == '__main__':
    main()