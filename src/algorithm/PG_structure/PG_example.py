import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
from torch.distributions import Categorical
import gym
import numpy as np
import matplotlib.pyplot as plt


class PG_policy_net(nn.Module):
    def __init__(self, state_size=4, action_size=2):
        super(PG_policy_net, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, action_size)

    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.softmax(self.fc2(X), dim=1)
        return X


class PG_Agent():
    def __init__(self, state_size=2, action_size=3, lr=1e-3, gamma=0.95):
        self.state_size = state_size
        self.action_size = action_size
        self.policy = PG_policy_net(state_size=state_size, action_size=action_size)
        self.lr = lr
        self.gamma = gamma
        self.optim = Adam(self.policy.parameters(), lr=self.lr)

    def get_action(self, state):
        prob = self.policy(state)
        print("prob", prob)
        act_dist = Categorical(prob)
        print("act_dist", act_dist)
        action = act_dist.sample()
        # print("action", action)
        return int(action), act_dist.log_prob(action)

    def reward_process(self, reward_mem):
        discounted_reward = np.zeros_like(np.array(reward_mem))
        running_add = 0
        for t in reversed(range(0, len(reward_mem))):
            running_add = running_add * self.gamma + reward_mem[t]
            discounted_reward[t] = running_add
        # print("discounted_reward", discounted_reward)
        discounted_reward -= np.mean(discounted_reward)
        discounted_reward /= np.std(discounted_reward)
        # print("discounted_reward", discounted_reward)
        return discounted_reward

    def episode_learn(self, reward_mem, act_log_mem):
        reward_mem = self.reward_process(reward_mem)
        policy_loss = list()
        for log_prob, reward in zip(act_log_mem, reward_mem):
            print("log_prob", log_prob)
            print("reward", reward)
            policy_loss.append(-log_prob * reward)
        print("policy_loss", policy_loss)
        self.policy.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        self.optim.step()


env = gym.make('CartPole-v1')
action_size = env.action_space.n
state_size = env.observation_space.shape

agent = PG_Agent(state_size=state_size[0], action_size=action_size, lr=1e-3)  # lr=5e-4

episode = 1500
step = 200

reward_total = list()
for ep in range(episode):
    reward_mem = list()
    act_log_mem = list()
    state = env.reset()[0]
    for i in range(step):
        state = Variable(torch.FloatTensor(state))
        shape = [1]
        shape.extend(list(state.size()))
        state = state.view(shape)
        act, act_log_prob = agent.get_action(state)
        # print("act_log_prob", act_log_prob)

        back_info = env.step(act)
        n_state, reward, terminal, _ = back_info[0], back_info[1], back_info[2], back_info[3]

        # n_state, reward, terminal, _ = env.step(act)
        reward_mem.append(reward)
        act_log_mem.append(act_log_prob)
        if terminal:
            # print(ep, i)
            break
        state = n_state
    print(ep, i)
    reward_total.append(sum(reward_mem))
    agent.episode_learn(reward_mem, act_log_mem)

plt.plot(reward_total)
plt.show()