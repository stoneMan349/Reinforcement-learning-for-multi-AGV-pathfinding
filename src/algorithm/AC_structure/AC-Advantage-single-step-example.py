import gym
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("CartPole-v1", render_mode="rgb_array").unwrapped
# render_mode = human, rgb_array和ansi

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
gamma = 0.99
lr = 0.0001


class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, self.action_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        output = F.softmax(output, dim=-1)
        return output


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value


def trainIters(actor, critic, n_iters):
    optimizerA = optim.Adam(actor.parameters())
    optimizerC = optim.Adam(critic.parameters())
    score_i = []
    for iter in range(n_iters):
        state = env.reset()[0]
        env.reset()
        for i in count():
            prob, value = actor(torch.FloatTensor(state).to(device)), critic(torch.FloatTensor(state).to(device))
            dist =Categorical(prob)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            back_info = env.step(action.cpu().numpy())
            next_state, reward, done = back_info[0], back_info[1], back_info[2]

            # 单步训练
            # state_val and next_state_val
            state_val = critic(torch.FloatTensor(state).to(device))
            next_state_val = critic(torch.FloatTensor(next_state).to(device))
            state = next_state
            if done:
                next_state_val = torch.tensor([-1]).float().unsqueeze(0).to(device)
            # loss value
            reward = torch.tensor([reward], dtype=torch.float, device=device)
            critic_loss = F.mse_loss(reward + gamma*next_state_val, state_val)
            advantage = reward + gamma * next_state_val.item() - state_val.item()
            actor_loss = -log_prob * advantage

            # backward
            optimizerA.zero_grad()
            optimizerC.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            optimizerA.step()
            optimizerC.step()

            if done or i > 500:
                print('Iteration: {}, Score: {}'.format(iter, i))
                break
        score_i.append(i)
    plt.figure()
    plt.plot(score_i, color="g")
    plt.show()
    env.close()


if __name__ == '__main__':
    actor = Actor(state_size, action_size).to(device)
    critic = Critic(state_size, action_size).to(device)
    trainIters(actor, critic, n_iters=2000)