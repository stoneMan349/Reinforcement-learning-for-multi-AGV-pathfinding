import gym, os
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from numpy import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("CartPole-v1", render_mode="rgb_array").unwrapped
# render_mode = human, rgb_array和ansi

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
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
        # print("state", state)
        output = F.relu(self.linear1(state))
        # print("output_1", output)
        output = F.relu(self.linear2(output))
        # print("output_2", output)
        output = self.linear3(output)
        # print("F.softmax(output, dim=-1)", F.softmax(output, dim=-1))
        # print("output", output)
        distribution = Categorical(F.softmax(output, dim=-1))
        return distribution


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
        # print("value", value)
        return value


def compute_returns(next_value, rewards, masks, gamma=0.99):
    # R = next_value
    R = 0
    # print("R", R)
    returns = []
    for step in reversed(range(len(rewards))):
        # print("step", step)
        # print("len(rewards)", len(rewards))
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


def trainIters(actor, critic, n_iters):
    optimizerA = optim.Adam(actor.parameters())
    optimizerC = optim.Adam(critic.parameters())
    score_i = []
    for iter in range(n_iters):
        state = env.reset()[0]
        # print("state", state)
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0
        env.reset()

        for i in count():
            env.render()
            state = torch.FloatTensor(state).to(device)
            dist, value = actor(state), critic(state)
            # print("dist", dist)

            action = dist.sample()
            # print("action", action)
            back_info = env.step(action.cpu().numpy())
            next_state, reward, done = back_info[0], back_info[1], back_info[2]
            # print("reward", reward)

            log_prob = dist.log_prob(action).unsqueeze(0)
            entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
            masks.append(torch.tensor([1-done], dtype=torch.float, device=device))

            state = next_state
            # print("i", i)
            # print("len(rewards)", len(rewards))
            # print("len(values)", len(values))

            if done or i > 500:
                print('Iteration: {}, Score: {}'.format(iter, i))
                break

        score_i.append(i)
        # print(score_i)
        score_j = score_i[::]
        score_j.reverse()
        aaaa = score_j[0:100]
        if mean(aaaa) > 1000:
            break
        # print(mean(aaaa))
        # print("aaaa", aaaa)

        next_state = torch.FloatTensor(next_state).to(device)
        next_value = critic(next_state)
        # print("rewards", rewards)
        returns = compute_returns(next_value, rewards, masks)
        # print("log_probs", log_probs)

        log_probs = torch.cat(log_probs)
        print("log_probs", log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        print("returns", returns)
        print("values", values)
        advantage = returns - values
        print("advantage", advantage)
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        optimizerA.zero_grad()
        optimizerC.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        optimizerA.step()
        optimizerC.step()
    plt.figure()
    plt.plot(score_i, color="g")
    plt.show()
    # torch.save(actor, 'model/actor.pkl')
    # torch.save(critic, 'model/critic.pkl')
    env.close()


if __name__ == '__main__':
    if os.path.exists('model/actor.pkl'):
        actor = torch.load('model/actor.pkl')
        print('Actor Model loaded')
    else:
        actor = Actor(state_size, action_size).to(device)
    if os.path.exists('model/critic.pkl'):
        critic = torch.load('model/critic.pkl')
        print('Critic Model loaded')
    else:
        critic = Critic(state_size, action_size).to(device)
    trainIters(actor, critic, n_iters=500)