import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from collections import namedtuple
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import torch.optim as optim
import gym
np.set_printoptions(threshold=np.inf)

# architecture used for layout smallGrid
""" Actor Critic """
"pytorch神经网络要求的输入都是mini-batch型的，维度为[batch_size, channels, w, h]"


class Actor(nn.Module):
    def __init__(self, num_inputs=1, num_actions=4, map_xdim=9, map_ydim=10):
        super(Actor, self).__init__()
        # structure of neural network_picture
        node_num_1 = 32
        node_num_2 = 64
        node_num_3 = 128
        node_num_4 = 256

        self.conv1 = nn.Conv2d(num_inputs, node_num_1, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv2 = nn.Conv2d(node_num_1, node_num_2, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv_pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv3 = nn.Conv2d(node_num_2, node_num_3, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv4 = nn.Conv2d(node_num_3, node_num_4, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv_pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc1 = nn.Linear(node_num_4*int(map_xdim/4)*int(map_ydim/4), 256)
        self.fc_action = nn.Linear(256, num_actions)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv_pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv_pool2(x)

        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x_action = self.fc_action(x)
        x_action = F.softmax(x_action, dim=-1)
        return x_action


class Critic(nn.Module):
    def __init__(self, num_inputs=1, num_actions=4, map_xdim=9, map_ydim=10):
        super(Critic, self).__init__()
        # structure of neural network_picture
        node_num_1 = 32
        node_num_2 = 64
        node_num_3 = 128
        node_num_4 = 256

        self.conv1 = nn.Conv2d(num_inputs, node_num_1, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv2 = nn.Conv2d(node_num_1, node_num_2, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv_pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv3 = nn.Conv2d(node_num_2, node_num_3, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv4 = nn.Conv2d(node_num_3, node_num_4, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv_pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc1 = nn.Linear(node_num_4 * int(map_xdim / 4) * int(map_ydim / 4), 256)
        self.fc_2 = nn.Linear(256, 128)
        self.fc_state = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv_pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv_pool2(x)

        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = F.relu(self.fc_2(x))
        x_state = self.fc_state(x)
        return x_state


class ActorLinear(nn.Module):
    def __init__(self, num_inputs=1, num_actions=4):
        super(ActorLinear, self).__init__()
        # structure of neural network_picture
        self.fc1 = nn.Linear(num_inputs, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


class CriticLinear(nn.Module):
    def __init__(self, num_inputs=1, num_actions=4):
        super(CriticLinear, self).__init__()
        # structure of neural network_picture
        self.fc1 = nn.Linear(num_inputs, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Agent:
    def __init__(self, actor_net, critic_net, device):
        # 构建device
        self.device = device
        # self.device = torch.device("cpu")
        self.actor_net = actor_net.to(self.device)
        self.critic_net = critic_net.to(self.device)

        # Hyperparameters
        self.GAMMA = 0.99  # discount factor

        self.lr_start = 0.00015
        self.lr = self.lr_start
        self.lr_end = 0.01
        self.lr_count = 0

        self.log_prob = 0
        self.state_val = 0
        self.next_state = 0
        self.reward = 0

        self.optimizerA = optim.RMSprop(self.actor_net.parameters(), 0.000005)  # RMSprop, 0.0025  0.00005
        # self.optimizerC = optim.RMSprop(self.critic_net.parameters(), 0.00005)  # AdamW, 0.00002
        self.optimizerC = optim.Adagrad(self.critic_net.parameters(), 0.0001)  # AdamW, 0.00002

        # self.optimizerA = optim.Adam(self.actor_net.parameters(), 0.0001)
        # self.optimizerC = optim.Adam(self.critic_net.parameters(), 0.0001)

        # self.optim = torch.optim.RMSprop(self.policy_network.parameters(), self.lr, alpha=0.95, eps=0.01)  # 效果一般，收敛性不够好
        # self.optim = torch.optim.Adam(self.policy_network.parameters(), self.lr)   # 效果不好
        # self.optim = torch.optim.SGD(self.policy_network.parameters(), self.lr)  # 效果好,lr=0.1  # 比RMSprop好
        # self.optim = torch.optim.ASGD(self.policy_network.parameters(), self.lr)  # 效果好，lr=0.1  # 比RMSprop好，比Adagrad差
        # self.optim = torch.optim.Adagrad(self.policy_network.parameters(), self.lr)  # 效果好，lr=0.01  # 训练效果好，但是会出现死锁
        # self.optim = torch.optim.Adadelta(self.policy_network.parameters(), self.lr)  # 效果不好，lr=1
        # self.optim = torch.optim.Adamax(self.policy_network.parameters(), self.lr)  # 效果不好，lr=0.001
        # self.optim = torch.optim.Rprop(self.policy_network.parameters(), self.lr)  # 效果不好
        # self.optim = torch.optim.AdamW(self.policy_network.parameters(), self.lr)  # 效果不好
        self.loss_function = torch.nn.SmoothL1Loss()
        self.loss_value = []

    def choose_action(self, obs):
        state = torch.from_numpy(obs).float().unsqueeze(0)
        state = state.to(self.device)  # transform to GPU
        prob, value = self.actor_net(state), self.critic_net(state)
        act_dist = Categorical(prob)
        action = act_dist.sample()

        log_prob = act_dist.log_prob(action)
        self.log_prob = log_prob
        self.state_val = value
        return action.item()

    def store_transition(self, reward, obs_next):
        self.next_state = obs_next
        self.reward = reward
        self.update_network()

    def update_network(self):
        state_val = self.state_val
        next_state = torch.from_numpy(self.next_state).float().unsqueeze(0)
        next_state = next_state.to(self.device)  # transform to GPU
        next_state_val = self.critic_net(next_state)

        critic_loss = F.mse_loss(self.reward + self.GAMMA * next_state_val, state_val)
        advantage = self.reward + self.GAMMA * next_state_val.item() - state_val.item()
        actor_loss = -self.log_prob * advantage

        self.optimizerA.zero_grad()
        self.optimizerC.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.optimizerA.step()
        self.optimizerC.step()

        self.loss_value.append(actor_loss.detach().cpu().item())

        # store loss value
        # if loss.item() >= 0.5:
        #     self.loss_value.append(0.5)
        # else:
        #     self.loss_value.append(loss.item())


""""Structure test with an example of Cartpole Game and linear NNs"""
"""recommended optimizer and learning : 
    self.optimizerA = optim.Adam(self.actor_net.parameters(), 0.001)
    self.optimizerC = optim.Adam(self.critic_net.parameters(), 0.001)"""
if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode="rgb_array")
    action_size = env.action_space.n
    state_size = env.observation_space.shape

    actor_net = ActorLinear(state_size[0], action_size)
    critic_net = CriticLinear(state_size[0], action_size)
    agent = Agent(actor_net, critic_net)

    episode = 1000  # max number of episode
    step = 250  # max value in one episode

    reward_total = list()
    for ep in range(episode):
        state = env.reset()[0]  # init environment
        for i in range(step):
            # print(state)
            act = agent.choose_action(state)  # choose action
            # perform action
            back_info = env.step(act)
            n_state, reward, terminal, _ = back_info[0], back_info[1], back_info[2], back_info[3]

            state = n_state
            if terminal:  # gym环境不会给最后的动作负的奖励值
                reward = -1
            agent.store_transition(reward, n_state)
            if terminal:
                break
        print(ep, i)
        reward_total.append(i)

    plt.figure()
    plt.plot(reward_total)
    plt.show()
    # self.optimizerA = optim.Adam(self.actor_net.parameters(), 0.001)  # Adam
    # self.optimizerC = optim.Adam(self.critic_net.parameters(), 0.001)



