import torch.nn as nn
import torch.nn.functional as F
import torch
import gym
import numpy as np
from torch.distributions import Categorical
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)

""" Policy Gradient """


class Net(nn.Module):
    def __init__(self, num_inputs=1, num_actions=4, map_xdim=9, map_ydim=10):
        super(Net, self).__init__()
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
        # self.fc1 = nn.Linear(1024, 256)
        self.fc_action = nn.Linear(256, num_actions)
        # info for training

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv_pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv_pool2(x)

        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x_action = self.fc_action(x)
        x_action = F.softmax(x_action, dim=1)
        return x_action


class NetLinear(nn.Module):
    def __init__(self, state_size=4, action_size=2):
        super(NetLinear, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, action_size)

    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.softmax(self.fc2(X), dim=1)
        return X


class Agent:
    def __init__(self, pg_net, device):
        """Agent: control training"""
        """device and neural network"""
        self.device = device
        self.pg_net = pg_net.to(self.device)
        """Hyper-parameters"""
        self.eps = 1e-8  # self.eps = np.finfo(np.float32).eps.item()
        self.GAMMA = 0.95  # discount factor
        """learning rate"""
        self.lr_start = 0.00010  # 0.000005专家训练使用的是该值，没有专家则使用0.0001
        self.lr = self.lr_start
        self.lr_end = 0.01
        self.train_times = 0
        """data storage"""
        self.act_log_mem = []
        self.reward_mem = []
        self.loss_record = []
        self.optim = torch.optim.RMSprop(self.pg_net.parameters(), lr=self.lr)  # RMSprop, lr=0.00015, WORKS! 0.00008 works

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
        obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        prob = self.pg_net(obs)
        act_dist = Categorical(prob)
        action = act_dist.sample()
        act_log = act_dist.log_prob(action)
        self.act_log_mem.append(act_log)
        return action.item()

    def store_transition(self, reward, is_done):
        self.reward_mem.append(reward)
        if is_done:
            self.update_network()

    def reward_process(self):
        discounted_reward = []
        running_add = 0
        for t in reversed(range(0, len(self.reward_mem))):
            running_add = running_add * self.GAMMA + self.reward_mem[t]
            discounted_reward.insert(0, running_add)
        discounted_reward = discounted_reward - np.mean(discounted_reward)
        discounted_reward = discounted_reward/(np.std(discounted_reward)+self.eps)
        return discounted_reward

    def update_network(self):
        self.train_times += 1
        self.change_learning_rate()

        reward_mem = self.reward_process()
        policy_loss = list()
        for log_prob, reward in zip(self.act_log_mem, reward_mem):
            policy_loss.append(-log_prob * reward)
        """update"""
        self.pg_net.zero_grad()
        loss = torch.stack(policy_loss).sum()
        self.loss_value.append(loss.item())
        loss.backward()
        self.optim.step()

        self.act_log_mem = []
        self.reward_mem = []

    def change_learning_rate(self):
        if self.train_times == 4250:
            # self.lr = self.lr*0.5
            self.lr = self.lr


""""Structure test with an example of Cartpole Game and a linear NN"""
"""recommended optimizer and learning : self.optim = torch.optim.Adam(self.pg_net.parameters(), lr=0.001)"""
if __name__ == '__main__':
    # main()
    env = gym.make('CartPole-v1')
    action_size = env.action_space.n
    state_size = env.observation_space.shape

    pg_net = NetLinear(state_size=state_size[0], action_size=action_size)
    agent = Agent(pg_net)

    episode = 500
    step = 200

    reward_total = list()
    for ep in range(episode):
        state = env.reset()[0]
        for i in range(step):
            act = agent.choose_action(state)

            back_info = env.step(act)
            n_state, reward, terminal, _ = back_info[0], back_info[1], back_info[2], back_info[3]

            agent.store_transition(reward, False)  # in this example, update_network is not activate by store_transition
            if terminal:
                break
            state = n_state
        print(ep, i)
        reward_total.append(sum(agent.reward_mem))
        agent.update_network()

    plt.figure()
    plt.plot(reward_total)
    plt.show()




