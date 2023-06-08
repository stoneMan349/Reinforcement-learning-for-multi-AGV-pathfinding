from collections import deque
import random
import torch.nn as nn
import torch.nn.functional as F
import torch
import time
import numpy as np
# import SumTree
import src.utils.astar as astar
np.set_printoptions(threshold=np.inf)

# architecture used for layout smallGrid
""" Deep Q Network """


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

        self.fc3 = nn.Linear(node_num_4*int(map_xdim/4)*int(map_ydim/4), 256)
        self.fc4 = nn.Linear(256, num_actions)
        self.dropout = nn.Dropout(p=0.5)  # dropout训练

        # info for training
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv_pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv_pool2(x)

        x = F.relu(self.fc3(x.view(x.size(0), -1)))
        x = self.fc4(x)
        return x


class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])


class Memory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)


class Agent:
    def __init__(self, policy_net, target_net):
        # 构建device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")

        self.policy_network = policy_net.to(self.device)
        self.target_network = target_net.to(self.device)

        # info for training
        self.epsilon_start = 0.8
        self.epsilon = self.epsilon_start
        self.epsilon_end = 1
        self.epsilon_count = 0

        # memory replay and score databases
        self.replay_mem = deque()
        self.memory_size = 50000  # 经验池条数
        self.start_training_info_number = 100  # 开始训练时的经验条数
        self.learn_step_counter = 0
        self.TARGET_REPLACE_ITER = 100  # update network_picture step
        self.batch_size = 32  # memory replay batch size
        self.GAMMA = 0.95  # discount factor
        # create prioritized replay memory using SumTree
        self.memory = Memory(self.memory_size)

        self.lr_start = 0.01
        self.lr = self.lr_start
        self.lr_end = 0.01
        self.lr_count = 0

        # self.optim = torch.optim.RMSprop(self.policy_network.parameters(), self.lr, alpha=0.95, eps=0.01)  # 效果一般，收敛性不够好
        # self.optim = torch.optim.Adam(self.policy_network.parameters(), self.lr)   # 效果不好
        # self.optim = torch.optim.SGD(self.policy_network.parameters(), self.lr)  # 效果好,lr=0.1  # 比RMSprop好
        # self.optim = torch.optim.ASGD(self.policy_network.parameters(), self.lr)  # 效果好，lr=0.1  # 比RMSprop好，比Adagrad差
        self.optim = torch.optim.Adagrad(self.policy_network.parameters(), self.lr)  # 效果好，lr=0.01  # 训练效果好，但是会出现死锁
        # self.optim = torch.optim.Adadelta(self.policy_network.parameters(), self.lr)  # 效果不好，lr=1
        # self.optim = torch.optim.Adamax(self.policy_network.parameters(), self.lr)  # 效果不好，lr=0.001
        # self.optim = torch.optim.Rprop(self.policy_network.parameters(), self.lr)  # 效果不好
        # self.optim = torch.optim.AdamW(self.policy_network.parameters(), self.lr)  # 效果不好
        self.loss_function = torch.nn.SmoothL1Loss()
        self.loss_value = []

    def choose_action(self, obs, current_place, target_place, valid_path_matrix, matrix_padding=0):
        if np.random.uniform() < self.epsilon:  # greedy
            state = torch.from_numpy(obs).float().unsqueeze(0)  # array to torch
            # state = torch.unsqueeze(state, 0)
            state = state.to(self.device)  # transform to GPU
            t_s = time.time()
            actions_value = self.policy_network.forward(state)  # get action
            t_e = time.time()
            action = torch.max(actions_value.cpu(), 1)[1].data.numpy()
            # action = np.random.randint(0, 4)  # (完全随机)
            # action = np.array([action])
        else:  # a_star
            t_s = time.time()
            action = self.find_action_astar(valid_path_matrix, current_place, target_place)
            t_e = time.time()
            action = np.array([action])
        t_ = t_e-t_s
        return action, t_

    def choose_action_as(self, current_place, target_place, valid_path_matrix, matrix_padding=0):
        action = self.find_action_astar(valid_path_matrix, current_place, target_place, matrix_padding)
        action = np.array([action])
        return action

    def find_action_astar(self, matrix_valid_map, current_position, target_position):
        path_founder = astar.FindPathAstar(matrix_valid_map, (current_position[0]-1, current_position[1]-1),
                                           (target_position[0]-1, target_position[1]-1))
        find_target, path_list, path_map, action_list = path_founder.run_astar_method()
        if find_target == False or len(action_list) == 0:
            action_str = "STOP"
        else:
            action_str = action_list[0]
        return self.get_value(action_str)

    def get_value(self, direction):
        if direction == 'UP':
            return 0.
        if direction == 'RIGHT':
            return 1.
        if direction == 'DOWN':
            return 2.
        if direction == 'LEFT':
            return 3.
        if direction == 'STOP':
            return 4.

    def store_transition(self, s, a, r, s_, is_done):
        """store experience in a 'prioritized replay' way"""
        """value by prediction"""
        state = torch.from_numpy(s).float().unsqueeze(0)  # array to torch
        state = state.to(self.device)  # transform to GPU
        target = self.policy_network.forward(state).cpu()  # get action
        a = int(a)
        # print("a", a)
        old_val = target[0][a]
        """value by reward"""
        state = torch.from_numpy(s_).float().unsqueeze(0)  # array to torch
        state = state.to(self.device)  # transform to GPU
        target_val = self.policy_network.forward(state)  # get action
        if is_done == 1:
            new_val = r
        else:
            new_val = r + self.GAMMA * torch.max(target_val)
        """difference between old_val and new_val"""
        error = abs(old_val - new_val).cpu()
        error = error.detach().numpy()
        self.memory.add(error, (np.array(s), a, r, np.array(s_), is_done))

        if self.memory.tree.n_entries >= self.start_training_info_number:
            self.update_network()

    def update_network(self):
        if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:  # 一开始触发，然后每100步触发
            self.target_network.load_state_dict(self.policy_network.state_dict())  # 将评估网络的参数赋给目标网络
        self.learn_step_counter += 1

        # 数据抽样
        # batch = random.sample(self.replay_mem, self.batch_size)
        batch, idxs, is_weights = self.memory.sample(self.batch_size)
        batch_s, batch_a, batch_r, batch_n, batch_is_done = zip(*batch)

        # convert from numpy to pytorch
        batch_s = torch.from_numpy(np.stack(batch_s)).float().to(self.device)  # .to(torch.float32)
        # print("batch_s", batch_s)
        batch_r = torch.Tensor(batch_r).unsqueeze(1).to(self.device)
        # print(batch_r)
        print("batch_a", batch_a)
        batch_a = torch.LongTensor(batch_a).unsqueeze(1).to(self.device)
        print("batch_a_", batch_a)
        batch_n = torch.from_numpy(np.stack(batch_n)).float().to(self.device)
        batch_is_done = torch.LongTensor(batch_is_done).unsqueeze(1).to(self.device)

        state_action_values = self.policy_network(batch_s).gather(1, batch_a)

        # get V(s') DDQN修改
        next_state_values = self.target_network(batch_n)
        # Compute the expected Q values
        next_state_values = next_state_values.detach().max(1)[0]
        next_state_values = next_state_values.unsqueeze(1)
        # DDQN修改
        next_state_values_ = self.target_network(batch_n)
        next_state_values__ = self.policy_network(batch_n)

        # print("torch.max(next_state_values__, 1)[1].unsqueeze(1)", torch.max(next_state_values__, 1)[1].unsqueeze(1))
        next_state_values___ = next_state_values_.gather(1, torch.max(next_state_values__, 1)[1].unsqueeze(1))
        # print("next_state_values___", next_state_values___)

        # expected_state_action_values = (next_state_values * self.GAMMA)*(1-batch_is_done) + batch_r
        expected_state_action_values = (next_state_values___ * self.GAMMA) * (1 - batch_is_done) + batch_r  # DDQN

        # calculate loss
        # print("state_action_values", state_action_values)
        # print("expected_state_action_values", expected_state_action_values)
        loss = self.loss_function(state_action_values, expected_state_action_values)
        # optimize model - update weights
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # store loss value
        if loss.item() >= 0.5:
            self.loss_value.append(0.5)
        else:
            self.loss_value.append(loss.item())

    def change_learning_rate(self, times):
        if self.lr_count == times:
            print("the value of current learning rate is {}".format(self.lr))
        if self.lr_count > times:
            return
        else:
            self.lr = self.lr - (self.lr_start - self.lr_end)/times
        self.lr_count += 1

    def change_explore_rate(self, times):
        if self.epsilon_count >= times:
            self.epsilon = self.epsilon_end
        else:
            self.epsilon = self.epsilon + (self.epsilon_end - self.epsilon_start)/times
        self.epsilon_count += 1

        if self.epsilon_count == times:
            print("exploring rate is 1.")





