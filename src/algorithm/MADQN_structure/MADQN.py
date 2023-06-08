from collections import deque
import torch.nn as nn
import torch.nn.functional as F
import torch
import time  # t_e = time.time()
import numpy as np
from algorithm.MADQN_structure.PER import Memory as Memory
import src.utils.astar as astar
from src.utils.utils import Direction as Dir
np.set_printoptions(threshold=np.inf)


class Net(nn.Module):
    def __init__(self, num_inputs=1, num_actions=4, map_xdim=9, map_ydim=10):
        super(Net, self).__init__()
        """structure of neural network_picture"""
        # neutron numbers
        node_num_1 = 32
        node_num_2 = 64
        node_num_3 = 128
        node_num_4 = 256
        # convolutional layers
        self.conv1 = nn.Conv2d(num_inputs, node_num_2, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv2 = nn.Conv2d(node_num_2, node_num_2, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv_pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv3 = nn.Conv2d(node_num_2, node_num_3, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv4 = nn.Conv2d(node_num_3, node_num_3, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv_pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv5 = nn.Conv2d(node_num_3, node_num_4, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv6 = nn.Conv2d(node_num_4, node_num_4, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv_pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # linear layers
        # self.fc3 = nn.Linear(node_num_4*int(map_xdim/4)*int(map_ydim/4), 256)
        self.fc3 = nn.Linear(2304, 256)
        # self.fc3 = nn.Linear(node_num_4 * int(map_xdim / 2) * int(map_ydim / 2), 256)
        self.fc4 = nn.Linear(256, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = self.conv_pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        # x = self.conv_pool2(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.conv_pool3(x)

        x = F.relu(self.fc3(x.view(x.size(0), -1)))
        x = self.fc4(x)
        return x


class Agent:
    def __init__(self, policy_net, target_net, device):
        """init agent"""
        """create device"""
        self.device = device
        """create policy and target network"""
        self.policy_network = policy_net.to(self.device)
        self.target_network = target_net.to(self.device)
        """parameters for exploring"""
        self.epsilon_start = 0.8
        self.epsilon = self.epsilon_start
        self.epsilon_end = 1
        self.epsilon_count = 0
        """memory replay and score databases"""
        self.memory_size = 50000  # 经验池条数
        self.start_training_info_number = 100  # 开始训练时的经验条数
        self.learn_step_counter = 0
        self.TARGET_REPLACE_ITER = 100  # update network_picture step
        self.batch_size = 32  # memory replay batch size
        self.GAMMA = 0.95  # discount factor
        """create prioritized replay memory using SumTree"""
        self.memory = Memory(self.memory_size)
        """learning rate"""
        self.lr_start = 0.01
        self.lr = self.lr_start
        self.lr_end = 0.001
        self.lr_count = 0
        """optimizers"""
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
        """loss function"""
        self.loss_value = []
        self.dir = Dir()

    def choose_action(self, obs, current_place, target_place, valid_path_matrix, epsilon=0.):
        epsilon = max(epsilon, self.epsilon)  # use_nn的时候，传入epsilon=1
        if np.random.uniform() < epsilon:  # greedy
            # state
            state = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)  # array to torch
            # action
            actions_value = self.policy_network.forward(state)  # get action
            action = torch.max(actions_value.cpu(), 1)[1].data.numpy()
            # action = np.random.randint(0, 4)  # (完全随机)
            # action = np.array([action])
        else:  # a_star
            action = self.find_action_astar(valid_path_matrix, current_place, target_place)
            action = np.array([action])
            # action = np.random.randint(0, 4, (1))
            # print("action", action)
        return action

    def store_transition(self, s, a, r, s_, is_done):
        """store experience in a 'prioritized replay' way"""
        """value by prediction"""
        state = torch.from_numpy(s).float().unsqueeze(0).to(self.device)  # array to torch
        target = self.policy_network.forward(state).cpu()  # get action
        old_val = target[0][int(a)]
        """value by reward"""
        state = torch.from_numpy(s_).float().unsqueeze(0).to(self.device)  # array to torch
        target_val = self.policy_network.forward(state)  # get action
        if is_done == 1:
            new_val = r
        else:
            new_val = r + self.GAMMA * torch.max(target_val)
        """difference between old_val and new_val"""
        error = abs(old_val - new_val).cpu()
        error = error.detach().numpy()
        self.memory.add(error, (np.array(s), a, r, np.array(s_), is_done))
        """"update NN"""
        if self.memory.tree.n_entries >= self.start_training_info_number:
            self.update_network()

    def update_network(self):
        """update target network"""
        if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:  # update Target Network per 100 steps
            self.target_network.load_state_dict(self.policy_network.state_dict())  # Copy the parameters of P to T
        self.learn_step_counter += 1
        """sample data"""
        batch, idxs, is_weights = self.memory.sample(self.batch_size)
        batch_s, batch_a, batch_r, batch_n, batch_is_done = zip(*batch)
        """convert from numpy to pytorch"""
        batch_s = torch.from_numpy(np.stack(batch_s)).float().to(self.device)
        batch_r = torch.Tensor(batch_r).unsqueeze(1).to(self.device)
        batch_a = torch.LongTensor(batch_a).unsqueeze(1).to(self.device)
        batch_n = torch.from_numpy(np.stack(batch_n)).float().to(self.device)
        batch_is_done = torch.LongTensor(batch_is_done).unsqueeze(1).to(self.device)
        """get Q(s)"""
        state_action_values = self.policy_network(batch_s).gather(1, batch_a)
        """Standard DQN: Compute the expected Q values"""
        # next_state_values = self.target_network(batch_n)
        # next_state_values = next_state_values.detach().max(1)[0]
        # next_state_values = next_state_values.unsqueeze(1)
        # expected_state_action_values = (next_state_values * self.GAMMA)*(1-batch_is_done) + batch_r
        """DDQN"""
        next_state_values_target = self.target_network(batch_n)
        next_state_values_policy = self.policy_network(batch_n)
        next_state_values = next_state_values_target.gather(1, torch.max(next_state_values_policy, 1)[1].unsqueeze(1))
        expected_state_action_values = (next_state_values * self.GAMMA) * (1 - batch_is_done) + batch_r
        """calculate loss, update weights"""
        loss = self.loss_function(state_action_values, expected_state_action_values)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        """store loss value"""
        if loss.item() >= 0.5:
            self.loss_value.append(0.5)
        elif loss.item() <= -0.5:
            self.loss_value.append(-0.5)
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

    def find_action_astar(self, matrix_valid_map, current_position, target_position):
        path_founder = astar.FindPathAstar(matrix_valid_map, (current_position[0] - 1, current_position[1] - 1),
                                           (target_position[0] - 1, target_position[1] - 1))
        find_target, path_list, path_map, action_list = path_founder.run_astar_method()
        if find_target is False or len(action_list) == 0:
            action_str = "STOP"
        else:
            action_str = action_list[0]
        return self.dir.action_str_value(action_str)
