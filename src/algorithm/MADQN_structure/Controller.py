"""
DQN algorithm applying on multi-AGV
It is actually a DDQN algorithm
The exploration strategy is replaced by A*
"""
from algorithm.MADQN_structure.MADQN import Net as Net
from algorithm.MADQN_structure.MADQN import Agent as Agent
from src.algorithm.Manager.StateManager import StateManager as stateManager
import src.algorithm.Manager.SaveManager as saveManager
import os
import datetime
import torch
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.inf)


class MADQNAgentController:
    """a link between environment and algorithm"""
    # state_number要修改
    def __init__(self, rmfs_scene, map_xdim, map_ydim, max_task, control_mode=1, state_number=4, expert_guiding=False):  # expert未启用
        print("start simulation with DQN algorithm")
        print("map_xdim:", map_xdim, "map_ydim:", map_ydim, "state_number:", state_number)
        '''received parameters'''
        self.control_mode = control_mode
        self.state_number = state_number
        self.rmfs_model = rmfs_scene
        '''create/load neural network_picture'''
        self.stateManager = stateManager()
        self.storage_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], "network_picture")
        """create device"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # self.device = torch.device("cpu")
        """create_agent"""
        self.agent = None
        self.create_agent(map_xdim, map_ydim)
        '''training parameters'''
        self.simulation_times = 150
        self.max_value = max_task*3 - 1
        self.max_value_times = 0
        self.duration_times = 100
        """"hyper parameter"""
        self.lr_start_decay = False
        self.lifelong_reward = []
        self.action_length_record = 0
        """"hyper parameter"""
        self.reward_acc = 0
        self.veh_group = []
        self.logs = []

    def create_agent(self, map_xdim, map_ydim):
        """create/load neural network_picture and create agent"""
        policy_net, target_net = None, None
        if self.control_mode == "train_NN":
            print("create NN")
            policy_net = Net(self.state_number, self.rmfs_model.action_number, map_xdim, map_ydim)
            target_net = Net(self.state_number, self.rmfs_model.action_number, map_xdim, map_ydim)
        elif self.control_mode == "use_NN":
            print("load NN")
            policy_net = torch.load(os.path.join(self.storage_path, "policy_net.pt"))
            target_net = torch.load(os.path.join(self.storage_path, "target_net.pt"))
        '''create Agent object'''
        self.agent = Agent(policy_net, target_net, self.device)

    def self_init(self):
        self.reward_acc = 0
        self.veh_group = []
        self.action_length_record = 0

    def model_run(self):  # mainloop for training/running
        start_time = datetime.datetime.now()
        print("model is controlled by neural network")
        print("Training starts at"+str(start_time))
        for i_episode in range(self.simulation_times):
            """init model and agent"""
            self.self_init()
            self.rmfs_model.init()
            """run model"""
            running_time = self.rmfs_model.run_game(control_pattern="intelligent", smart_controller=self, render=True)  # transfer the controller to the model
            """info storage and print"""
            self.lifelong_reward.append(int(self.reward_acc))
            log = 'i_episode: {},\t reward_accu: {}, \t action_length: {},\t running times: {}'.format(i_episode, self.reward_acc, self.action_length_record, running_time)
            self.logs.append(log)
            print(log)
            """调整训练参数"""
            if self.lr_start_decay:  # -------------------------------------------------------------------还没改
                self.agent.change_learning_rate(times=100)  # 改变探索率
                self.agent.change_explore_rate(times=100)  # 改变lr
            if i_episode % 100 == 0:  # neural network auto-save
                self.save_neural_network(auto=True)
            if self.check_determination(self.reward_acc):  # check whether determination condition meets
                break
        self.save_neural_network(auto=False)  # save network
        print("all_reward:" + str(self.lifelong_reward))
        self.logs.append(self.lifelong_reward)
        self.save_log()  # save logs
        end_time = datetime.datetime.now()
        print("Training ends at" + str(end_time))
        print(end_time-start_time)
        saveManager.draw_picture(self.lifelong_reward, title="Cumulative Reward", x_label="training episodes", y_label="cumulative reward",
                                 color="g", save_path=os.path.join(self.storage_path, "Cumulative Reward"), smooth=True)
        saveManager.draw_picture(self.agent.loss_value, title="Loss Value", x_label="Training steps", y_label="loss value",
                                 color="k", save_path=os.path.join(self.storage_path, "Loss Value"))
        plt.show()

    def choose_action(self, all_info, this_veh):  # all_info=[layout  , current_place, target_place]
        """build a VehObj to store information"""
        veh_found, veh_obj = False, None
        for veh in self.veh_group:
            if this_veh == veh.veh_name:
                veh_found, veh_obj = True, veh
                break
        if not veh_found:
            veh_obj = VehObj(this_veh)
            self.veh_group.append(veh_obj)
        """get observation and other info"""
        obs, this_veh_cp, this_veh_tp, valid_path_matrix = self.stateManager.create_state(all_info, this_veh, obs_clip=True)
        obs = np.array(obs)
        """get action"""
        veh_obj.obs_current = obs
        epsilon = 1. if self.control_mode == "use_NN" else 0.
        action_l = self.agent.choose_action(obs, current_place=this_veh_cp, target_place=this_veh_tp, valid_path_matrix=valid_path_matrix, epsilon=epsilon)   # state should be formatted as array
        action = action_l[0]
        """record info"""
        veh_obj.action.append(action)
        self.action_length_record += 1
        return action

    def store_info(self, all_info, reward, is_end, this_veh):
        self.reward_acc += reward
        if self.control_mode == "use_NN":
            """using NN, no need to store info and train NN"""
            return
        """get veh_obj"""
        veh_obj = None
        for veh in self.veh_group:
            if this_veh == veh.veh_name:
                veh_obj = veh
                break
        obs, this_veh_cp, this_veh_tp, valid_path_matrix = self.stateManager.create_state(all_info, this_veh, obs_clip=True)
        obs = np.array(obs)
        # 先存储经验，在保留新数据
        veh_obj.obs_next, veh_obj.reward = obs, reward
        is_done = 1 if is_end else 0
        """store info"""
        self.agent.store_transition(veh_obj.obs_current, veh_obj.action[-1], veh_obj.reward, veh_obj.obs_next, is_done)

    def check_determination(self, reward_accu):
        """check whether the determination meets"""
        """记录最大值次数"""
        if int(reward_accu) >= self.max_value:
            self.lr_start_decay = True  # 有一次最大值，学习率和探索率开始更新
            self.max_value_times = self.max_value_times+1
        else:
            self.max_value_times = 0
        """判断是否结束"""
        end_training = False
        if self.max_value_times == self.duration_times:
            end_training = True
        return end_training

    def save_neural_network(self, auto=False):
        if self.control_mode == "use_NN":
            """using NN, no need to store info and train NN"""
            return
        if auto:
            print("neural network auto-saved")
            torch.save(self.agent.policy_network, os.path.join(self.storage_path, "policy_net_auto.pt"))
            torch.save(self.agent.target_network, os.path.join(self.storage_path, "target_net_auto.pt"))
        else:
            print("final neural network saved")
            torch.save(self.agent.policy_network, os.path.join(self.storage_path, "policy_net.pt"))
            torch.save(self.agent.target_network, os.path.join(self.storage_path, "target_net.pt"))

    def save_log(self):
        if self.control_mode == "use_NN":
            """using NN, no need to store info and train NN"""
            return

        with open(os.path.join(self.storage_path, "logs.txt"), 'w') as f:
            for one_log in self.logs:
                f.write(str(one_log))
                f.write("\r\n")


class VehObj:
    """veh object"""
    def __init__(self, this_veh):
        self.veh_name = this_veh
        self.obs_current = 0
        self.obs_next = 0
        self.obs_forbidden_matrix = 0
        self.action = []
        self.reward = 0
        self.is_end = False
        self.last_state = 0
        self.last_state_store = False

