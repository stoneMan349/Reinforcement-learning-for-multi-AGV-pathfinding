"""
AC algorithm
Behavioral cloning can be used to improve training speed
"""
from algorithm.AC_structure.AC import Actor as Actor
from algorithm.AC_structure.AC import Critic as Critic
from algorithm.AC_structure.AC import Agent as Agent
from algorithm.Manager.ExpertManager import Expert as Expert
from src.algorithm.Manager.StateManager import StateManager as stateManager
import src.algorithm.Manager.SaveManager as saveManager
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.inf)


class ACAgentController:
    """
        a link between environment and algorithm
        """
    def __init__(self, rmfs_scene=None, map_xdim=1, map_ydim=1, max_task=1, control_mode="train_NN", state_number=4, expert_guiding=True):
        """a link between environment and algorithm"""
        print("start simulation with AC algorithm")
        print("map_xdim:", map_xdim, "map_ydim:", map_ydim, "state_number:", state_number)
        '''received parameters'''
        self.control_mode = control_mode
        self.state_number = state_number
        self.rmfs_model = rmfs_scene  # get RMFS object
        '''path and state management'''
        self.stateManager = stateManager()
        self.storage_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], "network_picture")
        """create device"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # self.device = torch.device("cpu")
        """create_agent"""
        self.agent = None
        self.create_agent(map_xdim, map_ydim)
        '''training parameters'''
        self.simulation_times = 10000
        self.max_value = max_task*3
        self.max_value_times = 0
        self.duration_times = 100
        """"hyper parameter"""
        self.lr_start_decay = False
        self.lifelong_reward = []
        self.lifelong_action = []
        self.action_length_record = 0
        """":parameter"""
        self.reward_acc = 0
        self.veh_group = []
        self.logs = []
        """"expert"""
        self.expert_guiding = expert_guiding
        self.expert = Expert(filename="expert_data_3_2_3_3_2_1.csv") if self.expert_guiding else None
        self.expert_mode_alternation = True
        self.expert_mode_pretraining = False
        self.expert_lr_start = 0.0005
        self.expert_lr_end = 0.0003
        self.expert_lr = self.expert_lr_start
        self.expert_working = True

    def create_agent(self, map_xdim, map_ydim):
        """create/load neural network_picture and create agent"""
        actor_net, critic_net = None, None
        if self.control_mode == "train_NN":
            print("create NN")
            actor_net = Actor(self.state_number, self.rmfs_model.action_number, map_xdim, map_ydim)
            critic_net = Critic(self.state_number, self.rmfs_model.action_number, map_xdim, map_ydim)
        elif self.control_mode == "use_NN":
            print("load NN")
            actor_net = torch.load(os.path.join(self.storage_path, 'actor_net.pt'))
            critic_net = torch.load(os.path.join(self.storage_path, 'critic_net.pt'))
        self.agent = Agent(actor_net, critic_net, self.device)

    def self_init(self):
        self.reward_acc = 0
        self.veh_group = []
        self.action_length_record = 0

    def model_run(self):  # mainloop for training/running
        print("model is controlled by neural network")
        """完全预训练"""
        if self.expert_mode_pretraining:
            for i in range(1000):
                log = "pre-training:" + str(i + 1)
                print(log)
                self.logs.append(log)
                self.expert.pre_training(self.agent.actor_net, self.device, lr_=self.expert_lr)
        bc_loss_list = []
        for i_episode in range(self.simulation_times):
            """专家训练和强化学习混合训练"""
            # if self.expert_working:  # 能平均获得1/3的总收益，就不用专家了
            #     accu_reward = self.lifelong_reward[-10:]
            #     accu_reward_avg = sum(accu_reward)/max(len(accu_reward), 1)
            #     if accu_reward_avg >= self.max_value*0.8:
            #         print("no expert any more")
            #         self.expert_working = False
            if self.expert_mode_alternation and self.expert_working:
                for i in range(4):
                    bc_loss = self.expert.pre_training(self.agent.actor_net, self.device, lr_=self.expert_lr)
                    bc_loss_list.append(bc_loss)
            if self.expert_working:
                bc_loss_cut = bc_loss_list[-10:]
                if sum(bc_loss_cut)/max(len(bc_loss_cut), 1) < 1e-6:  # 1e-5
                    print("no expert any more")
                    self.expert_working = False
            """init model and agent"""
            self.self_init()
            self.rmfs_model.init()
            """run model"""
            running_time = self.rmfs_model.run_game(control_pattern="intelligent", smart_controller=self)
            """info storage and print"""
            self.lifelong_reward.append(int(self.reward_acc))
            self.lifelong_action.append(self.action_length_record)
            log = 'i_episode: {},\t reward_accu: {}, \t action_length: {},\t running times: {}'.format(i_episode, self.reward_acc, self.action_length_record, running_time)
            self.logs.append(log)
            print(log)
            """调整训练参数"""
            if self.lr_start_decay:  # 改变lr
                self.expert_lr = self.expert_lr_end

            if i_episode % 100 == 0:
                self.save_neural_network(auto=True)
            if self.check_determination(self.reward_acc):  # check whether determination condition meets
                break
        self.save_neural_network(auto=False)
        print("all_reward:" + str(self.lifelong_reward))
        self.logs.append(self.lifelong_reward)
        saveManager.save_logs(os.path.join(self.storage_path, "logs"), self.logs)
        saveManager.draw_picture(self.lifelong_reward, title="Cumulative Reward", x_label="training episodes", y_label="cumulative reward",
                                 color="g", save_path=os.path.join(self.storage_path, "Cumulative Reward"), smooth=True)
        saveManager.draw_picture(self.agent.loss_value, title="Loss Value", x_label="Training steps", y_label="loss value",
                                 color="k", save_path=os.path.join(self.storage_path, "Loss Value"))
        # self.draw_picture(self.lifelong_action, p_title="Action Length", p_xlabel="Training steps", p_ylabel="Loss value",p_color="r")
        plt.show()

    def choose_action(self, all_info, this_veh):  # all_infor=[layout  , current_place, target_place
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
        obs, this_veh_cp, this_veh_tp, valid_path_matrix = self.stateManager.create_state(all_info, this_veh, obs_clip=False)
        obs = np.array(obs)
        """get action"""
        # veh_obj.obs_current = obs
        # veh_obj.obs_valid_matrix = valid_path_matrix
        # action = ""
        # overestimated_flag = False
        # while True:
        #     try:
        action = self.agent.choose_action(obs)  # state should be formatted as array
            # except:
            #     # the neural network may be overestimated by Pre_training for lr_ is a bit large
            #     # if this happens, do Pre_training again
            #     overestimated_flag = True
            # if overestimated_flag:
            #     print("neural network is overestimated")
            #     self.expert.pre_training(self.agent.actor_net, self.device, lr_=self.expert_lr)
            # else:
            #     break

        """record info"""

        veh_obj.action.append(action)
        self.action_length_record += 1
        return action

    def store_info(self, all_info, reward, is_end, this_veh):
        print(reward)
        self.reward_acc += reward
        if self.control_mode == "use_NN":
            """using NN, no need to store info and train NN"""
            return
        veh_obj = None
        for veh in self.veh_group:
            if this_veh == veh.veh_name:
                veh_obj = veh
                break
        obs, this_veh_cp, this_veh_tp, valid_path_matrix = self.stateManager.create_state(all_info, this_veh, obs_clip=False)
        obs = np.array(obs)
        # 先存储经验，在保留新数据
        veh_obj.obs_next, veh_obj.reward = obs, reward
        self.agent.store_transition(veh_obj.reward, veh_obj.obs_next)

    def check_determination(self, reward_accu):
        """check whether the determination meets"""
        """记录最大值次数"""
        if int(reward_accu) >= self.max_value:
            self.lr_start_decay = True  # 有一次最大值，学习率和探索率开始更新
            self.max_value_times = self.max_value_times + 1
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
            print("neural network auto-save")
            torch.save(self.agent.actor_net, os.path.join(self.storage_path, "actor_net_auto.pt"))
            torch.save(self.agent.critic_net, os.path.join(self.storage_path, "critic_net_auto.pt"))
        else:
            torch.save(self.agent.actor_net, os.path.join(self.storage_path, "actor_net.pt"))
            torch.save(self.agent.critic_net, os.path.join(self.storage_path, "critic_net.pt"))


class VehObj:
    """veh object"""
    def __init__(self, this_veh):
        self.veh_name = this_veh
        """"逐个检查"""
        # self.obs_list = []
        self.obs_current = 0
        self.obs_next = 0
        self.obs_forbidden_matrix = 0
        self.obs_valid_matrix = 0
        self.action = []
        self.reward = 0
        self.is_end = False
        self.last_state = 0
        self.last_state_store = False

