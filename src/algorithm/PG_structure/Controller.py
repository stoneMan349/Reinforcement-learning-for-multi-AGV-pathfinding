"""
PG algorithm
Behavioral cloning can be used to improve training speed
"""
from algorithm.PG_structure.PG import Net as Net
from algorithm.PG_structure.PG import Agent as Agent
from algorithm.Manager.ExpertManager import Expert as Expert
from src.algorithm.Manager.StateManager import StateManager as stateManager
import src.algorithm.Manager.SaveManager as saveManager
import torch
import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.inf)


class PGAgentController:
    """a link between environment and algorithm"""
    def __init__(self, rmfs_scene=None, map_xdim=1, map_ydim=1, max_task=1, control_mode="train_NN", state_number=4, expert_guiding=True):
        print("start simulation with PG algorithm")
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
        self.action_length_record = 0
        """":parameter"""
        self.reward_acc = 0
        self.veh_group = []
        self.logs = []
        """"expert"""
        self.expert_guiding = expert_guiding
        self.expert = Expert(filename="expert_data_2_3_4_2_2_1.csv") if self.expert_guiding else None
        # expert_data_3_2_2_2_1_1.csv expert_data_4_2_3_3_2_1.csv expert_data_3_2_3_3_2_1.csv
        self.expert_mode_alternation = True
        self.expert_mode_pretraining = False
        self.expert_lr_start = 0.00025
        self.expert_lr_end = 0.00025
        self.expert_lr = self.expert_lr_start
        self.expert_working = True
        # self.expert_mode_list = ["alternation", "pretraining"]  # 交替训练，完全预训练
        # self.expert_mode = self.expert_mode_list[0]

    def create_agent(self, map_xdim, map_ydim):
        """create/load neural network_picture and create agent"""
        pg_net = None
        if self.control_mode == "train_NN":
            print("create NN")
            pg_net = Net(self.state_number, self.rmfs_model.action_number, map_xdim, map_ydim)
        elif self.control_mode == "use_NN":
            print("load NN")
            pg_net = torch.load(os.path.join(self.storage_path, 'pg_net.pt'))
        '''create Agent object'''
        self.agent = Agent(pg_net, self.device)

    def self_init(self):
        self.reward_acc = 0
        self.veh_group = []
        self.action_length_record = 0

    def model_run(self):  # mainloop for training/running
        start_time = datetime.datetime.now()
        log = "Training starts at" + str(start_time)
        print(log)
        self.logs.append(log)
        print("model is controlled by neural network")
        """完全预训练"""
        if self.expert_mode_pretraining:
            for i in range(10000):
                log = "pre-training:" + str(i + 1)
                print(log)
                self.logs.append(log)
                bc_loss = self.expert.pre_training(self.agent.pg_net, self.device, lr_=0.0003)
                print(bc_loss)

        bc_loss_list=[]
        for i_episode in range(self.simulation_times):
            """专家训练和强化学习混合训练"""
            # if self.expert_working:  # 能平均获得1/3的总收益，就不用专家了
                # accu_reward = self.lifelong_reward[-10:]
                # accu_reward_avg = sum(accu_reward)/max(len(accu_reward), 1)
                # if accu_reward_avg >= self.max_value*0.9 and min(accu_reward)>=self.max_value*0.75:
                #     print("no expert any more")
                #     self.expert_working = False
                # 重新设计专家退出机制

            if self.expert_mode_alternation and self.expert_working:
                for i in range(5):
                    bc_loss = self.expert.pre_training(self.agent.pg_net, self.device, lr_=self.expert_lr)
                    bc_loss_list.append(bc_loss)
            if self.expert_working:
                bc_loss_cut = bc_loss_list[-100:]
                if sum(bc_loss_cut)/max(len(bc_loss_cut), 1) < 1e-10:  # 1e-5
                    self.expert_working = True
                    # self.expert_working = False
            """init model and agent"""
            self.self_init()
            self.rmfs_model.init()
            """run model"""
            running_time = self.rmfs_model.run_game(control_pattern="intelligent", smart_controller=self, render=True)
            """info storage and print"""
            self.lifelong_reward.append(int(self.reward_acc))
            log = 'i_episode: {},\t reward_accu: {}, \t action_length: {},\t running times: {}'.format(i_episode, self.reward_acc, self.action_length_record, running_time)
            self.logs.append(log)
            print(log)
            """调整训练参数"""
            # 改变lr # -------------------------------------------------------------------还没改
            if self.lr_start_decay:  # 在update里面放了一个，但要根据训练效果调整lr，还是需要放在这里
                self.expert_lr = self.expert_lr_end

            if i_episode % 100 == 0:  # neural network auto-save
                self.save_neural_network(auto=True)
            if self.check_determination(self.reward_acc):  # check whether determination condition meets
                break
            # if i_episode % 5000 == 0:
            #     self.logs.append(self.lifelong_reward)
            #     self.save_log()  # save logs
        self.save_neural_network(auto=False)  # save network
        print("all_reward:" + str(self.lifelong_reward))
        self.logs.append(self.lifelong_reward)

        end_time = datetime.datetime.now()
        log = "Training ends at" + str(end_time)
        print(log)
        self.logs.append(log)
        log = "Duration time:"+str(end_time-start_time)
        print(log)
        self.logs.append(log)
        self.save_log()  # save logs
        saveManager.draw_picture(self.lifelong_reward, title="Cumulative Reward", x_label="training episodes", y_label="cumulative reward",
                                 color="g", save_path=os.path.join(self.storage_path, "Cumulative Reward"), smooth=True)
        saveManager.draw_picture(self.agent.loss_value, title="Loss Value", x_label="Training steps", y_label="loss value",
                                 color="k", save_path=os.path.join(self.storage_path, "Loss Value"))
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
        # action = ""
        # overestimated_flag = False
        # while True:
        #     try:
        #         action = self.agent.choose_action(obs)  # state should be formatted as array
        #     except:
        #         # the neural network may be overestimated by Pre_training for lr_ is a bit large
        #         # if this happens, do Pre_training again
        #         overestimated_flag = True
        #     if overestimated_flag:
        #         print("neural network is overestimated")
        #         self.expert.pre_training(self.agent.pg_net, self.device, lr_=0.0005)
        #     else:
        #         break

        action = self.agent.choose_action(obs)  # state should be formatted as array

        """record info"""
        veh_obj.action.append(action)
        self.action_length_record += 1
        return action

    def store_info(self, all_info, reward, is_end, this_veh):
        # print(reward)
        self.reward_acc += int(reward)
        if self.control_mode == "use_NN":
            """using NN, no need to store info and train NN"""
            return

        # veh_obj = None
        # for veh in self.veh_group:
        #     if this_veh == veh.veh_name:
        #         veh_obj = veh
        #         break
        """"store reward, and train NN (if reward = 1 or reward = -1)"""
        if reward == 1:
            is_end = True
        self.agent.store_transition(reward, is_end)

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
            torch.save(self.agent.pg_net, os.path.join(self.storage_path, "pg_net_auto.pt"))
        else:
            print("final neural network saved")
            torch.save(self.agent.pg_net, os.path.join(self.storage_path, "pg_net.pt"))

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

