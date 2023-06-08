"""
Create Expert object, collect expert experience, pre-training neural network.
"""
import torch
import numpy as np
import random
from src.algorithm.Manager.StateManager import StateManager as stateManager
import os
import json
import csv
csv.field_size_limit(500*1024*1024)


class Expert:
    def __init__(self, multi_agv_scene=None, ss_x_width=3, ss_y_width=2, ss_x_num=2, ss_y_num=2, ps_num=2, explorer_num=1, filename=None):
        self.storage_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], "ExpertData")
        if filename == None:
            file_name = self.create_file_name([ss_x_width, ss_y_width, ss_x_num, ss_y_num, ps_num, explorer_num])
        else:
            file_name = filename
        self.file_name = os.path.join(self.storage_path, file_name)
        self.trajectory = []
        self.headers = ["layout", "info", "veh_name", "action"]
        self.multi_agv_scene = multi_agv_scene
        self.stateManager = stateManager()
        self.expert_size = self.count_lines()
        """采样模式"""
        self.sample_mode_list = ["points", "trajectory"]
        self.sample_mode = self.sample_mode_list[0]
        self.batch_size = 512
        self.lr = 0.0005
        """一次性读取"""
        self.all_data = []

    def create_data_by_self(self, times=50):
        self.check_csv()  # 检查文件是否存在
        csvfile = open(self.file_name, mode="a", encoding='utf-8', newline='')
        writer = csv.writer(csvfile)
        writer.writerow(self.headers)
        for i in range(times):
            print("expert is collecting data, times %d" % (i + 1))
            self.multi_agv_scene.init()  # 一条完整的轨迹为一个存储单元
            expert_info, expert_name, expert_action = self.multi_agv_scene.run_game(control_pattern="A_star")
            states = []
            layout = expert_info[0][0]
            for info in expert_info:
                state = info[1:]
                states.append(json.dumps(state))
            writer.writerow([layout, states, expert_name, expert_action])
            # self.write_data_csv([layout, states, expert_name, expert_action])
        csvfile.close()
        print("Collecting finished, line number is "+str(self.count_lines()))

    def create_data_by_rl(self):  # 边收集数据边训练
        pass

    def sample_data(self, sample_n=1):
        if not self.all_data:  # 数据未读取
            self.all_data = self.read_all_data_csv()
        order = random.randint(1, self.expert_size)
        sample_data = self.all_data[order]
        expert_s, expert_a = None, None
        if self.sample_mode == self.sample_mode_list[0]:  # 采集batch_size个点
            expert_s, expert_a = self.analyse_data(sample_data, sample_type="P")
        if self.sample_mode == self.sample_mode_list[1]:  # 采集一条完整路径
            expert_s, expert_a = self.analyse_data(sample_data, sample_type="T")
        return expert_s, expert_a

    def analyse_data(self, sample_data, sample_type="T"):
        layout = json.loads(sample_data[0])
        all_info = self.data_restore(sample_data[1], layout)
        veh_name = json.loads(sample_data[2].replace("'", '"'))
        action = json.loads(sample_data[3])
        expert_s, expert_a = [], []
        if sample_type == "T":
            expert_s, expert_a = self.form_trajectory(all_info, veh_name, action)
        if sample_type == "P":
            expert_s, expert_a = self.form_point(all_info, veh_name, action)
        return expert_s, expert_a

    def form_trajectory(self, all_info, veh_name, action):
        expert_s, expert_a = [], []
        for info_, name_ in zip(all_info, veh_name):
            state, _, _, _ = self.stateManager.create_state(info_, name_)
            expert_s.append(np.array(state))
        expert_a = action
        return expert_s, expert_a

    def form_point(self, all_info, veh_name, action):
        expert_s, expert_a = [], []
        if len(all_info) <= self.batch_size:  # 轨迹点数小于采集批量
            # print("The number of Data is to small")
            self.batch_size = len(all_info)
        expert_s, expert_a = [], []
        orders = random.sample(range(0, len(all_info)), self.batch_size)
        for i in orders:
            state, _, _, _ = self.stateManager.create_state(all_info[i], veh_name[i])
            expert_s.append(np.array(state))
            expert_a.append(action[i])
        return expert_s, expert_a

    @staticmethod
    def data_restore(data, layout):
        data = data.replace('"', "T")
        data = data.replace("'", '"')
        data = data.replace("T", "'")
        arr = json.loads(data)
        all_info = []
        for arr_ in arr:
            arr__ = arr_.replace("'", '"')
            arr___ = json.loads(arr__)
            arr___.insert(0, layout)
            all_info.append(arr___)
        return all_info

    def pre_training(self, policy_network, device, lr_=0.0005):
        optimizer = torch.optim.Adam(policy_network.parameters(), lr=lr_)  # rl太大会导致梯度消失
        expert_s, expert_a = self.sample_data(sample_n=1)
        # print("len(expert_s)", len(expert_s))

        obs = torch.from_numpy(np.stack(expert_s)).float().to(device)
        actions = torch.LongTensor(np.array(expert_a)).unsqueeze(1).to(device)
        log_probs = torch.log(policy_network(obs).gather(1, actions)).to(device)

        bc_loss = torch.mean(-log_probs)  # 最大似然估计
        # print("bc_loss", bc_loss)

        optimizer.zero_grad()
        bc_loss.backward()
        optimizer.step()

        return bc_loss

    def write_data_csv(self, state_action=None):
        with open(self.file_name, mode="a", encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if state_action is None:
                writer.writerow(self.headers)
            else:
                writer.writerow(state_action)

    def read_data_csv(self, read_line=1):
        with open(self.file_name, "r", encoding='utf-8')as csvfile:
            reader = csv.reader(csvfile)
            reader_list = list(reader)
            return reader_list[read_line]

    def read_all_data_csv(self):
        with open(self.file_name, "r", encoding='utf-8')as csvfile:
            reader = csv.reader(csvfile)
            reader_list = list(reader)
            return reader_list

    def clear_csv(self):
        with open(self.file_name, mode="w", encoding='utf-8', newline='') as csvfile:
            csvfile.truncate()

    def check_csv(self):
        if os.path.exists(self.file_name):
            print("csv file already exists, recover it")
            self.clear_csv()

    @staticmethod
    def create_file_name(paras):
        file_name = "expert_data"
        for para in paras:
            file_name = file_name + "_" + str(para)
        file_name = file_name + ".csv"
        return file_name

    def count_lines(self):
        line_num = 0
        if os.path.exists(self.file_name):
            with open(self.file_name, "r", encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    line_num += 1
            line_num = line_num-1  # 减去标题行
        return line_num
