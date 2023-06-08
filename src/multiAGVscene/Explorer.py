import copy
import sys
import utils.astar as astar
from utils.utils import Direction as Dir
from utils.utils import Working as Wokring
from utils.settings import SuperParas as SuperPara
import os

sys.path.append(os.path.dirname(__file__))


class Explorer:
    def __init__(self, layout, veh_name="veh1", icon_name="veh1"):
        self.layout = layout
        """--------------init attributes--------------"""
        """work time """
        # 从储位取货架,在拣货台放货架,在储位放货架 {"GET": 5, "PUT": 5, "RET": 5}
        # self.Working_Time = {0: 5, 1: 5, 2: 5}
        self.Working = False
        self.working_type = ""
        self.time_counting = 0
        """basic information"""
        self.dir = Dir()
        self.working_manager = Wokring()
        self.action_str = self.dir.value_str[1]
        self.action_list = []
        self.current_place = [1, 1]
        self.target_position = [1, 1]
        self.last_place = [1, 1]
        self.task_order = 0
        self.task_stage = 0
        self.running_state = "Start"
        self.loaded = False
        self.explorer_name = veh_name
        self.icon_path = self.get_icon()
        self.has_created = False
        self.all_assigned = False
        """"special parameters"""
        self.always_loaded = SuperPara.Explorer_Always_Loaded
        self.always_empty = SuperPara.Explorer_Always_Empty
        self.action_distribution = [0, 0, 0, 0, 0]  # up, right, down, left, stop
        """"init"""
        self.init()

    def init(self):
        self.last_place = [1, 1]
        self.current_place = [1, 1]
        self.task_order = 0
        self.task_stage = 0
        self.running_state = "Start"
        self.loaded = False
        self.has_created = False
        self.all_assigned = False
        self.action_list = []

        self.Working = False
        self.working_type = 0
        self.time_counting = 0

    def get_icon(self):
        icon_path = "multiAGVscene/icons/" + str(self.explorer_name) + "_" + str(self.task_stage + 1) + ".png"
        return icon_path

    def create_explorer(self):
        self.has_created = True
        self.get_task()

    def get_task(self):
        """"has veh created"""
        if not self.has_created:
            # veh entity has not been build
            return

        """judge if has unassigned tasks"""
        if len(self.layout.task_list) == len(self.layout.task_arrangement[0]) and self.task_stage == 0:
            self.all_assigned = True  # 所有任务已经分配
            # 判断任务是否全部完成
            if len(self.layout.task_list) == sum(self.layout.task_arrangement[2]):
                self.layout.task_finished = True
                self.running_state = "AllTaskFinished"
            return
        """"assign task"""
        if self.task_stage == 0:
            # 获取任务编号，将以分配的任务进行添加
            self.task_order = len(self.layout.task_arrangement[0])
            self.layout.task_arrangement[0].append(self.task_order)
            self.layout.task_arrangement[1].append(self.explorer_name)
            self.layout.task_arrangement[2].append(0)

            self.target_position = [self.layout.task_list[self.task_order][0],
                                    self.layout.task_list[self.task_order][1]]
            self.layout.change_layout(self.layout.task_list[self.task_order][0] - 1,
                                      self.layout.task_list[self.task_order][1] - 1, 1.3)
        elif self.task_stage == 1:
            self.target_position = [self.layout.task_list[self.task_order][2],
                                    self.layout.task_list[self.task_order][3]]
        elif self.task_stage == 2:
            self.target_position = [self.layout.task_list[self.task_order][0],
                                    self.layout.task_list[self.task_order][1]]
        self.load_condition(self.task_stage)

    def load_condition(self, task_stage):
        # change by parameter
        if self.always_loaded:
            self.loaded = True
            return
        if self.always_empty:
            self.loaded = False
            return
        # change by task_stage
        if task_stage == 0:
            self.loaded = False
        elif task_stage == 1:
            self.loaded = True
        elif task_stage == 2:
            self.loaded = True

    def action_format(self, input_action):
        """transfer action format to int"""
        if isinstance(input_action, str):
            action_value = self.dir.action_str_value(input_action)
        else:
            action_value = input_action
        action_str = self.dir.action_value_str(action_value)
        return action_value, action_str

    def action_logical(self, action_value):
        action_value_dict = {0: [0, -1], 1: [1, 0], 2: [0, 1], 3: [-1, 0], 4: [0, 0]}
        return action_value_dict[action_value]

    def execute_action(self, input_action, all_info=[], explorer_group=None):
        """"basic info and manipulation"""
        self.action_distribution = [0, 0, 0, 0, 0]
        self.action_distribution[self.dir.action_str_value(input_action)] = 1
        action_value, self.action_str = self.action_format(input_action)
        action_result = self.action_logical(action_value)
        """check action, and get reward, is_end"""
        current_place = [0, 0]  # (x,y)
        current_place[0] = self.current_place[0] + action_result[0]
        current_place[1] = self.current_place[1] + action_result[1]
        reward, is_end = self.check_action(all_info, current_place,
                                           explorer_group)  # all_info is a list including all vehs' info
        """execute action"""
        self.action_list.append(input_action)
        if is_end:
            pass
        else:
            self.current_place = copy.deepcopy(current_place)
        return reward, is_end

    def check_action(self, all_info, current_place, explorer_group):
        """Legitimacy test"""
        reward, is_end = 0, False
        self.running_state = "Normal"
        """out of boundary"""
        if current_place[0] < 1 or current_place[1] < 1 or \
                current_place[0] > self.layout.scene_x_width or current_place[1] > self.layout.scene_y_width:
            self.running_state = "illegal action"
            reward, is_end = -1, True
            print("out of boundary")
            return reward, is_end
        """"hit storage position or picking station """
        # hit storage position
        if (current_place[0], current_place[1]) in self.layout.storage_station_list and \
                self.loaded is True and current_place != self.target_position:
            self.running_state = "hit s_station"
            reward, is_end = -1, True
            print("hit s_station")
            return reward, is_end
        # hit picking position
        if (current_place[0], current_place[1]) in self.layout.picking_station_list and \
                current_place != self.target_position:  # no need to check load condition
            self.running_state = "hit p_station"
            reward, is_end = -1, True
            print("hit p_station")
            return reward, is_end
        """"hit other veh """
        for i in range(1, len(all_info)):  # the first position of info is layout
            one_veh = all_info[i]
            veh_name_, current_place_ = one_veh[0], one_veh[1]
            if veh_name_ == self.explorer_name:  # target_veh
                continue
            else:
                if current_place == current_place_:
                    reward, is_end = -1, True
                    print("hit other veh")
                    return reward, is_end
        """reach target place"""
        if current_place[0] == self.target_position[0] and current_place[1] == self.target_position[1]:
            self.Working = True
            self.time_counting = 0
            if (current_place[0], current_place[1]) in self.layout.storage_station_list:
                if self.task_stage == 0:
                    self.running_state = "Lifting"
                    # self.working_type = self.working_manager.work_type_val_str[3]
                else:
                    self.running_state = "Downing"
                    # self.working_type = self.working_manager.work_type_val_str[4]
            if (current_place[0], current_place[1]) in self.layout.picking_station_list:
                self.running_state = "Picking"
                # self.working_type = self.working_manager.work_type_val_str[5]
            reward, is_end = 1, False
            self.continue_working()
            return reward, is_end
        """normal action"""
        if SuperPara.Sparse_Reward:
            reward = self.rectify_reward(explorer_group, current_place)
        return reward, is_end

    def rectify_reward(self, explorer_group, current_place):
        path_founder = astar.FindPathAstar(self.create_valid_matrix(explorer_group),
                                           (self.current_place[0] - 1, self.current_place[1] - 1),
                                           (self.target_position[0] - 1, self.target_position[1] - 1))
        find_target, path_list, path_map, action_list = path_founder.run_astar_method()
        path_length_last = len(action_list)
        path_founder = astar.FindPathAstar(self.create_valid_matrix(explorer_group),
                                           (current_place[0] - 1, current_place[1] - 1),
                                           (self.target_position[0] - 1, self.target_position[1] - 1))
        find_target, path_list, path_map, action_list = path_founder.run_astar_method()
        path_length_new = len(action_list)
        # print(self.current_place)
        # print(current_place)
        # print(self.target_position)

        reward = 0
        """根据位移前后的相对位置判断奖励值"""
        if path_length_last == path_length_new:
            reward = 0
        elif path_length_last > path_length_new:
            reward = 0.001
        elif path_length_last < path_length_new:
            reward = -0.002
        """根据距离目标位置的A*距离判断奖励值——分段距离"""
        # if path_length_new <= 5:
        #     reward = 0.005
        # # elif path_length_last > 5 and path_length_new <= 10:
        # #     reward = 0.002
        # elif path_length_last > 5 and path_length_new <= 10:
        #     reward = 0.001
        # elif path_length_last > 15:
        #     reward = 0
        """根据距离目标位置的A*距离判断奖励值——曲线距离"""
        max_length = len(self.layout.layout_original)+len(self.layout.layout_original[0])
        max_length = 0.5*max_length  # max_length会影响奖励值的差异
        reward1 = 0.001*(max_length-path_length_new)/max_length

        # reward = reward + reward1
        reward = reward1
        return reward

    def continue_working(self):
        self.task_stage = self.task_stage + 1
        if self.task_stage == 3:
            self.layout.change_layout(self.layout.task_list[self.task_order][0] - 1,
                                      self.layout.task_list[self.task_order][1] - 1, 1.8)
            self.task_stage = 0
            self.layout.task_arrangement[2][self.task_order] = 1
        self.icon_path = self.get_icon()
        self.get_task()

    def create_valid_matrix(self, explorer_group):
        valid_matrix, valid_matrix_one, value = [], [], 0
        for j in range(len(self.layout.layout_original)):
            for i in range(len(self.layout.layout_original[0])):
                cell_value = self.layout.layout_original[j][i]
                if cell_value == 0:
                    value = 1
                elif cell_value == 1:
                    value = 0 if self.loaded else 1  # 目标储位依然允许到达
                elif cell_value == 2:
                    value = 0
                if self.target_position[0] - 1 == i and self.target_position[1] - 1 == j:
                    value = 1
                valid_matrix_one.append(value)
            valid_matrix.append(valid_matrix_one)
            valid_matrix_one = []
        # adjust according to the position of other AGV
        if explorer_group is not None:
            if len(explorer_group) > 1:
                for explorer in explorer_group:
                    valid_matrix[explorer.current_place[1] - 1][explorer.current_place[0] - 1] = 0
            # valid_matrix[self.current_place[1]-1][self.current_place[0]-1] = 0
        # print("valid_matrix", valid_matrix)
        return valid_matrix

    def find_path_astar(self, explorer_group=None):
        path_founder = astar.FindPathAstar(self.create_valid_matrix(explorer_group),
                                           (self.current_place[0] - 1, self.current_place[1] - 1),
                                           (self.target_position[0] - 1, self.target_position[1] - 1))
        find_target, path_list, path_map, action_list = path_founder.run_astar_method()
        # print("self.current_place", self.current_place)
        # print("self.target_position", self.target_position)
        # print("find_target", find_target)
        # print("path_list", path_list)
        # print("path_map", path_map)
        # print("action_list", action_list)
        if find_target == False or len(action_list) == 0:
            action_str = "STOP"
        else:
            action_str = action_list[0]
        # print(action_list)
        return action_str
