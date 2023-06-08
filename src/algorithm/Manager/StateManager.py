"""
Create obs and other matrices.
Clip obs and matrices.
All matrices are created using List object
this method is used by all rl algorithm and expert manager
"""
import copy


class StateManager:
    def __init__(self):
        self.padding_size = 0

    def create_state(self, all_info, this_veh, obs_clip=False, padding_size=3):
        """Create State"""
        self.padding_size = padding_size
        """variants"""
        layout = all_info[0]  # all_info = [[layout], [veh1], [veh2]…]
        """obtain information about current_place, target_place, occupied_place, occupied_target,veh_loaded"""
        current_place, target_place, occupied_place, occupied_target, veh_loaded = \
            self.all_info_analysis(all_info, this_veh)
        """"format observations"""
        valid_path_matrix, forbidden_path_matrix = \
            self.create_path_matrix(layout, veh_loaded, current_place, target_place, occupied_place)
        other_agv_matrix = self.create_other_matrix(layout, valid_path_matrix, occupied_place, target_place)
        current_position_matrix, target_position_matrix =\
            self.create_position_matrix(layout, current_place, target_place)
        """"format clipped observations"""
        if obs_clip:
            current_position_matrix_ = self.clip_matrix(current_place, current_position_matrix)
            target_position_matrix_ = self.clip_matrix(current_place, target_position_matrix, target_place=target_place)
            valid_path_matrix_ = self.clip_matrix(current_place, valid_path_matrix)
            other_agv_matrix_ = self.clip_matrix(current_place, other_agv_matrix)
            # obs = np.array((current_position_matrix_, target_position_matrix_, valid_path_matrix_))
            obs = [current_position_matrix_, target_position_matrix_, valid_path_matrix_]
        else:
            # obs = np.array((current_position_matrix, target_position_matrix, valid_path_matrix))
            obs = [current_position_matrix, target_position_matrix, valid_path_matrix]
        """obs: neural network uses obs to make decision"""
        """current_place, target_place, valid_path_matrix: astar algorithm uses them to make decision"""
        return obs, current_place, target_place, valid_path_matrix

    def create_path_matrix(self, layout, veh_loaded, current_place, target_place, occupied_place):
        """Create valid_path_matrix, forbidden_path_matrix"""
        # valid_path_matrix, forbidden_path_matrix
        valid_path, forbidden_path, = self.layout_to_matrix(layout, veh_loaded)
        valid_path_matrix = copy.deepcopy(valid_path)
        forbidden_path_matrix = copy.deepcopy(forbidden_path)
        """调整valid_path_matrix和forbidden_path_matrix"""
        # 根据current_position和target_position调整
        valid_path_matrix[current_place[1] - 1][current_place[0] - 1] = 1.0  # current_place_array样式[[x],[y]]
        forbidden_path_matrix[current_place[1] - 1][current_place[0] - 1] = 0.0
        valid_path_matrix[target_place[1] - 1][target_place[0] - 1] = 1.0
        forbidden_path_matrix[target_place[1] - 1][target_place[0] - 1] = 0.0
        # 其他车辆对道路的占用
        if occupied_place:  # 非单辆车
            for o_place in occupied_place:
                valid_path_matrix[o_place[1] - 1][o_place[0] - 1] = 0.0
                forbidden_path_matrix[o_place[1] - 1][o_place[0] - 1] = 1.0

        return valid_path_matrix, forbidden_path_matrix

    def create_other_matrix(self, layout, valid_path_matrix, occupied_place, target_place):
        """"标记其他AGV位置和四格周边位置"""
        basic_matrix = self.create_basic_matrix(layout)
        other_agv_matrix = copy.deepcopy(basic_matrix)
        up, right, down, left = (0, -1), (1, 0), (0, 1), (-1, 0)
        four_dict = [up, right, down, left]
        if occupied_place:  # 非单辆车
            for o_place in occupied_place:
                o_place_x, o_place_y = o_place[0] - 1, o_place[1] - 1
                other_agv_matrix[o_place_y][o_place_x] = 1.0  # current_place_array样式[[x],[y]]
                for one_direction in four_dict:
                    pos = [o_place_x+one_direction[0], o_place_y+one_direction[1]]
                    if pos[0] < 0 or pos[1] < 0 or pos[0] >= len(valid_path_matrix[0]) or pos[1] >= len(valid_path_matrix):
                        continue
                    else:
                        other_agv_matrix[pos[1]][pos[0]] = 1.0  # current_place_array样式[[x],[y]]
        other_agv_matrix[target_place[1] - 1][target_place[0] - 1] = 0.0  # 当前AGV的目标位置，标记为0
        return other_agv_matrix

    def create_position_matrix(self, layout, current_place, target_place):
        basic_matrix = self.create_basic_matrix(layout)
        current_position_matrix = copy.deepcopy(basic_matrix)
        target_position_matrix = copy.deepcopy(basic_matrix)
        # 构建current_position_matrix
        current_position_matrix[current_place[1] - 1][current_place[0] - 1] = 1.0
        # 构建target_position_matrix
        target_position_matrix[target_place[1]-1][target_place[0]-1] = 1.0
        return current_position_matrix, target_position_matrix

    @staticmethod
    def layout_to_matrix(layout, veh_loaded):
        """制作原始的valid_path和forbidden_path"""
        valid_path, valid_path_one_line, forbidden_path, forbidden_path_one_line = [], [], [], []
        for map_one_line in layout:
            for one_cell in map_one_line:
                # 0:道路；1:存储站;2:拣选站;3:障碍物
                if one_cell == 0:
                    valid_path_one_line.append(1.)
                    forbidden_path_one_line.append(0.)
                elif one_cell == 1:
                    if veh_loaded == 0:
                        valid_path_one_line.append(1.)
                        forbidden_path_one_line.append(0.)
                    else:
                        valid_path_one_line.append(0.)
                        forbidden_path_one_line.append(1.)
                elif one_cell == 2:
                    valid_path_one_line.append(0.)
                    forbidden_path_one_line.append(1.)
                elif one_cell == 3:
                    valid_path_one_line.append(0.)
                    forbidden_path_one_line.append(1.)
                else:
                    print("create_path_matrix:wrong matrix")
            valid_path.append(valid_path_one_line)
            forbidden_path.append(forbidden_path_one_line)
            valid_path_one_line, forbidden_path_one_line = [], []
        return valid_path, forbidden_path

    @staticmethod
    def all_info_analysis(all_info, this_veh):
        current_place, target_place, occupied_place, occupied_target, veh_loaded = 0, 0, [], [], False
        for i in range(1, len(all_info)):
            one_veh = all_info[i]
            veh_name_, current_place_, target_place_, veh_loaded_ = one_veh[0], one_veh[1], one_veh[2], one_veh[3]
            if veh_name_ == this_veh:  # target_veh
                current_place, target_place, veh_loaded = current_place_, target_place_, veh_loaded_
            else:
                occupied_place.append(current_place_)
                occupied_target.append(target_place_)
        return current_place, target_place, occupied_place, occupied_target, veh_loaded

    @staticmethod
    def create_basic_matrix(layout):
        basic_matrix, basic_matrix_one_line = [], []
        for map_one_line in layout:
            for one_cell in map_one_line:
                basic_matrix_one_line.append(0.)
            basic_matrix.append(basic_matrix_one_line)
            basic_matrix_one_line = []
        return basic_matrix

    def clip_matrix(self, current_place, matrix, padding_stuff=0, target_place=None):
        """先按照padding_size做填充"""
        new_matrix = []
        line_padding = [padding_stuff]*self.padding_size
        row_padding = [padding_stuff]*(self.padding_size*2+len(matrix[0]))
        for content in matrix:
            line = line_padding+content+line_padding
            new_matrix.append(line)
        for i in range(self.padding_size):
            new_matrix.insert(0, row_padding)
            new_matrix.append(row_padding)
        """锁定位置裁剪"""
        x = (current_place[0] - 1) + self.padding_size
        y = (current_place[1] - 1) + self.padding_size
        x_start, x_end = x-self.padding_size, x+self.padding_size
        y_start, y_end = y-self.padding_size, y+self.padding_size
        """target place需要额外的处理——target place等价转化"""
        if target_place is not None:
            x_t = (target_place[0] - 1) + self.padding_size
            y_t = (target_place[1] - 1) + self.padding_size
            if x_start <= x_t <= x_end and y_start <= y_t <= y_end:  # 目标点在裁剪区间
                pass
            else:
                if x_start <= x_t <= x_end:  # x轴落在区间内
                    if y_t < y_start:
                        new_matrix[y_start][x_t] = 1
                    if y_t > y_end:
                        new_matrix[y_end][x_t] = 1
                elif y_start <= y_t <= y_end:  # y轴落在区间内
                    if x_t < x_start:
                        new_matrix[y_t][x_start] = 1
                    if x_t > x_end:
                        new_matrix[y_t][x_end] = 1
                else:  # 不落在任何区间内
                    if x_t < x_start and y_t < y_start:
                        new_matrix[y_start][x_start] = 1
                    if x_t < x_start and y_t > y_end:
                        new_matrix[y_end][x_start] = 1
                    if x_t > x_end and y_t < y_start:
                        new_matrix[y_start][x_end] = 1
                    if x_t > x_end and y_t > y_end:
                        new_matrix[y_end][x_end] = 1
        """裁剪"""
        final_matrix = []
        for i in range(y_start, y_end+1):
            line = new_matrix[i][x_start:x_end + 1]
            final_matrix.append(line)
        return final_matrix





