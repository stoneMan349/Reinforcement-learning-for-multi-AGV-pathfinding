import copy

import numpy as np


class Cell:
    def __init__(self, position=(0, 0)):
        self.position = position
        self.parent = None

        self.g = 0
        self.h = 0
        self.f = 0
        self.all_cost = 0


class Gridworld:
    def __init__(self, world_map):
        """
        :param world_map:[[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        """
        self.wm = np.array(world_map)
        self.world_x_limit = len(world_map[0])
        self.world_y_limit = len(world_map)

    def show(self):
        print(self.wm)

    def get_neigbours(self, cell):
        """
        Return the neighbours of cell
        """
        neighbour_cord = [
            (-1, 0),
            (0, -1),
            (0, 1),
            (1, 0),
        ]
        current_x = cell.position[0]
        current_y = cell.position[1]
        neighbours = []
        for n in neighbour_cord:
            x = current_x + n[0]
            y = current_y + n[1]
            if 0 <= x < self.world_x_limit and 0 <= y < self.world_y_limit and self.wm[y][x] == 1.:
                c = Cell()
                c.position = (x, y)
                c.parent = cell
                neighbours.append(c)
        return neighbours


class FindPathAstar:
    def __init__(self, world_map, start_pos, target_pos):
        # print("world_map", world_map)
        # print("start_pos", start_pos)
        # print("target_pos", target_pos)
        self.wm = Gridworld(world_map)
        self.start_cell = Cell(start_pos)
        self.target_cell = Cell(target_pos)
        self.find_target = False
        self.path_list = []
        self.path_map = 0
        self.action_list = []

    def run_astar_method(self):
        self.astar_method()
        if self.find_target:
            self.astar_plot_map()
            self.astar_plot_action_route()
        return self.find_target, self.path_list, self.path_map, self.action_list

    def astar_method(self):
        _open = []
        _closed = []
        _open.append(self.start_cell)
        self.find_target = False
        # print("_open", _open)
        # print("len(_open)", len(_open))

        while _open:
            # print("_open", _open)
            # print("_________________")
            # for n in _open:
            #     print(n.position)
            min_f = np.argmin([n.all_cost for n in _open])  # 最小值的下标
            current_cell = _open[min_f]
            _closed.append(_open.pop(min_f))  # 移除列表中的一个元素（默认最后一个元素），并且返回该元素的值

            if current_cell.position == self.target_cell.position:
                self.find_target = True
                break
            for n in self.wm.get_neigbours(current_cell):
                is_in_close = False
                for c in _closed:
                    if c.position == n.position:
                        is_in_close = True
                        break
                if is_in_close:
                    # print("__________________________________________________________________________")
                    continue
                n.g = current_cell.g + 1
                x1, y1 = n.position
                x2, y2 = self.target_cell.position
                n.h = (y2 - y1) ** 2 + (x2 - x1) ** 2
                n.f = n.h + n.g

                already_in = False
                for c in _open:
                    if c.position == n.position:
                        if c.f < n.f:
                            pass
                        else:
                            _open.remove(c)
                            _open.append(n)
                        already_in = True
                        break
                if already_in:
                    pass
                else:
                    _open.append(n)

        if self.find_target:
            path = []
            while current_cell.parent is not None:
                path.append(current_cell.position)
                current_cell = current_cell.parent
            path.append(current_cell.position)
            self.path_list = copy.deepcopy(path)
        # return path[::-1]

    def astar_plot_map(self):
        self.path_map = self.wm.wm
        for n in self.path_list:
            x = n[0]
            y = n[1]
            self.path_map[y][x] = -1

    def astar_plot_action_route(self):
        self.action_list = []
        action_str = ""
        for i in range(len(self.path_list)-1, 0, -1):
            current_position = self.path_list[i]
            next_position = self.path_list[i-1]
            if current_position[0] < next_position[0]:
                action_str = "RIGHT"
            elif current_position[0] > next_position[0]:
                action_str = "LEFT"
            if current_position[1] < next_position[1]:
                action_str = "DOWN"
            elif current_position[1] > next_position[1]:
                action_str = "UP"
            self.action_list.append(action_str)


if __name__ == "__main__":
    # valid_path = [[1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
    #               [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
    #               [0.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
    #               [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
    #               [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
    #               [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
    #               [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
    #               [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
    #               [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
    #               [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
    #               [1.0000, 0.0000, 1.0000, 1.0000, 0.0000, 1.0000, 1.0000],
    #               [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]]
    valid_path = [[1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
                  [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
                  [0.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
                  [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
                  [1.0000, 1.0000, 1.0000, 1.0000, 0.0000, 0.0000, 0.0000],
                  [1.0000, 1.0000, 1.0000, 1.0000, 0.0000, 1.0000, 1.0000],
                  [1.0000, 1.0000, 1.0000, 1.0000, 0.0000, 0.0000, 1.0000],
                  [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
                  [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
                  [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
                  [1.0000, 0.0000, 1.0000, 1.0000, 0.0000, 1.0000, 1.0000],
                  [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]]
    # start_position = (1, 4)  # (x,y)先横坐标，后纵坐标
    # target_position = (6, 6)
    start_position = (0, 3)  # (x,y)先横坐标，后纵坐标
    target_position = (5, 5)
    founder = FindPathAstar(valid_path, start_position, target_position)
    find_target, path_list, path_map, action_list = founder.run_astar_method()
    print("find_target", find_target)
    print("path_list", path_list)
    print("path_map", path_map)
    print("action_list", action_list)

    # world = Gridworld()
    # #   stat position and Goal
    # start = Cell()
    # start.position = (0, 0)
    # goal = Cell()
    # goal.position = (4, 4)
    # print(f"path from {start.position} to {goal.position}")
    # s = astar(world, start, goal)
    # #   Just for visual reasons
    # for i in s:
    #     world.w[i] = 1
    # print(world.w)
