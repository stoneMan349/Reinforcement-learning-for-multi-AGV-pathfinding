import copy
import random
import sys
from collections import namedtuple
import os
sys.path.append(os.path.dirname(__file__))


class Layout:
    """"This class is used to create the layout of the multiAGV scene
    1.create layout
    2.create task
    3.manage layout information
    """
    def __init__(self, storage_station_x_width=3, storage_station_y_width=2, storage_station_x_num=2,
                 storage_station_y_num=2, picking_station_number=2, layout_list=None, task_list=None):
        """--------------create layout--------------"""
        if layout_list is None:
            """storage station location"""
            self.storage_station_x_width = storage_station_x_width  # island size
            self.storage_station_y_width = storage_station_y_width  # island size
            self.storage_station_interval = 1  # island interval
            self.storage_station_bottom = 3
            self.storage_station_x_num = storage_station_x_num  # island number
            self.storage_station_y_num = storage_station_y_num  # island number
            self.storage_station_list = self.__create_storage_station()  # 从1开始计数

            """size of scene"""
            self.scene_x_width = self.storage_station_x_num*(self.storage_station_x_width+self.storage_station_interval)+self.storage_station_interval
            self.scene_y_width = self.storage_station_y_num*(self.storage_station_y_width+self.storage_station_interval)+self.storage_station_interval+self.storage_station_bottom

            """picking station location"""
            self.picking_station_number = picking_station_number
            self.picking_station_list = self.__create_picking_station()
        else:
            """create storage station, picking station location by list"""
            self.storage_station_list, self.picking_station_list = self.__create_ss_ps_by_list(layout_list)
            """size of scene"""
            self.scene_x_width = len(layout_list[0])
            self.scene_y_width = len(layout_list)

        """--------------create task--------------"""
        self.task_list = None
        self.task_is_fixed = True
        if task_list is None:
            self.task_is_fixed = False
        else:
            self.task_list = task_list
            self.task_is_fixed = True
        self.task_arrangement = [[], [], []]  # record task arrangement: [[order1, order2],[veh1, vhe2], [0, 1]]
        self.task_finished = False
        self.layout, self.layout_original = [], []

        self.init()

    def init(self):
        """create task"""
        if not self.task_is_fixed:
            self.task_list = self.__create_task()
        self.task_finished = False
        self.task_arrangement = [[], [], []]  # [order], [veh], [stage]
        """"layout in matrix"""
        self.layout = self.__create_layout()
        self.layout_original = copy.deepcopy(self.layout)

    def __create_ss_ps_by_list(self, layout_list):
        picking_station_class = namedtuple('ps', 'x_position y_position')
        picking_station_list = []

        storage_station_class = namedtuple('ss', 'x_position y_position')
        storage_station_list = []
        for y in range(len(layout_list)):
            for x in range(len(layout_list[0])):
                if layout_list[y][x] == 0:
                    pass
                elif layout_list[y][x] == 1:  # storage station
                    storage_station_list.append(storage_station_class(x+1, y+1))
                elif layout_list[y][x] == 2:  # picking station
                    picking_station_list.append(picking_station_class(x+1, y+1))

        return storage_station_list, picking_station_list

    def __create_picking_station(self):
        picking_station_class = namedtuple('ps', 'x_position y_position')
        picking_station_list = []
        for i in range(self.picking_station_number):
            x_position = int(self.scene_x_width*(i+1)/(self.picking_station_number+1)+1)
            y_position = self.scene_y_width-1
            picking_station_list.append(picking_station_class(x_position, y_position))
        return picking_station_list

    def __create_storage_station(self):
        storage_station_class = namedtuple('ss', 'x_position y_position')
        storage_station_list = []
        for y_num in range(self.storage_station_y_num):
            for y_width in range(self.storage_station_y_width):
                y_position = y_num*(self.storage_station_y_width+self.storage_station_interval)+y_width+1+1
                for x_num in range(self.storage_station_x_num):
                    for x_width in range(self.storage_station_x_width):
                        x_position = x_num*(self.storage_station_x_width+self.storage_station_interval)+x_width+1+1
                        storage_station_list.append(storage_station_class(x_position, y_position))
        return storage_station_list

    def __create_task(self):
        task_list = []
        ss_list = copy.deepcopy(self.storage_station_list)
        ps_list = copy.deepcopy(self.picking_station_list)
        while True:
            i = random.randint(0, len(ss_list)-1)
            j = random.randint(0, len(ps_list)-1)
            start_location = ss_list.pop(i)
            target_location = ps_list[j]
            task_list.append(start_location+target_location)
            if len(ss_list) == 0:
                break
        return task_list

    def __create_layout(self):
        # 0:track, 1:occupied ss, 2:empty ss. 3:ps
        line = []
        layout = []
        for j in range(self.scene_y_width):
            for i in range(self.scene_x_width):
                block_value = 0
                if (i+1, j+1) in self.storage_station_list:
                    block_value = 1
                if (i+1, j+1) in self.picking_station_list:
                    block_value = 2
                line.append(block_value)
            layout.append(line)
            line = []
        return layout

    def change_layout(self, x_dim, y_dim, value):
        self.layout[y_dim][x_dim] = value
