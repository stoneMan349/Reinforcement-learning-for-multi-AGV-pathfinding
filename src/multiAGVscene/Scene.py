import sys
import pygame
from utils.utils import Direction as Dir
from utils.utils import Working as Wokring
from utils.utils import ColorBox as ColorBox
from utils.settings import SuperParas as SuperPara
import os

sys.path.append(os.path.dirname(__file__))


"""重大修改：调整结构，不一定每次都要渲染图像
核心都在rungame方法进行修改"""
class Scene:
    def __init__(self, layout, explorer_group):
        """"all parameters about drawing scene"""
        self.layout = layout
        self.control_pattern = ""
        self.clock = None
        self.running_time = 0
        self.FPS = SuperPara.FPS  # 300
        self.x_width = self.layout.scene_x_width
        self.y_width = self.layout.scene_y_width
        self.max_training_steps = int((self.x_width/2+self.y_width) * 3 * ((self.x_width*self.y_width)/2.5))
        # other parameters related to scene
        self.border_width = 30
        self.line_width = 2
        self.color_box = ColorBox()
        # size of main interface
        self.cell_width = 36
        self.interface_width = self.x_width * self.cell_width - (self.x_width - 1) * self.line_width
        self.interface_height = self.y_width * self.cell_width - (self.y_width - 1) * self.line_width
        self.interface_start_x = self.border_width
        self.interface_start_y = self.border_width
        # size of sidebar
        self.sidebar_width = 200
        self.sidebar_height = self.interface_height + 2 * self.border_width
        self.sidebar_start_x = self.interface_width + 2 * self.border_width
        self.sidebar_start_y = 0
        # size of screen
        self.screen_width = self.interface_width + 2 * self.border_width + self.sidebar_width
        self.screen_height = self.interface_height + 2 * self.border_width
        # parameters related to AGV
        self.AGV_icon_scale = 0.8
        self.explorer_group = explorer_group
        if len(self.explorer_group) == 0:
            print("WARNING: the number of veh is zero")
            return
        # all surfaces
        self.screen = None
        self.interface = None
        self.sidebar = None
        """"all parameters about training"""
        self.working_manager = Wokring()
        self.dir = Dir()
        self.action_number = self.dir.action_num()
        self.smart_controller = None
        """是否现实动画"""
        self.render = True

    def init(self):
        self.layout.init()
        for explorer in self.explorer_group:
            explorer.init()
        """observations"""

    def render_init(self):
        """渲染screen"""
        pygame.init()
        pygame.display.set_caption('multiAGV World')
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.screen.fill(self.color_box.GRAY_Color)
        # screen display
        self.refresh_screen(self.explorer_group[0])
        self.clock = pygame.time.Clock()
        self.clock.tick(self.FPS)

    def render_step(self):
        """渲染layout和explorer"""
        pass

    def run_game(self, control_pattern="manual", smart_controller=None, render=True):
        self.render = render

        self.control_pattern = control_pattern  # 0: "train_NN", 1: "use_NN", 2: "A_star", 3: "auto", 4: "manual"
        self.explorer_group[0].create_explorer()  # 创建第一辆veh实体，后续车辆需要在画面刷新后创建
        """"--screen--"""
        if self.render:
            self.render_init()

        if self.control_pattern == "manual":
            self.run_mode(self.control_pattern)
        if self.control_pattern in ["A_star", "auto"]:
            return self.run_mode(self.control_pattern)
        if self.control_pattern == "intelligent":
            return self.run_mode(self.control_pattern, smart_controller)

    def run_mode(self, running_type="manual", smart_controller=None):
        """control running process"""
        """---establish running mode---"""
        self.running_time = 0
        if running_type == "manual":
            if len(self.explorer_group) > 1:
                print("WARNING: manual mode can only control one AGV")
                return
        self.smart_controller = smart_controller
        expert_info = []
        expert_name = []
        expert_action = []
        """---main loop---"""
        while True:
            self.running_time += 1
            if self.render:
                self.create_interface()  # create interface
            input_action, key_input_action = "", ""
            """standard code: exit game"""
            if self.render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sys.exit()
                    if event.type == pygame.KEYDOWN and running_type in ["manual"]:
                        key_input_action = self.get_action_by_input(event.key)
            """---explorers take action in turns---"""
            for explorer in self.explorer_group:
                """---state check---"""
                # explorer has created
                if not explorer.has_created:
                    break
                if self.layout.task_finished:
                    return expert_info, expert_name, expert_action  # 使用A_star方法收集专家经验
                # current explorer has no more task
                if explorer.all_assigned:
                    self.patch_agv_icon(explorer)
                    continue
                """---get action---"""
                all_info = []
                if running_type in ["A_star", "auto"]:
                    input_action = explorer.find_path_astar(self.explorer_group)  # through strategy
                    expert_info.append(self.create_info())
                    expert_name.append(explorer.explorer_name)
                    expert_action.append(self.dir.action_str_value(input_action))
                if running_type in ["manual"]:
                    input_action = key_input_action
                if running_type in ["intelligent"]:
                    all_info = self.create_info()  # all_info=[layout, [veh1_details], [veh2_details]...]
                    input_action = self.smart_controller.choose_action(all_info, explorer.explorer_name)  # get action
                    input_action = self.dir.action_value_str(input_action)
                """execute action"""
                if running_type == "intelligent":
                    reward, is_end = explorer.execute_action(input_action, all_info, self.explorer_group)
                    # print("is_end", is_end)
                    # print("reward", reward)
                    self.patch_agv_icon(explorer)
                    # 矫正一些值
                    is_end = True if (self.layout.task_finished or self.running_time >= self.max_training_steps) else is_end
                    self.smart_controller.store_info(self.create_info(), reward, is_end, explorer.explorer_name)
                    if is_end:
                        return self.running_time
                else:
                    if input_action != "":
                        explorer.execute_action(input_action)
                    self.patch_agv_icon(explorer)
            """查看是否可以创建新的veh（初始位置空出）"""
            flags = self.check_new_veh()
            if flags != 0:
                explorer = self.explorer_group[flags]
                self.patch_agv_icon(explorer)

            if self.render:
                self.create_sidebar()  # update sidebar
                self.screen.blit(self.interface, (self.interface_start_x, self.interface_start_y))
                self.screen.blit(self.sidebar, (self.sidebar_start_x, self.sidebar_start_y))
                pygame.display.flip()  # 更新屏幕内容
            else:
                pass

    def get_action_by_input(self, event_key):
        key_input_action = ""
        if event_key == pygame.K_UP:
            key_input_action = self.dir.value_str[0]
        if event_key == pygame.K_DOWN:
            key_input_action = self.dir.value_str[2]
        if event_key == pygame.K_LEFT:
            key_input_action = self.dir.value_str[3]
        if event_key == pygame.K_RIGHT:
            key_input_action = self.dir.value_str[1]
        return key_input_action

    def refresh_screen(self, explorer):
        if self.render:
            self.create_interface()  # update interface
            self.patch_agv_icon(explorer)  # update AGV
            self.create_sidebar()  # update sidebar
            # update screen
            self.screen.blit(self.interface, (self.interface_start_x, self.interface_start_y))
            self.screen.blit(self.sidebar, (self.sidebar_start_x, self.sidebar_start_y))
            pygame.display.flip()  # 更新屏幕内容
        else:
            pass

    def patch_agv_icon(self, explore_group):
        if self.render:
            if not isinstance(explore_group, list):
                explore_group = [explore_group]
            for explore in explore_group:
                agv_image = pygame.image.load(explore.icon_path)
                agv_image = pygame.transform.scale(agv_image, (
                self.cell_width * self.AGV_icon_scale, self.cell_width * self.AGV_icon_scale))  # 图像缩放
                agv_image = pygame.transform.rotate(agv_image, self.image_rotate_angle(explore.action_str))
                agv_position = self.position_rectify(explore.current_place[0], explore.current_place[1], is_agv=True)
                self.interface.blit(agv_image, agv_position)
        else:
            pass

    def image_rotate_angle(self, action):
        angle = 0
        if action == self.dir.value_str[0]:
            angle = 90
        if action == self.dir.value_str[1]:
            angle = 0
        if action == self.dir.value_str[2]:
            angle = -90
        if action == self.dir.value_str[3]:
            angle = 180
        return angle

    def create_interface(self):
        # interface
        self.interface = pygame.Surface((self.interface_width, self.interface_height), flags=pygame.HWSURFACE)
        self.interface.fill(color=self.color_box.WHITE_COLOR)
        # draw blocks
        for y_dim in range(self.y_width):
            for x_dim in range(self.x_width):
                pygame.draw.rect(self.interface, self.color_box.BLACK_COLOR, (
                x_dim * (self.cell_width - self.line_width), y_dim * (self.cell_width - self.line_width),
                self.cell_width, self.cell_width), self.line_width)
                # draw picking_station and storage_station
                if (x_dim + 1, y_dim + 1) in self.layout.picking_station_list:
                    self.draw_block(self.interface, self.color_box.GRAY_Color, x_dim, y_dim)
                    img = pygame.image.load("multiAGVscene/icons/cart.png")
                    img = pygame.transform.scale(img, (self.cell_width*self.AGV_icon_scale, self.cell_width*self.AGV_icon_scale))
                    self.interface.blit(img, self.position_rectify(x_dim+1, y_dim+1))
                if (x_dim + 1, y_dim + 1) in self.layout.storage_station_list:
                    if self.layout.layout[y_dim][x_dim] == 1.8:
                        self.draw_block(self.interface, self.color_box.PINK_COLOR, x_dim, y_dim)
                        # img = pygame.image.load("icons/pod_empty.png")
                        # img = pygame.transform.scale(img, (self.cell_width * self.AGV_icon_scale, self.cell_width * self.AGV_icon_scale))
                        # self.interface.blit(img, self.position_rectify(x_dim + 1, y_dim + 1))
                    elif self.layout.layout[y_dim][x_dim] == 1.3:
                        self.draw_block(self.interface, self.color_box.GREEN_COLOR, x_dim, y_dim)
                        # img = pygame.image.load("icons/pod_full.png")
                        # img = pygame.transform.scale(img, (self.cell_width * self.AGV_icon_scale, self.cell_width * self.AGV_icon_scale))
                        # self.interface.blit(img, self.position_rectify(x_dim + 1, y_dim + 1))
                    else:
                        self.draw_block(self.interface, self.color_box.RED_COLOR, x_dim, y_dim)
                        # img = pygame.image.load("icons/pod_full.png")
                        # img = pygame.transform.scale(img, (self.cell_width * self.AGV_icon_scale, self.cell_width * self.AGV_icon_scale))
                        # self.interface.blit(img, self.position_rectify(x_dim + 1, y_dim + 1))
                # draw axis value
                if x_dim == 0:
                    self.draw_scale(self.screen, float(y_dim), "y")
                if y_dim == 0:
                    self.draw_scale(self.screen, float(x_dim), "x")
        # self.draw_scale(self.screen, float(self.x_width), "x")
        # self.draw_scale(self.screen, float(self.y_width), "y")

    def create_sidebar(self):
        """"--sidebar--"""
        # sidebar
        self.sidebar = pygame.Surface((self.sidebar_width, self.sidebar_height), flags=pygame.HWSURFACE)
        self.sidebar.fill(color=self.color_box.GRAY_Color)
        # title
        font_title = pygame.font.SysFont("Times New Roman", 30)
        title = font_title.render(str("AMR World"), True, self.color_box.BLACK_COLOR)
        time = font_title.render(str("Time:" + str(self.running_time)), True, self.color_box.BLACK_COLOR)
        title_rect = title.get_rect()
        self.sidebar.blit(title, (self.sidebar_width / 2 - title_rect.width / 2, self.sidebar_height / 15))
        self.sidebar.blit(time, (self.sidebar_width / 2 - title_rect.width / 2, self.sidebar_height / 15 + 40))
        # title
        font_agv = pygame.font.SysFont("Times New Roman", 15)
        t_l = font_agv.render(str("t_pos:        " + str(self.explorer_group[0].target_position)), True,
                              self.color_box.BLACK_COLOR)  # target location
        c_l = font_agv.render(str("c_pos:        " + str(self.explorer_group[0].current_place)), True,
                              self.color_box.BLACK_COLOR)  # current location
        l_l = font_agv.render(str("l_pos:        " + str(self.explorer_group[0].last_place)), True,
                              self.color_box.BLACK_COLOR)  # last location
        r_s = font_agv.render(str("state:        " + str(self.explorer_group[0].running_state)), True,
                              self.color_box.BLACK_COLOR)  # running_state
        self.sidebar.blit(t_l, (20, self.sidebar_height / 3))
        self.sidebar.blit(c_l, (20, self.sidebar_height / 3 + 20))
        self.sidebar.blit(l_l, (20, self.sidebar_height / 3 + 40))
        self.sidebar.blit(r_s, (20, self.sidebar_height / 3 + 60))
        # 输出动作分布
        act = font_agv.render(str("action: "), True,
                              self.color_box.BLACK_COLOR)  # action
        self.sidebar.blit(act, (20, self.sidebar_height / 3 + 80))
        pygame.draw.rect(self.sidebar, self.color_box.BLACK_COLOR, (60+2, self.sidebar_height / 3 + 80, 110, 62), 1)
        font_action = pygame.font.SysFont("Times New Roman", 10)
        up = font_action.render(str("U:"+str(self.explorer_group[0].action_distribution[0])), True, self.color_box.BLACK_COLOR)  # action
        right = font_action.render(str("R: " + str(self.explorer_group[0].action_distribution[1])), True, self.color_box.BLACK_COLOR)  # action
        down = font_action.render(str("D: "+str(self.explorer_group[0].action_distribution[2])), True, self.color_box.BLACK_COLOR)  # action
        left = font_action.render(str("L: "+str(self.explorer_group[0].action_distribution[3])), True, self.color_box.BLACK_COLOR)  # action
        self.sidebar.blit(left, (60+4, self.sidebar_height / 3 + 105))
        self.sidebar.blit(up, (95+4, self.sidebar_height / 3 + 80))
        self.sidebar.blit(right, (130+4, self.sidebar_height / 3 + 105))
        self.sidebar.blit(down, (95+4, self.sidebar_height / 3 + 130))
        # Author
        font_author = pygame.font.SysFont("Times New Roman", 15)
        author_detail = font_author.render(str("Author: Stone"), True, self.color_box.BLACK_COLOR)
        self.sidebar.blit(author_detail, (20, 5 * self.sidebar_height / 6))

    def draw_scale(self, screen, value, axis):
        font = pygame.font.SysFont("Times New Roman", 12)
        rect = font.render(str(value + 1), True, self.color_box.BLACK_COLOR)
        if axis == "x":
            screen.blit(rect, (value * (
                        self.cell_width - self.line_width) + self.interface_start_x - rect.get_width() / 2 + self.cell_width / 2,
                               self.border_width / 3))
        elif axis == "y":
            screen.blit(rect, (self.border_width / 4, value * (
                        self.cell_width - self.line_width) + self.interface_start_y - rect.get_height() / 2 + self.cell_width / 2))

    def draw_block(self, interface, color, x_dim, y_dim):
        pygame.draw.rect(interface, color, (x_dim * (self.cell_width - self.line_width) + self.line_width,
                                            y_dim * (self.cell_width - self.line_width) + self.line_width,
                                            self.cell_width - self.line_width,
                                            self.cell_width - self.line_width))

    def position_rectify(self, x_dim, y_dim, is_agv=False):
        x_position = (x_dim - 1) * (self.cell_width - self.line_width) + self.line_width
        y_position = (y_dim - 1) * (self.cell_width - self.line_width) + self.line_width
        if is_agv:
            x_position = x_position + self.cell_width * (1-self.AGV_icon_scale) * 0.25
            y_position = y_position + self.cell_width * (1-self.AGV_icon_scale) * 0.25
        else:
            x_position = x_position + self.cell_width * (1 - self.AGV_icon_scale) * 0.25
            y_position = y_position + self.cell_width * (1 - self.AGV_icon_scale) * 0.25
        position = (x_position, y_position)
        return position

    def create_info(self):
        """infos for reinforcement learning"""
        layout = self.layout.layout_original
        all_info = [layout]
        for explorer in self.explorer_group:
            if explorer.has_created:
                one_explorer = [explorer.explorer_name, explorer.current_place, explorer.target_position,
                                explorer.loaded]
                all_info.append(one_explorer)
            else:
                break
        return all_info

    def check_new_veh(self):
        init_pos_occupy = False
        flags = 0
        for explore_num in range(len(self.explorer_group)):
            if self.explorer_group[explore_num].has_created:
                if self.explorer_group[explore_num].current_place == [1, 1]:
                    init_pos_occupy = True
            else:  # 所有被创建的车辆都已经检查过了
                if not init_pos_occupy:
                    self.explorer_group[explore_num].create_explorer()
                    flags = explore_num
                    break
        return flags
