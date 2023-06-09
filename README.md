<h1 align="center">RL on Multi-AGV Collision-free Pathfinding</h1>
<h2 align="center">
This README is not finished yet. And some Exper_Data for training is too big to upload, you can collect them by yourself.
</h2>

<p align="center">
<img src="./ReadMe_Assets/fig1_title.png" width="80%"></p>


## Introduction

**Example: Multi-AGV in an robotic mobile fulfilment system**

<p align="center">
<img src="./ReadMe_Assets/video1-overview.gif" width="80%"></p>

<img align="right" src="./ReadMe_Assets/fig5_project_structure.png" width="25%">

**This project consists of two main parts: a multi-AGV learning environment and several reinforcement learning (RL) algorithms, including existing algorithms as well as our own innovation.**

1.The multi-AGV learning environment consists of three classes: Layout Class, which manages the layout of the scene; Explorer Class, which manages the AGVs; and Scene Class, which is a container that accommodates both the layout and AGVs so that the AGVs can operate in a specific layout.

2.As for the RL algorithms, we have implemented several well-known approaches such as Deep Q-Network (DQN), Double Deep Q-Network (DDQN), Actor-Critic (AC), and Policy-Gradient (PG), as well as our own novel algorithm, A* guiding DQN (AG-DQN).  
-*Aiming at solving pathfinding problem on a 2D map, we used a Convolutional Neural Network and specifically designed state representation.*  
-*To enhance the training effect, we applied techniques such as Behavioral Cloning, Sparse Reward, and Limited Visual.*

3.The figure on the right shows the structure of this project.

**This project was created by [LeiLuo](https://scholar.google.com/citations?user=auFJLXkAAAAJ) under the supervision of Professor [Zhao Ning](http://me.ustb.edu.cn/shiziduiwu/jiaoshixinxi/2022-03-24/434.html) form University of Science and Technology of Beijing (Beijing, China)** .

## Robotic Mobile Fulfilment System (RMFS)
<p align="center">
<img src="./ReadMe_Assets/fig3_RMFS.gif" width="35%">

1. The figure shows an RMFS made by [Quicktron Robots](https://www.quicktron.com/).
2. **The components of an RMFS** include AGVs (for transferring shelves), shelves (for storing goods), a track (on which the AGVs can move), a picking station (where workers can pick goods), and a charge room (where the AGVs can recharge).
4. **The goal of the RMFS** is to have AGVs transfer shelves to the picking station, where workers can correctly select, package, and deliver the goods.
5. **The aim of the AGVs** is to transfer the necessary shelves to the picking station and return them to their original location once the workers have finished picking the goods.
6. **AGVs have a special rule** that when they are empty, they can travel under the shelves to shorten their travel path. However, when the AGVs are full, they are only allowed to move on the tracks.
7. **The objective of this project** is to develop effective methods to guide the AGVs in finding the shortest and safest path to complete a large number of tasks without collisions.

*Tips:  
-Unlike other pathfinding scenes, the RMFS is unique in that it is full of obstacles (shelves) that pose a challenge to AGVs, and the path is often too narrow to accommodate two AGVs at once.  
-Our method is not limited to the RMFS scenario but can also be applied in other pathfinding situations if they are created using a similar grip-based approach.*

## How to use
### 1. src.main.py is the Entry of the project  
Just run src.main.py to run this project.  
### 2. Two ways to create a scene  
<p align="center">
<img src="./ReadMe_Assets/fig2_scene_construction.png" width="40%">
<img src="./ReadMe_Assets/fig2_scene_construction2.PNG" width="39%"></p>

*Notes:  
The red block represents storage station, the gray block represents picking station, the white block represents track.  
The green block represents current target place, the pink block represents current target place's task has been finished.  
The left figure represents construction method 2.1. The right figure represents construction method 2.2*

**2.1 Create a rectangular layout by entering 5 parameters**  
Adjust following parameters to create a rectangular layout
```python
ss_x_width, ss_y_width, ss_x_num, ss_y_num, ps_num = 4, 2, 2, 2, 2
# ss_x_width: The number of storage stations in the x-axis direction of the storage station island
# ss_y_width: The number of storage stations in the y-axis direction of the storage station island
# ss_x_num: The number of storage station island in X-axis direction
# ss_y_num: The number of storage station island in y-axis direction
# ps_num: The number of picking station
```

**2.2 Create a special layout by entering 'layout_list'**  
Enter a list to create a special layout
```python
    layout_list = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 1, 1, 1, 1, 1, 1, 0, 0],
                   [0, 1, 1, 1, 1, 1, 1, 0, 0],
                   [0, 0, 0, 0, 0, 0, 1, 1, 0],
                   [0, 0, 0, 2, 0, 0, 1, 1, 0],
                   [0, 0, 0, 0, 0, 0, 1, 1, 0],
                   [0, 1, 1, 1, 1, 1, 1, 0, 0],
                   [0, 1, 1, 1, 1, 1, 1, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0]] 
# 0: track,  1: storage station,  2: picking station.
# Note: Any storage station needs to be connected to a road in at least one direction.
```
### 3. Five control modes of AGVs
Select the control modes by changing parameter 'control_mode'
```python
    control_type = {0: "train_NN", 1: "use_NN", 2: "A_star", 3: "manual", 4: "Expert"}
    control_mode = 3
```

**3.1 Manual mode**  
Control AGV by keyboard. (In this mode, you can only control a single AGV)
```python
    control_mode = 3
```

**3.2 A\* mode**  
Control AGV by A* algorithm.
```python
    control_mode = 2
```
**3.3 Training mode**  
Training a neural network to guide AGV. (RL algorithms will be discussed in next section)
```python
    control_mode = 0
```

**3.4 RL mode**  
Control AGV by a well-trained neural network.
```python
    control_mode = 1
```

**3.5 Expert mode**  
Collect expert experience by using A* algorithm to control AGV.
```python
    control_mode = 4
```

### 4. Three RL algorithms
We provide three algorithms including PG, AC and DQN.  
Import different packages to experience these algorithms, the control mode should choose to "Training mode"
```python
# from algorithm.AC_structure.Controller import ACAgentController as modelController
from algorithm.PG_structure.Controller import PGAgentController as modelController
# from src.algorithm.MADQN_structure.Controller import MADQNAgentController as modelController
```
*Tips:You can check the logic of these algorithms online, so we won’t go into details here.*

## Details of RL algorithm
### 1.State
1.1 We use three matrices to construct the State. Valid_Location_Matrix, Current_Location_Matrix and Target_Location_Matrix.  
1.2 Valid_Location_Matrix describes which block (marked as 1) the AGV can access.  
1.2 Current_Location_Matrix describes which block (marked as 1) is the AGV's current location.  
1.3 Target_Location_Matrix describes which block (marked as 1) is the AGV's target location.  
<p align="center">
<img src="./ReadMe_Assets/fig7_state.png" width="70%"></p>

[*Tips: It is unwise to create matrices the same size as the scene, especially as the scene gets bigger. We use Limited Visual to improve training performance.*](#LimitedVisual)
### 2.Action
Action space includes five actions: up, right, down, left and stop. Each action corresponds to a number.  
You can look them up in src.utils.utils.py  
```python
str_value = {"UP": 0, "RIGHT": 1, "DOWN": 2, "LEFT": 3, "STOP": 4}
value_str = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT", 4: "STOP"}
```
*Tips: If there is only one AGV, then action "stop" is unnecessary.*
### 3.Reward
3.1 Corresponding to the result of performing an action, the agent can get three types of rewards: positive reward (reward value +1), negative reward (reward value -1) and normal reward (reward value 0).  
3.2 When the AGV reaches the destination, it gets a positive reward; when the AGV hits an obstacle or runs out of the scene, it gets a negative reward; in other cases, the AGV gets a normal reward.  
[*Tips: This is a typical sparse reward problem. We use Reward Reshaping to improve training performance.*](#RewardReshaping)
### 4.Convolutional Neural Network
According to the formation of State, convolutional neural network is used.   
<p align="center">
<img src="./ReadMe_Assets/fig6_NN_structure.png" width="30%"></p>

## Other Technics
### 1.A* guiding DQN (AG-DQN)
1.1 This is the most efficient method we have found to greatly improve the training effect of the DQN algorithm on AGV Pathfinding Problem.  
1.2 The core change is to replace the random exploration used by the exploration method with the A* algorithm.  
*Tips: We will put some figures of the training process later.*
### 2.Behavioral Cloning
2.1 Behavioral Cloning is a specific method imitation learning.  
2.2 Simply put, Behavioral Cloning is used to enhance the utilization of data, thereby increasing the training speed of neural networks. Because the neural network is initialized completely randomly, if we let the agent interact with the environment from the beginning, it will not accumulate useful experience and the neural network will be difficult to optimize. So we can use some expert experience to pre-train the neural network, and then let the agent interact with the environment to continue to optimize the neural network.  
2.3 The A* algorithm is used as an expert for the pathfinding problem.  
2.4 Behavioral Cloning performs well in PG and AC algorithms.  

*Here are some pages where you can learn more about behavior cloning. But they are all in Chinese, you can search for more relevant English materials on the Internet.*  
[Behavioral Cloning,](https://hrl.boyuai.com/chapter/3/%E6%A8%A1%E4%BB%BF%E5%AD%A6%E4%B9%A0) [Code](https://github.com/boyu-ai/Hands-on-RL/blob/main/%E7%AC%AC15%E7%AB%A0-%E6%A8%A1%E4%BB%BF%E5%AD%A6%E4%B9%A0.ipynb)  
*Tips: We will put some figures of the training process later.*
<span id="RewardReshaping"></span>
### 3.Reward Reshaping

<span id="LimitedVisual"></span>
### 4.Limited Visual



## Other Tips
### Using our environment to test your algorithms

### How to import a serial of task

### It is not sufficient in solving multi-AGV pathfinding in huge scene, it only works with small scenes and several AGVs

### Ways to improve it effect in solving multi-AGV pathfinding in huge scene



## Citation
If you find our project helpful, please cite our paper related to this project:  
[1] Luo L, Zhao N, Zhu Y, et al. A* guiding DQN algorithm for automated guided vehicle pathfinding problem of robotic mobile fulfillment systems[J]. Computers & Industrial Engineering, 2023, 178: 109112.

