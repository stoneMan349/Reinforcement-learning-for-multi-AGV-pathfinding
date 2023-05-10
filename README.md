<h1 align="center">RL on Multi-AGV Collision-free Pathfinding</h1>
<h2 align="center">

[![Mentioned in Awesome Vue.js](https://awesome.re/mentioned-badge.svg)](https://github.com/vuejs/awesome-vue)

</h2>

<p align="center">
  
<img src="https://img.shields.io/npm/dy/silentlad">

<img src="https://img.shields.io/badge/made%20by-silentlad-blue.svg" >

<img src="https://img.shields.io/badge/vue-2.2.4-green.svg">

<img src="https://badges.frapsoft.com/os/v1/open-source.svg?v=103" >

<img src="https://beerpay.io/silent-lad/VueSolitaire/badge.svg?style=flat">

<img src="https://img.shields.io/github/stars/silent-lad/VueSolitaire.svg?style=flat">

<img src="https://img.shields.io/github/languages/top/silent-lad/VueSolitaire.svg">

<img src="https://img.shields.io/github/issues/silent-lad/VueSolitaire.svg">

<img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat">
</p>

<img src="./ReadMe_Assets/fig1_Layout.jpg" width="100%">


## Operation Example

**Multi-AGV controlled by neural network**

<p align="center">
<img src="./ReadMe_Assets/fig2_AGV_controller_by_neural_network.gif" width="80%"></p>

**This project consists of two main parts: a multi-AGV learning environment and several reinforcement learning (RL) algorithms, including existing algorithms as well as our own innovation.**

1.The multi-AGV learning environment consists of three classes: Layout Class, which manages the layout of the scene; Explorer Class, which manages the AGVs; and Scene Class, which is a container that accommodates both the layout and AGVs so that the AGVs can operate in a specific layout.

2.As for the RL algorithms, we have implemented several well-known approaches such as Deep Q-Network (DQN), Double Deep Q-Network (DDQN), Actor-Critic (AC), and Policy-Gradient (PG), as well as our own novel algorithm, A* guiding DQN (AG-DQN).  
-*Aiming at solving pathfinding problem on a 2D map, we used a Convolutional Neural Network and specifically designed state representation.*  
-*To enhance the training effect, we applied techniques such as Behavioral Cloning, Sparse Reward, and Limited Visual.*

**This project was created by [LeiLuo](https://scholar.google.com/citations?user=auFJLXkAAAAJ) under the supervision of Professor Zhao Ning form University of Science and Technology of Beijing (Beijing, China)** .

## Robotic mobile and fulfilment (RMFS)


**Tips:  
-RMFS is full of obstacles, and the path is too narrow to accommodate two AGVs at the same time.  
-Our method can be used in other pathfinding situation if it is a scene created by grip.**


## How to use
**src.main.py is the Entry of the project**
1. Create a simple Scene and control a single AGV using keyboard 
2. Create a simple Scene and let the AGV be controlled by A* algorithm 
3. Train a single AGV in a simple Scene using AG-DQN algorithm 
4. Create a simple Scene and control a single AGV using a trained neural network 

**Please note that other details will be discussed in the following sections**


## State, Action and Reward
### State
### Action
### Reward


## Convolutional Neural Network


## Behavioral Cloning


## Sparse Reward


## Limited Visual



## Other Tips
### Using our environment to test your algorithms
### How to create a special layout
### How to import a serial of task
### Each algorithm includes a simple example

### It is not sufficient in solving multi-AGV pathfinding in huge scene, it only works with small scenes and several AGVs

### Ways to improve it effect in solving multi-AGV pathfinding in huge scene



## Citation
If you find our project helpful, please cite our paper:  
[1] Luo L, Zhao N, Zhu Y, et al. A* guiding DQN algorithm for automated guided vehicle pathfinding problem of robotic mobile fulfillment systems[J]. Computers & Industrial Engineering, 2023, 178: 109112.
