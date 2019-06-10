[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"


# Project 2: Continuous Control

### Summary: 
In this notebook I implemented a **Deep Deterministic Policy Gradient** ([DDPG](https://arxiv.org/abs/1509.02971)) to solve Unity's *Reacher* environment for the case of a single agent. I explored hyperparameters that made the vanilla case (which is DDPG with Action Space Noise, a stochastic process added to the output of the Actor network), and then I introduceced two variations:


 - **Parameter Space Noise** ([PSN](https://arxiv.org/abs/1706.01905)), where the noise is added directly to the weights of the Actor network, and
 
 - **Prioritized Experience Replay** ([PER](https://arxiv.org/abs/1511.05952)], where the memory buffer is stored and sampled by following a distribution that assigns higher priority to transitions that result into higher TD errors.

### Goal and Motivation:
The main goal for this project is to assess the improvements on the learning algorithm by switching the exploration strategy from perturbating the actions to perturbating the actor weights in an adaptative way. My main motivation is that I have not seen PSN included in State-of-the-Art models, yet I find it a simple but versatile approach for the exploration problem. Also, I wanted to include a more advanced memory model than the uniformly-sampled used in the DDPG paper, so I explored the incorporation of PER into my learner.

### Results:
I conducted an ablation study in order to identify the individual contributions of both PSN and PER to the vanilla DDPG case. The number of episodes that took each agent to solve the environment are:

 
 1. **Vanilla DDPG**: not solving - more than 1000 episodes

 2. **DDPG with PSN**: 257 episodes
 
 3. **DDPG with PER**: 452 episodes
 
 4. **DDPG with PSN and PER**: 276 episodes

### Environment

For this project, the work will be with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The Benchmark Mean Reward is 30.

### Solving criteria

This project uses a Unity environment that contains a single agent. The task is episodic, and in order to solve the environment,  the agent must get an average score of +30 over 100 consecutive episodes.
```python
env = UnityEnvironment(file_name='Reacher.exe')
```
### Instructions
This project runs locally on Windows 10 environment. Here are the steps to setup the environment:
1. Create (and activate) a new environment with Python 3.6.
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```

2. Install these dependencies in the environment on windows 10
	- __Install Unity ML-Agents__
	```bash
	pip3 install --user mlagents
	```	
	- __Install Unity Agents__
	```bash
	pip install unityagents
	```	
	- __Install Pytorch__
	```bash
	conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
	```
3. Download the `Reacher` environment from one of the links below and select the environment that matches your Windows operating system:
    - **_Version 1: One (1) Agent_**
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)
 
4. Place the file in the repo folder, and unzip (or decompress) the file.

### Train the agent 
Execute  the cells within `report.ipynb` in order to initialize the environment with making necessary adjustments for the path to the UnityEnvironment in the code, initialize the agent, and perform the training of a single agent.

### Results
Results shown that while the vanilla DDPG agent needs more than 1000 episodes to solve with the chosen set of hyperparameters (it worth to notice no hyperparameter tuning was performed), when switching the noise process to an adaptative noise in the state parameters (actor weights) the learning curve improves drastically. On the other hand, while sampling experiences in a prioritized way also improves the learning process, it also drastically increases the training wall time, mostly because the implementation of PER I made is the naive one, where the sampling process turns into a binary search of complexity O(n). The following curve shows all learning curves together for comparison:

![Results for all trained agents](https://github.com/thenickben/dlrnd-p2-reacher/blob/master/figures/results.png)