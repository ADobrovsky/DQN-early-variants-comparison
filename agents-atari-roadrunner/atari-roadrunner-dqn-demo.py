import time

import gymnasium as gym
from gymnasium.wrappers import FrameStack, AtariPreprocessing
import math
import random
import numpy as np
from collections import namedtuple, deque
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb  # for logging

# implementation follows in large parts the pytorch tutorial of Adam Paszke and Mark Towers
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# retrieved 11.06.2023


#proper layer initialisation
def init_cnn(module):
    #use kaiming for CNN with ReLu
    torch.nn.init.kaiming_normal_(module.weight, a=0, mode='fan_in', nonlinearity='relu')
    torch.nn.init.constant_(module.bias, 0)
    return module

def init_linear(module):
    torch.nn.init.xavier_normal_(module.weight)
    torch.nn.init.constant_(module.bias, 0)
    return module

class DQN(nn.Module):
    def __init__(self, input, outputs):  # heights width screen, number actions
        super(DQN, self).__init__()
        self.conv1 = init_cnn(nn.Conv2d(4, 16, kernel_size=4, stride=3, padding=1))
        self.conv2 = init_cnn(nn.Conv2d(16, 32, kernel_size=3, stride=2))
        self.conv3 = init_cnn(nn.Conv2d(32, 64, kernel_size=3, stride=1))
        #self.conv4 = init_layer(nn.Conv2d(32, 64, kernel_size=3, stride=1))
        self.fully = nn.Linear(7744, 512)
        self.denseOut = init_linear(nn.Linear(512, outputs))

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fully(x.view(x.size(0), -1)))
        return self.denseOut(x)

class DuelingDQN(nn.Module):
    def __init__(self, input, outputs):  # heights width screen, number actions
        super(DuelingDQN, self).__init__()
        self.conv1 = init_cnn(nn.Conv2d(4, 16, kernel_size=4, stride=3, padding=1))
        self.conv2 = init_cnn(nn.Conv2d(16, 32, kernel_size=3, stride=2))
        self.conv3 = init_cnn(nn.Conv2d(32, 64, kernel_size=3, stride=1))
        self.fully = nn.Linear(7744, 512)
        self.V = nn.Linear(512, 1)
        self.A = nn.Linear(512, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fully(x.view(x.size(0), -1)))
        # scalar state value
        V = self.V(x)
        # advantages for actions
        A = self.A(x)
        # aggregation of state value and action values
        Q = V + (A - torch.mean(A, dim=1, keepdim=True))
        return Q



def select_action(state):
    return policy_net(state.to(device)).max(1)[1].view(1, 1)  # pick best action form Q network


env = gym.make('RoadRunnerNoFrameskip-v4', render_mode="human")
env = AtariPreprocessing(env)
env = FrameStack(env,4)
print(env.observation_space)
obs, _ = env.reset()
print(obs.shape)


# use seed for reproducibility for comparing experiments
seed = 0
random.seed(seed)
torch.random.manual_seed(seed)
np.random.seed(seed)
env.action_space.seed(seed)
env.observation_space.seed(seed)

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'cuda available? {torch.cuda.is_available()}')

# Structure for the experience replay memory
# mapping state action pairs to their resulting next state and reward
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

env.reset()  # back to initial state of environment

# HYPERPARAMETERS
BATCH_SIZE = 'NONE'  # number of transitions sampled from the replay buffer
GAMMA = 'NONE' # discount factor, must be <1 for finite sum, 0:greedy
EPS_START = 'NONE'  # starting value of epsilon
EPS_END = 'NONE'  # final value of epsilon
EPS_DECAY = 'NONE' # controls the rate of exponential decay of epsilon, higher means a slower decay
LR = 'NONE'  # learning rate of the optimizer

TARGET_UPDATE = 'NONE'  # update rate for double deep Q-learning with hard update
TAU = 'NONE'  # update rate for double deep Q-learning with soft update

num_episodes = 600  # the number of episodes for the training

n_actions = env.action_space.n  # Get number of actions from gym action space
state, info = env.reset()
n_observations = len(state)  # Get the number of state observations

policy_net = DuelingDQN(n_observations, n_actions).to(device)  # init policy net

#TODO: specify the path to your models
PATH = 'C:/Users/'
# best mean cumulated rewards
MODEL_FILE_NAME = 'atari-DQN-fixed-target-soft-update-dueling-small-conv1337-5-4-model.pt'
#MODEL_FILE_NAME = 'atari-DQN-double-soft-update-dueling-small-conv1337-5-4-model.pt'
#MODEL_FILE_NAME = 'atari-DQN-double-soft-update-small-conv42-1-4-a-model.pt'
#policy_net = DQN(n_observations, n_actions).to(device)  # init policy net

#local optimum long duration run-very interesting:
MODEL_FILE_NAME = 'atari-DQN-dueling-small-conv1337-5-4-model.pt'

policy_net.load_state_dict(torch.load(PATH + MODEL_FILE_NAME))
policy_net.eval() # is not gonna be changed for demonstration purpose
print(policy_net)


# TODO: use your WandB credentials for logging
# wandb.login(relogin=True, key='')

# log the general experiment setup and hyperparameters
CRITERION_NAME = 'NONE'
MODEL_NAME = 'small-conv'

wandb.init(project='STML-project-gymnasium-atari-pinball-test',
           config={"epochs": num_episodes, "batch_size": BATCH_SIZE, "gamma": GAMMA, "eps_start": EPS_START,
                   "eps_end": EPS_END, "eps_decay": EPS_DECAY, "target-update": TARGET_UPDATE, "tau": TAU, "lr": LR,
                   "seed": seed, "model": MODEL_NAME,
                   "n_observations": n_observations, "n_actions": n_actions, "criterion": CRITERION_NAME})

cumulated_rewards = []
episode_durations = []
steps_done = 0
highest_mean_duration = 0  # determine the maximum average reward achieved so far

for i_episode in range(num_episodes):


    # Initialize the environment and get it's state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    cumulated_reward = 0  # cumulated reward for logging

    for t in count():
        action = select_action(state)  # Select and perform an action
        wandb.log({"action ": action.item(), "episode": i_episode})  # log selected action
        # send action to environment and receive reward, new observation and env state
        observation, reward, terminated, truncated, _ = env.step(action.item())
        cumulated_reward = cumulated_reward + reward  # add reward to cumulated reward
        reward = torch.tensor([reward], device=device)  # reward needs to be tensor for replay memory

        done = terminated or truncated

        if terminated:  # terminated = pole fell or cart to far from the middle position
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)


        # Move to the next state
        state = next_state

        if done:
            cumulated_rewards.append(cumulated_reward + 1)
            cumulated_rewards_t = torch.tensor(cumulated_rewards, dtype=torch.float)
            episode_durations.append(t + 1)
            episode_durations_t = torch.tensor(episode_durations, dtype=torch.float)
            # deactivated matplotlib plots, plotting is done by Weights&Biases
            wandb.log({"duration": t + 1, "reward_episode": cumulated_reward, "episode": i_episode})
            # Take 100 episode averages and plot them too
            if len(cumulated_rewards_t) >= 100:
                means = cumulated_rewards_t.unfold(0, 100, 1).mean(1).view(-1)
                means = torch.cat((torch.zeros(99), means))
                wandb.log({"mean_cumulated_reward": means[i_episode].item(), "episode": i_episode})

                # in the last episode, print best and last mean for documentation
                if (i_episode == num_episodes - 1):  # call in the last episode
                    print(f'last mean:  {means[-1]}')
                    wandb.run.summary["last_mean"] = means[-1]
                    print(f'max mean:  {torch.max(means)}')
                    wandb.run.summary["max_mean"] = torch.max(means)

            break

print('Complete')
env.render()
env.close()
wandb.finish()
