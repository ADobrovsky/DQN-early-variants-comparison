import time

import gymnasium as gym
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


# Simple NN structure as proposed by tutorial
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization.
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)



def select_action(state):
    return policy_net(state.to(device)).max(1)[1].view(1, 1)  # pick best action form Q network


env = gym.make('CartPole-goal',render_mode="human")


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

policy_net = DQN(n_observations, n_actions).to(device)  # init policy net
policy_net.eval() # is not gonna be changed for demonstration purpose
print(policy_net)


# TODO: use your WandB credentials for logging
# wandb.login(relogin=True, key='')

# log the general experiment setup and hyperparameters
MODEL_NAME = '3layer-tutorial-standard'
CRITERION_NAME = 'NONE'
wandb.init(project='STML-project-gymnasium-cart-pole-goal-test', save_code=False,
           config={"epochs": num_episodes, "batch_size": BATCH_SIZE, "gamma": GAMMA, "eps_start": EPS_START,
                   "eps_end": EPS_END, "eps_decay": EPS_DECAY, "target-update": TARGET_UPDATE, "tau": TAU, "lr": LR,
                   "seed": seed, "model": MODEL_NAME,
                   "n_observations": n_observations, "n_actions": n_actions, "criterion": CRITERION_NAME})

cumulated_rewards = []
episode_durations = []
steps_done = 0
highest_mean_duration = 0  # determine the maximum average reward achieved so far

# TODO specify the path to your models
PATH = 'C:/Users/'
# best:
MODEL_FILE_NAME = 'GOAL-DQN-double-soft-update-3layer-simple42-5-model.pt'
# local optimum balance to edge of field:
MODEL_FILE_NAME = 'GOAL-DQN-3layer-simple1337-5-model.pt'
# never reached a perfectly fast solution/ reaches goal slowly balancing:
MODEL_FILE_NAME = 'GOAL-DQN-fixed-target-hard-update-3layer-simple42-6-model.pt'

policy_net.load_state_dict(torch.load(PATH + MODEL_FILE_NAME))

for i_episode in range(num_episodes):

    # optionally display the environment
    # only possible when envoked environment with render mode = 'human'
    #env.render()
    #time.sleep(1 / 30)  # FPS

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
