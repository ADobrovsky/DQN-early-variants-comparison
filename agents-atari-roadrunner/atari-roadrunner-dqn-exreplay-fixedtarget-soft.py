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


# Experience Replay memory
class ReplayMemory(object):

    def __init__(self, capacity):  # doubly ended queue (rolling list)
        self.memory = deque([], maxlen=capacity)

    def push(self, *args): # save a transition
        self.memory.append(Transition(*args))

    def sample(self, batch_size):  # return random sample batch of memory
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


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
    global steps_done
    sample = random.random()  # pseudo-random floating point  0.0 <= X < 1.0
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)  # probability of random action, epsilon threshold computes decay
    wandb.log({"epsilon ": eps_threshold, "episode": i_episode})
    steps_done += 1  # action selection is equivalent to one step in the environment
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state.to(device)).max(1)[1].view(1, 1)  # pick best action form Q network
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)  # exploration


def optimize_model():
    if len(memory) < BATCH_SIZE:  # need enough transitions in memory first
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = (torch.cat([s for s in batch.next_state
                                        if s is not None]))#.to(device)
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = ((next_state_values * GAMMA) + reward_batch)
    # print("next state value: \n" + str(next_state_values))
    # print("next state action values: \n" + str(expected_state_action_values))

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))  # loss for update

    # log values for possible debugging
    wandb.log({"loss ": loss, "mean_state_action_value": torch.mean(state_action_values),
               "mean_next_state_values": torch.mean(next_state_values),
               "mean_expected_state_action_values": torch.mean(expected_state_action_values),
               "mean_reward_batch": torch.mean(reward_batch, dtype=torch.float), "episode": i_episode})

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()

    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


env = gym.make('RoadRunnerNoFrameskip-v4')#,render_mode="human")
env = AtariPreprocessing(env)
env = FrameStack(env,4)
print(env.observation_space)
obs, _ = env.reset()
print(obs.shape)

# use seed for (limited) reproducibility for comparing experiments
seed = 42
random.seed(seed)
torch.random.manual_seed(seed)
np.random.seed(seed)
env.action_space.seed(seed)
env.observation_space.seed(seed)
torch.cuda.manual_seed(seed)

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'cuda available? {torch.cuda.is_available()}')

# Structure for the experience replay memory
# mapping state action pairs to their resulting next state and reward
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

env.reset()  # back to initial state of environment

# HYPERPARAMETERS
BATCH_SIZE = 32  # number of transitions sampled from the replay buffer
GAMMA = 0.99  # discount factor, must be <1 for finite sum, 0:greedy
EPS_START = 1.00  # starting value of epsilon
EPS_END = 0.1  # final value of epsilon
EPS_DECAY = 2000  # controls the rate of exponential decay of epsilon, higher means a slower decay
LR = 1e-4  # learning rate of the optimizer

TARGET_UPDATE = 'NONE'  # update rate for double deep Q-learning with hard update
TAU = 0.001  # update rate for double deep Q-learning with soft update

num_episodes = 1000  # the number of episodes for the training

n_actions = env.action_space.n  # Get number of actions from gym action space
observation, info = env.reset()
n_observations = len(observation)  # Get the number of state observations

policy_net = DuelingDQN(n_observations, n_actions).to(device)  # init policy net
target_net = DuelingDQN(n_observations, n_actions).to(device)  # init target net
target_net.load_state_dict(policy_net.state_dict())  # use policy net weights for target net
print(policy_net)

# OPTIMIZER
#optimizer = optim.RMSprop(policy_net.parameters(), LR) # popular RL optimizer, here only policy net is trained, not target
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

# REPLAY MEMORY
memory = ReplayMemory(50000)  # buffer for replay memory, 10000

# TODO: use your WandB credentials for logging
#wandb.login(relogin=True, key='...')

# log the general experiment setup and hyperparameters
EXPERIMENT = "gymnasium-atari-roadrunner"
MODEL_NAME = 'small-conv'
CRITERION_NAME = 'Smooth L1 Loss'
MEMORY_SIZE = 50000
TYPE = 'atari-DQN-fixed-target-soft-update-dueling'
LR_NUM = "4a"
wandb.init(project='STML-project-gymnasium-atari-roadrunner',
           config={"type": TYPE,"epochs": num_episodes, "batch_size": BATCH_SIZE, "gamma": GAMMA, "eps_start": EPS_START,
                   "eps_end": EPS_END, "eps_decay": EPS_DECAY, "target-update": TARGET_UPDATE, "tau": TAU, "lr": LR,
                   "optimizer": optimizer, "seed": seed, "model": MODEL_NAME, "replay_memory": MEMORY_SIZE,
                   "n_observations": n_observations, "n_actions": n_actions, "criterion": CRITERION_NAME, "experiment": EXPERIMENT})

cumulated_rewards = []
episode_durations = []
steps_done = 0
highest_cumulated_reward = 0  # determine the maximum average reward achieved so far

# ToDo specify your save path
SAVE_PATH = 'C:/Users/' + TYPE + '-'\
            + MODEL_NAME + str(seed) + "-" + LR_NUM +'-model.pt'

for i_episode in range(num_episodes):

    # Initialize the environment and get it's state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    cumulated_reward = 0  # cumulated reward for logging

    for t in count():
        action = select_action(state)  # Select and perform an action

        # send action to environment and receive reward, new observation and env state
        observation, reward, terminated, truncated, _ = env.step(action.item())
        cumulated_reward = cumulated_reward + reward  # add reward to cumulated reward
        wandb.log({"action ": action.item(), "reward_step": reward, "episode": i_episode})  # log selected action and received reward

        # for checking that consistent rewards are assigned by the environment
        #if reward > 0 or cumulated_reward > 0 or reward <= -100 or cumulated_reward <= -100 :
            #print(" episode: " + str(i_episode) +", step " + str(t)+" : reward "+ str(reward) + " , cumulated: " + str(cumulated_reward))
        reward = torch.tensor([reward], device=device)  # reward needs to be tensor for replay memory

        done = terminated or truncated

        if terminated:  # terminated = pole fell or cart to far from the middle position
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:

            cumulated_rewards.append(cumulated_reward + 1)
            cumulated_rewards_t = torch.tensor(cumulated_rewards, dtype=torch.float)
            episode_durations.append(t + 1)
            episode_durations_t = torch.tensor(episode_durations, dtype=torch.float)
            # deactivated matplotlib plots, plotting is done by Weights&Biases
            wandb.log({"duration": t+1, "reward_episode": cumulated_reward, "episode": i_episode})
            # Take 100 episode averages and plot them too
            if len(cumulated_rewards_t) >= 50:
                means = cumulated_rewards_t.unfold(0, 50, 1).mean(1).view(-1)
                means = torch.cat((torch.zeros(49), means))
                wandb.log({"mean_cumulated_reward": means[i_episode].item(), "episode": i_episode})

                #  checkpoint save of the network for the best 100 episode average duration
                if (means[i_episode].item() > highest_cumulated_reward):
                    highest_cumulated_reward = means[i_episode].item()
                    torch.save(policy_net.state_dict(), SAVE_PATH)

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
