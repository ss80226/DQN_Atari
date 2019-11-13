import torch
import torch.optim as optim
import random
import gym
from collections import namedtuple
from replay_buffer import ReplayBuffer
from dqn import DQN
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import math
import wandb
import time
wandb.init(project="dqn-atari")

PATH = './dqn_checkpoint'
GPU = torch.cuda.is_available()
DEVICE = 'cuda' if GPU else 'cpu'
env = gym.make('SpaceInvaders-ram-v0').unwrapped
experience_sample = namedtuple('experience_sample', ('state', 'action', 'reward', 'next_state'))

print(DEVICE)

# parameters

BATCH_SIZE = 128
INPUT_DIM = 128
ACTION_DIM = 4
BUFFER_SIZE = 1000000
POLICY_ARGS = {'input_dim': INPUT_DIM, 'action_dim': ACTION_DIM}
EPISODE = int(1e4) #number of game time the agent play
GAMMA = 0.9
TARGET_UPDATE_EPISODE = 10
EPS_START = 0.9
EPS_END = 0.1
EPS_DECAY = 10000
# we need two DQN network

policy = DQN(POLICY_ARGS).to(DEVICE)
# print(policy)
target = DQN(POLICY_ARGS).to(DEVICE)

policy_weight = policy.state_dict()
target.load_state_dict(policy_weight)
target.eval() # fixed the target net, we don't want to train it
mse = nn.MSELoss()
optimizer = optim.RMSprop(policy.parameters())
replay_buffer = ReplayBuffer(BUFFER_SIZE)

# training phase
total_game_step = 0
for current_episode in range(EPISODE):
    state = env.reset() # get the initial observation
    game_step = 0
    total_reward = 0
    state = torch.tensor([state]).float().to(DEVICE)
    while True:
        game_step += 1
        total_game_step += 1
        action = policy.act(state, total_game_step, isTrain = True).to(DEVICE) # sample an action
        next_state, reward, done, _ = env.step(action.item()) # take action in environment
        total_reward += reward
        reward = torch.tensor([reward]).float().to(DEVICE)
        
        if done: # whether this episode is terminate (game end)
            next_state = None
        else:
            next_state = torch.tensor([next_state]).float().to(DEVICE)
        
        replay_buffer.store(state, action, reward, next_state)
        state = next_state

        # optimze model with batch_size sample from buffer

        if replay_buffer.lenth() > BATCH_SIZE: # only optimize when replay buffer have sufficient number of data
            samples = replay_buffer.sample(BATCH_SIZE)
            samples = experience_sample(*zip(*samples))
            state_batch = torch.cat(samples.state)
            action_batch = torch.cat(samples.action)
            reward_batch = torch.cat(samples.reward)

            # get the Q-value Q(s(j), a(j))
            q_value_array = policy(state_batch) # get 4 value of all actions [V(a0), V(a1), V(a2), V(a3)]
            q_value = q_value_array.gather(1, action_batch)

            # set y(j) = r(j) --- if next_state(j+1) is terminal
            #            r(j) + r* Max(Q(S(j+1))) --- for non-terminal next_state(j+1)
            # Note : use Q-function of target_network

            terminal_mask = torch.tensor(tuple(map(lambda a: a is not None, samples.next_state)), device = DEVICE)
            next_state_batch = torch.cat([a for a in samples.next_state if a is not None]) # select the non-teminal next_state
            # initialize Q(S(j+1)) as 0
            q_value_next = torch.zeros(BATCH_SIZE).to(DEVICE)
            q_value_next[terminal_mask] = (torch.max(target(next_state_batch), 1)[0]).detach()
            # print('www')
            y = reward_batch + (q_value_next * GAMMA)
            y = y.unsqueeze(1)
            
            # loss
            # loss = F.smooth_l1_loss(q_value, y)
            loss = mse(q_value, y)
            #  optimize
            optimizer.zero_grad()
            loss.backward()
            for param in policy.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()
            # print('shit')
            wandb.log({'loss': loss.item()})
        
        if done:
            epsilon = EPS_END + (EPS_START - EPS_END) * math.exp((-1.) * total_game_step/EPS_DECAY)
            print('-------------------')
            print('episode: {episode}, game_step: {game_step}, total_reward: {total_reward}, epsilon: {epsilon}' \
            .format(episode=current_episode, game_step=game_step, total_reward=total_reward, epsilon=epsilon))
            wandb.log({'total_reward': total_reward})
            break
        
    if current_episode % TARGET_UPDATE_EPISODE == 0:
        # print(current_episode)
        target.load_state_dict(policy.state_dict())
torch.save(target.state_dict(), PATH)

# for episode in range(10):
#     state = env.reset()
#     game_step = 0
#     total_reward = 0
#     state = torch.FloatTensor([state]).to(DEVICE)
#     while True:
#         env.render()
#         time.sleep(0.05)
#         game_step += 1
#         action = policy.act(state, 1, isTrain=False).to(DEVICE)
#         # print(action)
#         print(action.item())
#         # i = game_step%4
#         next_state, reward, done, _ = env.step(action.item()) # take action in environment
#         total_reward += reward
#         reward = torch.FloatTensor([reward]).to(DEVICE)
#         if done:
#             print('-------------------')
#             print('episode: {episode}, game_step: {game_step}, total_reward: {total_reward}' \
#             .format(episode=episode, game_step=game_step, total_reward=total_reward))
#             # wandb.log({'total_reward': total_reward})
#             break
#         else:
#             next_state = torch.FloatTensor([next_state]).to(DEVICE)
# env.close()
# env.render()
env.close()     





