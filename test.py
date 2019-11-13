import gym
import torch
from dqn import DQN
import time

BATCH_SIZE = 128
INPUT_DIM = 128
ACTION_DIM = 4
BUFFER_SIZE = 1000000
POLICY_ARGS = {'input_dim': INPUT_DIM, 'action_dim': ACTION_DIM}
GPU = torch.cuda.is_available()
DEVICE = 'cuda' if GPU else 'cpu'
PATH = './dqn_checkpoint'

policy = DQN(POLICY_ARGS).to(DEVICE)
policy.load_state_dict(torch.load(PATH))
policy.eval()
env = gym.make('SpaceInvaders-ram-v0').unwrapped
print('play 10 episode')

for episode in range(10):
    state = env.reset()
    game_step = 0
    total_reward = 0
    state = torch.FloatTensor([state]).to(DEVICE)
    while True:
        env.render()
        time.sleep(0.05)
        game_step += 1
        action = policy.act(state, 1, isTrain=False).to(DEVICE)
        # print(action)
        # print(state.squeeze()[:20])
        # print(action.item())
        # i = game_step%4
        next_state, reward, done, _ = env.step(
            action.item())  # take action in environment
        total_reward += reward
        reward = torch.FloatTensor([reward]).to(DEVICE)
        if done:
            print('--------------------')
            print('episode: {episode}, game_step: {game_step}, total_reward: {total_reward}' \
            .format(episode=episode, game_step=game_step, total_reward=total_reward))
            # wandb.log({'total_reward': total_reward})
            break
        else:
            state = torch.FloatTensor([next_state]).to(DEVICE)

env.close()
