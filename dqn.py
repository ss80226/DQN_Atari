import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
EPS_START = 0.9
EPS_END = 0.1
EPS_DECAY = 200
class DQN(nn.Module):
    def __init__ (self, args): 
        '''
        initialize the q-network
        input:
            - input_dim: state dimension
            - action_dim: action dimension
            - 
        network: 3 layer FC
        '''
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(args['input_dim'], 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, args['action_dim'])
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def act(self, state, steps_done, isTrain):
        sample_prob = random.random() # probability that sample actions from policy
        epsilon = EPS_END + (EPS_START - EPS_END) * math.exp((-1.) * steps_done/EPS_DECAY)
        # if steps_done > 1000000:
        #     epsilon = EPS_END
        if isTrain == False:
            with torch.no_grad():
                action_index = torch.max(self.forward(state), 1)[1].view(1, 1)
                return action_index
        elif sample_prob > epsilon: # take policy's action
            with torch.no_grad():
                # print('aw')
                # print(self.forward(state))
                # print(torch.max(self.forward(state), 1))
                action_index = torch.max(self.forward(state), 1)[1].view(1, 1)
                return action_index
        else: # take a random action
            random_action = torch.tensor([[random.randrange(4)]], dtype = torch.long)
            return random_action

