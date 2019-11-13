from collections import namedtuple
import random
experience_sample = namedtuple('experience_sample', ('state', 'action', 'reward', 'next_state'))
class ReplayBuffer(object):
    '''
    store and smaple experience data from environment for off-policy training
    experience sample shoud be in the form :  ('state', 'action', 'reward', 'next_state')
    '''
    def __init__(self, size):
        self.size = size #maximum size of the buffer
        self.curruent_index = 0
        self.buffer = []
        
    def store(self, *tuples):
        if len(self.buffer) < self.size:
            self.buffer.append(None)
        self.buffer[self.curruent_index] = experience_sample(*tuples)
        self.curruent_index += 1
        self.curruent_index = self.curruent_index % self.size # reset index if over the size
    
    def sample(self, sample_num):
        samples = random.sample(self.buffer, sample_num)
        return samples
    def lenth(self):
        return len(self.buffer)
        