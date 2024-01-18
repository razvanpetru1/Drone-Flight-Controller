import numpy as np
import random
from collections import deque


class ReplayMemory:
    def __init__(self, memlen):
        self.memory = deque(maxlen=memlen)

    def append(self, data):
        self.memory.append(data)

    def __len__(self):
        return len(self.memory)

    def get_batch(self, batch_size):

        minibatch = random.sample(self.memory, batch_size)

        # Check if minibatch is not empty
        if not minibatch:
            print("minibatch is empty!**************")
            
        state_batch = np.array([sample[0] for sample in minibatch])
        action_batch = np.array([sample[1] for sample in minibatch])
        reward_batch = np.array([sample[2] for sample in minibatch])
        next_state_batch = np.array([sample[3] for sample in minibatch])
        done_batch = np.array([sample[4] for sample in minibatch])

        # Return a tuple instead of individual arrays
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch
