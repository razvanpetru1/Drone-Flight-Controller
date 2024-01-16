import numpy as np
class CustomEnvironment:
    def __init__(self, state_size, action_size):
        
        # Initialize the environment with the given state and action sizes
        self.state_size = state_size
        self.action_size = action_size
        self.state = np.zeros(state_size)
        self.done = False

    def reset(self):
        # Reset the state to zeros and mark the environment as not done
        self.state = np.zeros(self.state_size)
        self.done = False
        return self.state

    def step(self, action):
        # Custom transition dynamics
        self.state += action
        reward = np.sum(self.state)  # Reward is the sum of the new state
        self.done = np.all(self.state > 5)  # Terminate if the sum exceeds 5

        # Return the new state, reward, termination flag, and info
        return self.state, reward, self.done, {}