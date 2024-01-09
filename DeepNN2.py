import numpy as np
import copy
from collections import deque
import random
from DroneEnvironment import DroneEnvironment


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

class ReLUActivation:
    def __call__(self, z, derivative=False):
        # ReLU activation function or its derivative if specified
        if derivative:
            return (z > 0) * 1
        return np.where(z > 0, z, 0)

class LinearActivation:
    def __call__(self, z, derivative=False):
        # Linear activation function or its derivative if specified
        if derivative:
            return 1
        return z

AF_MAP = {
    "linear": LinearActivation(),
    "relu": ReLUActivation(),
}

def calculate_neuron_sum(x, W, b):
    # Calculate the weighted sum of neuron inputs
    return W @ x + b

def mean_squared_error(t, y):
    # Calculate mean squared error between target and predicted values
    return np.sum((y - t) ** 2) / 2

class CustomNeuralNetwork:
    def __init__(self, architecture, activation_functions, learning_rate=0.1, momentum=0.9, seed=True):
        """
        Initialization method for the CustomNeuralNetwork class.
         Parameters:
        - architecture: List specifying the number of neurons in each layer.
        - activation_functions: List specifying the activation function for each layer.
        - learning_rate: Learning rate for gradient descent optimization. Default is 0.1.
        - momentum: Momentum for gradient descent optimization. Default is 0.9.
        - seed: Seed for random initialization. Default is True.         """

        if seed:
            np.random.seed(seed)
        # Neural network initialization with random weights and biases
        self.architecture = architecture
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.layers = [(
                0.01 * np.random.randn(architecture[i], architecture[i - 1]),
                0.01 * np.random.rand(architecture[i])
            ) for i in range(1, len(architecture))]
        
        self.num_layers = len(self.layers)
        # Variables for gradient descent Momentum
        self.prev_deltas = [0] * self.num_layers
        self.prev_dE_dWs = [0] * self.num_layers
        
        # Activation functions for each layer
        self.activation_functions = [AF_MAP[af] if isinstance(af, str) else af for af in activation_functions]
        
    def forward(self, x):
        # Forward pass through the neural network
        activations = [x]
        i = 0
        for w, b in self.layers:
            activations.append(
                self.activation_functions[i](
                    calculate_neuron_sum(activations[-1], w, b)
                )
            )
            i += 1
            
        return activations
    
    def backward(self, activations, target):
        # Backward pass to calculate gradients
        layer_count = self.num_layers
        deltas = [np.zeros(w.shape) for w, _ in self.layers]
        
        gradients_wrt_weights = [np.zeros(w.shape) for w, _ in self.layers]

        layer_count -= 1
        for i in range(layer_count, -1, -1):
            if i != layer_count:
                deltas[i] = (self.layers[i + 1][0].T @ deltas[i + 1]) * self.activation_functions[i](activations[i + 1], True)
            else:
                deltas[i] = (activations[i + 1] - target) * self.activation_functions[i](activations[i + 1], True)

            gradients_wrt_weights[i] = (activations[i].reshape(-1, 1) * deltas[i]).T
        #     Returns:    Tuple of deltas and gradients with respect to weights.
        return deltas, gradients_wrt_weights
    
    def update_weights(self, deltas, gradients_wrt_weights):
        # Gradient clipping
        clip_value = 1.0  # Adjust as needed
        gradients_wrt_weights = [np.clip(grad, -clip_value, clip_value) for grad in gradients_wrt_weights]

        # Update weights and biases using gradient descent with Momentum
        for i in range(len(self.layers)):
            W, b = self.layers[i]
            self.prev_dE_dWs[i] = self.learning_rate * gradients_wrt_weights[i] + self.momentum * self.prev_dE_dWs[i]
            self.prev_deltas[i] = self.learning_rate * deltas[i] + self.momentum * self.prev_deltas[i]
            W -= self.prev_dE_dWs[i]
            b -= self.prev_deltas[i]

    def train(self, X, y, number_episodes=50, error_calculation_interval=10):
        # Train the neural network using batch gradient descent
        mse_list = []
        record = True
        metric_monitoring_interval = 10
        save_model_interval = 50
        mse = 0
        for episode in range(1, number_episodes + 1):
            if episode % error_calculation_interval == 0:
                record = True
                mse = 0

            for x, t in zip(X, y):
                activations = self.forward(x)
                deltas, gradients_wrt_weights = self.backward(activations, t)
                self.update_weights(deltas, gradients_wrt_weights)
                if record:
                    mse += mean_squared_error(activations[-1], t)

            if record:
                mse_list.append(mse)
                record = False
        """
        # Save the trained model (placeholder, customize as needed)
        if episode % save_model_interval == 0:
            self.save_model(f"model_episode_{episode}.h5")

        # Monitor additional metrics (placeholder, customize as needed)
        if episode % metric_monitoring_interval == 0:
            print(f"episode: {episode}, Mean Squared Error: {mse}")
        """
        
        # Returns:    List of mean squared errors at specified intervals during training.
        return mse_list

    def save_model(self, filename):
            # Placeholder for saving the model
        # Implement the actual saving mechanism (e.g., using pickle, h5py, etc.)
        print(f"Saving model to {filename}")
        

    def train_on_batch(self, X, y, number_episodes=100, error_calculation_interval=10):
            # Train the neural network using mini-batch gradient descent
        batch_size = len(X)
        mse_list = []
        record = True
        mse = 0
        for episode in range(1, number_episodes + 1):
            if episode % error_calculation_interval == 0:
                record = True
                mse = 0
            delta_b = [0] * self.num_layers
            dE_dW_b = [0] * self.num_layers

            for x, t in zip(X, y):
                activations = self.forward(x)

                # Move the print statement here after the calculation of activations
                #print(f"Activations: {activations}")

                deltas, gradients_wrt_weights = self.backward(activations, t)

                #print(f"Deltas: {deltas}")
                #print(f"Gradients: {gradients_wrt_weights}")

                for i in range(self.num_layers):
                    delta_b[i] += deltas[i]
                    dE_dW_b[i] += gradients_wrt_weights[i]
                if record:
                    mse += mean_squared_error(activations[-1], t)

            for i in range(self.num_layers):
                delta_b[i] /= batch_size
                dE_dW_b[i] /= batch_size
            self.update_weights(delta_b, dE_dW_b)

            if record:
                mse_list.append(mse)
                record = False
        # Returns:   List of mean squared errors at specified intervals during training.
        return mse_list

    def predict_single(self, x):
        return self.forward(x)[-1]  # he last element of this list

    def predict(self, X):
        # For each input 'x' in the array 'X', calculate the output of the neural network
        # and extract the last element, which corresponds to the prediction.
        predictions = [self.forward(x)[-1] for x in X]

        # Convert the list of predictions into a NumPy array and return it.
        return np.array(predictions)

# Custom code to test the provided Neural Network implementation

# Custom dataset generation for binary classification
def generate_dataset(num_samples=100000, input_size=5, random_seed=True):
    if random_seed:
        np.random.seed(random_seed)

    X = np.random.rand(num_samples, input_size)
    y = np.random.randint(0, 2, size=num_samples)

    return X, y

class ReplayMemory:
    def __init__(self, memlen):
        self.memory = deque(maxlen=memlen)

    def append(self, data):
        self.memory.append(data)

    def __len__(self):
        return len(self.memory)

    def get_batch(self, batch_size):
        if len(self.memory) < batch_size:
            # Return a smaller batch if the memory doesn't have enough samples
            batch_size = len(self.memory)

        minibatch = random.sample(self.memory, batch_size)

        # Check if minibatch is not empty
        if not minibatch:
            print("minibatch is empty!")
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

        state_batch = np.array([sample[0] for sample in minibatch])
        action_batch = np.array([sample[1] for sample in minibatch])
        reward_batch = np.array([sample[2] for sample in minibatch])
        next_state_batch = np.array([sample[3] for sample in minibatch])
        done_batch = np.array([sample[4] for sample in minibatch])

        # Return a tuple instead of individual arrays
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch


# Test the Neural Network on the custom dataset
# Create a CustomEnvironment
#custom_env = CustomEnvironment(state_size=4, action_size=2)

# Q-network architecture and activation functions
# ReLU for the hidden layer. Introduces non-linearity to the model.
# Linear for the output layer.
#q_network = CustomNeuralNetwork(architecture=[custom_env.state_size, 8, custom_env.action_size], activation_functions=['relu', 'linear'], learning_rate=0.001, momentum=0.9, seed=42)


# Q-target network (initialized with the same weights)
#q_target_network = copy.deepcopy(q_network)

#custom_controller = CustomController()
#custom_env = DroneEnvironment(controller=custom_controller)

custom_env = DroneEnvironment()
q_network = CustomNeuralNetwork(architecture=[custom_env.observation_space.shape[0], 8, custom_env.action_space.n],
                                activation_functions=['relu', 'linear'],
                                learning_rate=0.001,
                                momentum=0.9,
                                seed=42)
q_target_network = CustomNeuralNetwork(architecture=[custom_env.observation_space.shape[0], 8, custom_env.action_space.n],
                                       activation_functions=['relu', 'linear'],
                                       learning_rate=0.001,
                                       momentum=0.9,
                                       seed=42)

# Replay Memory
replay_memory = ReplayMemory(memlen=100)


# Exploration schedule parameters
initial_epsilon = 0.1
epsilon_min = 0.01
epsilon_decay = 0.995
# Other hyperparameters
gamma = 0.95
batch_size = 32
target_update_frequency = 100  # Update target network every 100 steps
#i = 0

# Training loop
for episode in range(10):
    state = custom_env.reset()
    done = False
    total_reward = 0

    while not done:
        # Exploration schedule
        epsilon = max(initial_epsilon * epsilon_decay ** episode, epsilon_min)
        #print(i+1)
        
        # Epsilon-greedy policy
        if np.random.rand() < epsilon:
            action = np.random.randint(custom_env.action_size)
        else:
            q_values = q_network.predict_single(state)
            action = np.argmax(q_values)

        next_state, reward, done, _ = custom_env.step(action)
        total_reward += reward

        print("Training loop append: "," state:", state,"  action:" , action," reward", reward, " next state",next_state, done)
        # Store experience in replay memory
        replay_memory.append((state, action, reward, next_state, done))
        if len(replay_memory)<batch_size: continue
        # Sample a random minibatch from replay memory

        minibatch = replay_memory.get_batch(batch_size)

        # Check if minibatch is not empty
        if minibatch[0].size == 0:
            print("minibatch is empty")
            continue

        # Update Q-network
        #print(minibatch)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = minibatch

        for i in range(len(state_batch)):
            s, a, r, ns, d = state_batch[i], action_batch[i], reward_batch[i], next_state_batch[i], done_batch[i]
            target = r + (1 - d) * gamma * np.max(q_target_network.predict_single(ns))
            q_values = q_network.predict_single(s)
            q_values[a] = target
            q_network.train_on_batch(np.array([s]), np.array([q_values]))

        # Update target Q-network periodically
        if episode % target_update_frequency == 0:
            q_target_network.layers = copy.deepcopy(q_network.layers)

        state = next_state

    # Decay epsilon
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

#  we can use the trained Q-network for making decisions in the environment after the training


