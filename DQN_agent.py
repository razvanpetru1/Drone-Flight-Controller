import numpy as np
import copy

from DroneEnvironment import DroneEnvironment
from CustomNeuralNetwork import CustomNeuralNetwork
from ReplayMemory import ReplayMemory

# for env test: 
#from CustomEnvironment import CustomEnvironment


# Custom code to test CustomNeuralNetwork implementation


# Custom dataset generation for binary classification
def generate_dataset(num_samples=100000, input_size=5, random_seed=True):
    if random_seed:
        np.random.seed(random_seed)

    X = np.random.rand(num_samples, input_size)
    y = np.random.randint(0, 2, size=num_samples)

    return X, y


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

# Input layer: Neurons equal to the shape of the observation space 6.
# Hidden layer: 6 neurons.
# Output layer: Neurons equal to the size of the action space (size custom_env.action_space.n).
q_network = CustomNeuralNetwork(architecture =[custom_env.observation_space.shape[0], 6, custom_env.action_space.n],
                                activation_functions =['relu', 'linear'],
                                learning_rate = 0.01,
                                momentum = 0.9,
                                seed = True)
q_target_network = CustomNeuralNetwork(architecture =[custom_env.observation_space.shape[0], 6, custom_env.action_space.n],
                                       activation_functions =['relu', 'linear'],
                                       learning_rate = 0.01,
                                       momentum = 0.9,
                                       seed = True)

# Replay Memory
replay_memory = ReplayMemory(memlen=100)


# Exploration schedule parameters
initial_epsilon = 0.1
epsilon_min = 0.001
epsilon_decay = 0.995

# Other hyperparameters
gamma = 0.95
batch_size = 32
target_update_frequency = 100  # Update target network every 100 steps
#i = 0

# Training loop
for episode in range(1000):
    state = custom_env.reset()
    done = False
    total_reward = 0
    steps = 0
    while not done:
        # Exploration schedule
        epsilon = max(initial_epsilon * epsilon_decay ** episode, epsilon_min)
        #print(i+1)
        
        # Epsilon-greedy policy
        if np.random.rand() < epsilon:
            action = np.random.randint(custom_env.action_size)
        else:
            q_values = q_network.predict_single(state)
            
            action = np.argmax(q_values) # action is the index of the the actual maximum value in q_values
        steps = steps + 1
        """
        if action != 0:
            print("Step:", steps)
            print("q_values:", q_values)
            print("Index of maximum value:", action)
            print("Maximum value:", q_values[action])
        else:
            print(" 0 Step:", steps)
        """
        next_state, reward, done, _ , time = custom_env.step(action)
        total_reward += reward
        distance = custom_env.calculate_distance_to_target()
        #print('total reward: ',total_reward)
        #print('distance to target:', custom_env.calculate_distance_to_target())

        #print("Training loop append: "," state:", state,"  action:" , action," reward", reward, " next state",next_state, done)
        # Store experience in replay memory
        replay_memory.append((state, action, reward, next_state, done))
        if len(replay_memory)<batch_size: continue
        # Sample a random minibatch from replay memory

        minibatch = replay_memory.get_batch(batch_size)


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
            #mse_episod = mean_squared_error(target,np.array([q_values] )

         
    

        # Update target Q-network periodically
        if episode % target_update_frequency == 0:
            q_target_network.layers = copy.deepcopy(q_network.layers)

        state = next_state

    # Decay epsilon
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    print(f"Episode: {episode + 1}, Total Reward: {total_reward}, time in steps: {time}, epsilon: {epsilon}, distance: {distance}")

#  we can use the trained Q-network for making decisions in the environment after the training


