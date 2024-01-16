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

class DQNController:
    def __init__(self):
        self.current_DronePoss_x = None
        self.current_DronePoss_y = None
        self.current_targetPoss_x = None
        self.current_targetPoss_y = None
        self.drone_pitch = None

    def play_DQN(self, callback=None):

        custom_env = DroneEnvironment()

        q_network = CustomNeuralNetwork(architecture=[custom_env.observation_space.shape[0], 6, custom_env.action_space.n],
                                        activation_functions=['relu', 'linear'],
                                        learning_rate=0.01,
                                        momentum=0.9,
                                        seed=True)
        q_target_network = CustomNeuralNetwork(architecture=[custom_env.observation_space.shape[0], 6, custom_env.action_space.n],
                                               activation_functions=['relu', 'linear'],
                                               learning_rate=0.01,
                                               momentum=0.9,
                                               seed=True)

        replay_memory = ReplayMemory(memlen=100)

        initial_epsilon = 0.1
        epsilon_min = 0.01
        epsilon_decay = 0.995

        gamma = 0.95
        batch_size = 32
        target_update_frequency = 100

        for episode in range(1000):
            state = custom_env.reset()
            done = False
            total_reward = 0
            steps = 0

            while not done:

                epsilon = max(initial_epsilon * epsilon_decay ** episode, epsilon_min)

                if np.random.rand() < epsilon:
                    action = np.random.randint(custom_env.action_size)
                else:
                    q_values = q_network.predict_single(state)
                    action = np.argmax(q_values)

                steps = steps + 1
                next_state, reward, done, info = custom_env.step(action)
                time, (self.current_DronePoss_x, self.current_DronePoss_y), self.drone_pitch, (self.current_targetPoss_x, self.current_targetPoss_y) = info
                
                 # Call the callback function on each iteration
                if callback:
                    callback((self.current_DronePoss_x, self.current_DronePoss_y), self.drone_pitch, (self.current_targetPoss_x, self.current_targetPoss_y))
                #print("Current position of the drone: ",self.current_DronePoss, "target: ",self.current_targetPoss)

                total_reward += reward
                distance = custom_env.calculate_distance_to_target()

                replay_memory.append((state, action, reward, next_state, done))

                if len(replay_memory) < batch_size:
                    continue

                minibatch = replay_memory.get_batch(batch_size)

                if minibatch[0].size == 0:
                    print("minibatch is empty")
                    continue

                state_batch, action_batch, reward_batch, next_state_batch, done_batch = minibatch

                for i in range(len(state_batch)):
                    s, a, r, ns, d = state_batch[i], action_batch[i], reward_batch[i], next_state_batch[i], done_batch[i]

                    target = r + (1 - d) * gamma * np.max(q_target_network.predict_single(ns))

                    q_values = q_network.predict_single(s)
                    q_values[a] = target
                    q_network.train_on_batch(np.array([s]), np.array([q_values]))

                if episode % target_update_frequency == 0:
                    q_target_network.layers = copy.deepcopy(q_network.layers)

                state = next_state

               

            epsilon = max(epsilon * epsilon_decay, epsilon_min)
           # print("END episode   Coordonates drone:", int(custom_env.drone.x), int(custom_env.drone.y), "Coordonates target:", custom_env.drone.target_coordinates[0], custom_env.drone.target_coordinates[1])

            print(f"Episode: {episode + 1}, Total Reward: {total_reward}, time in steps: {time}, epsilon: {epsilon}, distance: {distance}")

    def get_current_possition_DQN(self):
        return self.current_DronePoss
   

    def get_target_coordonates_DQN(self):
        return self.current_targetPoss




def main():
    dqn_controller = DQNController()
    dqn_controller.play_DQN()

if __name__ == "__main__":
    main()