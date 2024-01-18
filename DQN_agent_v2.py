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
    def __init__(self,architecture,activation_functions,learning_rate=0.001,epsilon=0.1,epsilon_decay=0.995,epsilon_min=0.01,gamma=0.95,maxlen=1000):
       
        self.current_DronePoss_x = None
        self.current_DronePoss_y = None
        self.current_targetPoss_x = None
        self.current_targetPoss_y = None
        self.drone_pitch = None

        self.q_network = CustomNeuralNetwork(architecture, activation_functions)
        self.q_target_network = CustomNeuralNetwork(architecture, activation_functions)
        
        self.action_size = architecture[-1]

        self.RepMem = ReplayMemory(maxlen)

  
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma

        ## self.optimizer = tf.keras.optimizers.Adam(learning_rate=eta) TODO

        self.architecture = architecture
        self.activation_functions = activation_functions

    def update_target_network(self):
        self.q_target_network.layers = copy.deepcopy(self.q_network.layers)
    
    def decay_epsilon(self):
        self.epsilon = min(self.epsilon_min,self.epsilon*self.epsilon_decay)

    def e_greedy(self,state,custom_env):
        if np.random.rand() < self.epsilon: 
            return np.random.randint(custom_env.action_size)
    
       # q_values = self.q_network(np.array([state]))            # TODO, check here
        q_values = self.q_network.predict_single(state)
        return np.argmax(q_values)

    def learn(self,data,batch_size):

        self.RepMem.append(data)

        if len(self.RepMem) < batch_size: return 0
                
        # Sample a batch from the replay buffer
        minibatch = self.RepMem.get_batch(batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = minibatch
        

        # Q-values for the current state (s)
        q_values = self.q_network.predict(state_batch)
        
       # action_mask = np.eye(self.action_size)[np.arange(len(q_values)), np.argmax(q_values, axis=1)]  # One-hot encoding. The first array represents the row indices, and the second array represents the column indices. This way, you can achieve one-hot encoding without converting the Q-values to integers.
        #q_values = np.sum(q_values * action_mask, axis=1)

        # Q'-values for the next state
        nxt_q_vals = self.q_target_network.predict(next_state_batch)
        max_nxt_q_vals = np.max(nxt_q_vals, axis=1)


        # Calculate target Q-values using the Bellman equation
        target_q_vals = reward_batch + (1 - done_batch) * self.gamma * max_nxt_q_vals


        # Update Q-values based on the temporal difference error
        for i in range(len(state_batch)):
            target = target_q_vals[i]
            q_values[i, action_batch[i]] = target

        # Train the Q-network using the updated Q-values
        mse_list = self.q_network.train_on_batch(state_batch, q_values)
       # Train the Q-network using the updated Q-values
        #loss = np.mean(self.q_network.train_on_batch(state_batch, q_values))
        loss    = np.mean(np.square(target - q_values))

        #return np.mean(mse_list) 
        return loss



   


def main_callback(callback=None ):
    def play(agent,environment):
        state,_ = environment.reset()
        done = False
        rewards = 0
        
        while not done:
            pass
        #todo:

    def train(agent,env,num_episodes=100,batch_size=32,C=100):
        steps=0
        save_model_interval = 2
        for i in range(1,num_episodes+1):
            try:
                episode_reward = 0
                episode_loss = 0
                t = 0

                # Sample Phase
                agent.decay_epsilon()
                nxt_state = env.reset()
                done = False
                while not done:
                    state = nxt_state
                    action = agent.e_greedy(state,env)
                    nxt_state,reward,done,info = env.step(action)
                    time, (agent.current_DronePoss_x, agent.current_DronePoss_y), agent.drone_pitch, (agent.current_targetPoss_x, agent.current_targetPoss_y) = info
                
                    if callback:
                        callback((agent.current_DronePoss_x, agent.current_DronePoss_y), agent.drone_pitch, (agent.current_targetPoss_x, agent.current_targetPoss_y))

                    distance = custom_env.calculate_distance_to_target()
                    episode_reward += reward
                
                    # Learning Phase
                    episode_loss += agent.learn((state,action,reward,nxt_state,done),batch_size)
                    steps +=1
                    t+=1

                    if steps % C == 0: agent.update_target_network()               
                print(f"Episode: {i} Reward: {episode_reward} Loss: {episode_loss/t}, epsilon: {agent.epsilon}, time: {time}, distance: {distance}, Steps: {steps}")
            except KeyboardInterrupt:
                print(f"Training Terminated at Episode {i}")
                # Save the trained model (placeholder, customize as needed)
                if num_episodes % save_model_interval == 0:
                    agent.q_network.save_model(f"model_episode_{num_episodes}.h5")
                return 
        # Save the trained model (placeholder, customize as needed)
        if num_episodes % save_model_interval == 0:
            agent.q_network.save_model(f"model_episode_{num_episodes}.h5")


    custom_env = DroneEnvironment()
    arch = [6,6,5,5] # 6->6(sig)->5(relu)->5(lin)
    af = ["sigmoid","relu","linear"]
    agent = DQNController(arch,af,learning_rate=0.0005)
    train(agent,custom_env,num_episodes = 10000, batch_size=100)

if __name__ == "__main__":
    
    main_callback()


           

