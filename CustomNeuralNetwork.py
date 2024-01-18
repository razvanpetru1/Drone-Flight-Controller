import numpy as np
import os
import pickle

class sigmoid:
    def __init__(self,beta=1):
        self.beta = beta
    
    def __call__(self,z,derivative=False):
    
        if derivative: return self.beta*z * (1 - z)
        return 1 / (1 + np.exp(-z*self.beta))

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
    "sigmoid":sigmoid(1),
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
            print(f"Episode: {episode} Loss: {mse/t}")
        
        
        # Save the trained model (placeholder, customize as needed)
        if episode % save_model_interval == 0:
            self.save_model(f"model_episode_{episode}.h5")

        # Monitor additional metrics (placeholder, customize as needed)
        if episode % metric_monitoring_interval == 0:
            print(f"episode: {episode}, Mean Squared Error: {mse}")
        
        
        # Returns:    List of mean squared errors at specified intervals during training.
        return mse_list

    def save_model(self, filename, directory="."):
            # Save the trained model using pickle
            filepath = os.path.join(directory, filename)
            with open(filepath, 'wb') as file:
                pickle.dump(self.layers, file)
            print(f"Model saved to {filepath}")
        

    def train_on_batch(self, X, y, number_episodes=100, error_calculation_interval=10):
            # Train the neural network using mini-batch gradient descent
        batch_size = len(X)
        mse_list = []
        record = False
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
        return self.forward(x)[-1]  # The last element of this list

    def predict(self, X):
        # For each input 'x' in the array 'X', calculate the output of the neural network
        # and extract the last element, which corresponds to the prediction.
        predictions = [self.forward(x)[-1] for x in X]

        # Convert the list of predictions into a NumPy array and return it.
        return np.array(predictions)


    # test 

if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.metrics import accuracy_score
    import matplotlib.pyplot as plt 

    bc = load_breast_cancer()
    np.random.seed(69420)
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(bc.data,bc.target,test_size=0.3,shuffle=True)
    
    s01 = sigmoid(0.01) # lower beta to avoid overflows in exp
    model_bc = CustomNeuralNetwork([30,12,1],[s01,"linear","sigmoid"],seed=8,learning_rate=0.1,momentum=0.3)
    
   

    y_pred = model_bc.predict(X_test)
    y_pred = [np.round(y_) for y_ in y_pred]
    mse_l = model_bc.train(X_train,y_train,number_episodes=100)
    fig = plt.figure(figsize=(5,5))
    plt.plot(range(len(mse_l)),mse_l)
    plt.xlabel(f"Episodes:")
    plt.ylabel("MSE")
    plt.show()
    y_pred = model_bc.predict(X_test)
    y_pred = [np.round(y_) for y_ in y_pred]
    print(f"Accuracy of the model: {accuracy_score(y_test,y_pred)}")