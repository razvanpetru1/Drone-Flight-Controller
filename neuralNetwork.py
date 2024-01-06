import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize the neural network with random weights and biases
        np.random.seed(1)
        self.weights_input_hidden = 2 * np.random.random((input_size, hidden_size)) - 1
        self.bias_input_hidden = np.zeros((1, hidden_size))
        np.random.seed(1)
        self.weights_hidden_output = 2 * np.random.random((hidden_size, output_size)) - 1
        self.bias_hidden_output = np.zeros((1, output_size))

    def forward(self, input_data):
        # Forward propagation through the network
        hidden_layer_output = self.relu(np.dot(input_data, self.weights_input_hidden) + self.bias_input_hidden)
        output_layer_output = np.dot(hidden_layer_output, self.weights_hidden_output) + self.bias_hidden_output
        return output_layer_output, hidden_layer_output

    def relu(self, x):
        # Rectified Linear Unit (ReLU) activation function
        return np.maximum(0, x)

    def backward(self, input_data, target, output, hidden_output, learning_rate):
        # Backward propagation to update weights and biases based on the error
        output_error = target - output
        output_delta = output_error
        self.weights_hidden_output += learning_rate * hidden_output.T.dot(output_delta)
        self.bias_hidden_output += learning_rate * np.sum(output_delta, axis=0, keepdims=True)
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * (hidden_output > 0)
        self.weights_input_hidden += learning_rate * input_data.T.dot(hidden_delta)
        self.bias_input_hidden += learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)

    def train(self, input_data, target, epochs, learning_rate):
        # Training loop
        for epoch in range(epochs):
            output, hidden_output = self.forward(input_data)
            self.backward(input_data, target, output, hidden_output, learning_rate)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss {np.mean(np.square(target - output))}")

# Example usage:
input_size = 3
hidden_size = 6  # Increased hidden layer size
output_size = 4

X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

output_y = np.array([[1,1,1,1],
              [0,1,1,1],
              [1,1,0,1],
              [0,1,1,1]])

q_network = NeuralNetwork(input_size, hidden_size, output_size)
input_data = X

# Training loop
q_network.train(input_data, output_y, epochs=10000, learning_rate=0.1)

# After training, get the output
output, _ = q_network.forward(input_data)
print("Neural Network Output after Training:")
print(output)
