import matplotlib.pyplot as plt
import numpy as np

# Define the exponential decay function
def decay_function(steps, initial_value, decay_factor):
    return initial_value * (decay_factor ** steps)

# Parameters
initial_value = 0.9
decay_factor = 0.99995
total_steps = 100000

# Generate x values (steps)
steps = np.arange(0, total_steps + 1)

# Generate y values (epsilon values)
epsilon_values = decay_function(steps, initial_value, decay_factor)

# Plot the decay function
plt.plot(steps, epsilon_values, label='Exponential Decay')
plt.xlabel('Steps')
plt.ylabel('Epsilon Value')
plt.title('Exponential Decay: 0.9 * (0.999995)^{Steps}')
plt.axhline(y=0.1, color='r', linestyle='--', label='Target Value (0.1)')
plt.legend()
plt.show()