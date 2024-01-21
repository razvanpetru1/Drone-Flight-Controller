import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
file_path = 'episode_data_17.18.csv'
df = pd.read_csv(file_path, sep=',')
# Trim whitespaces from column names

df.columns = df.columns.str.strip()

# Check if 'Epsilon' is in the columns
if 'Epsilon' not in df.columns:
    print("Error: 'Epsilon' column not found in the DataFrame.")
else:
    # Extract relevant columns
    episode = df['Episode']
    epsilon = df['Epsilon']
    loss = df['Loss']
    steps = df['Steps']
    #steps = df['Reward']

    plt.plot(steps, loss, label='Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss Value')
    plt.title('Steps vs Loss')
    plt.legend()
    plt.show()

# Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plotting on the first subplot
    ax1.plot(episode, loss, label='Loss')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Loss Value')
    ax1.set_title('Episode vs Loss')
    ax1.legend()

    # Plotting on the second subplot
    ax2.plot(steps, epsilon, label='episode')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Epsilon Value')
    ax2.set_title('Steps vs Epsilon')
    ax2.legend()

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the plot
    plt.show()