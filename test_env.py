from DroneEnvironment import DroneEnvironment

env = DroneEnvironment()
obs = env.reset()


print("Observation space:")
print(env.observation_space)
print("")
print("Action space:")
print(env.action_space)
print("")
print("Action space sample:")
print(env.action_space.sample())

# Choose an action to execute n_steps times
action = 2
n_steps = 500
for step in range(n_steps):
    print("Step {}".format(step + 1))
    obs, reward, done, info = env.step(action)
    print("obs=", obs, "reward=", reward, "done=", done)
    
    if done:
        print("Done!", "reward=", reward)
        break