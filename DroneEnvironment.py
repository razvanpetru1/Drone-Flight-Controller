import numpy as np
import gymnasium as gym
from gym import spaces
from drone import Drone
from math import sin, cos, pi, sqrt

class DroneEnvironment():
    def __init__(self):
        #def __init__(self, controller):
        # Define action and observation spaces
        # 5 actions: Nothing, Up, Down, Right, Left .New
        self.action_size = 5   
        self.action_space = gym.spaces.Discrete(5)

        # 6 observations: angle_to_up, velocity, angle_velocity, distance_to_target, angle_to_target, angle_target_and_velocity
        self.state_size = 6
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,))
        # self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        # Initialize the drone
        self.drone = Drone()  # Assuming Drone class is defined somewhere

        # Set initial state
        self.state = self.get_state()  # Replace '?' with a proper method call
    
        self.time_limit = 200 # TODO has to be checked!!!!

        


    def reset(self):
        # Reset the environment and return the initial state
        self.drone = Drone()
        self.state = self.get_state()
        return self.state


    def calculate_reward(self):
        # Definition of tje  reward
        distance_to_target = self.calculate_distance_to_target()
         # Reward per step survived
        self.drone.reward += 1 / 60
        # Penalty according to the distance to target
        self.drone.reward -= distance_to_target / (100 * 60)
        if distance_to_target < self.drone.game_target_size:
            # Reward if close to target
            self.drone.get_next_target()
            self.drone.reward += 100
        # If out of time
        if self.drone.t > self.time_limit:
            done = True
            
        # If too far from target (crash)
        elif distance_to_target > 1000:
            self.drone.reward -= 1000
            done = True
            
        else:
            done = False
        return self.reward, done

    def step(self, action):
            # step loop
        self.reward = 0.00
        self.drone.action_space = int(action)

        
        target_point = self.drone.get_next_target()
        # Initialize accelerations
        self.drone.acceleration_x = 0
        self.drone.acceleration_y = self.drone.g   # acceleration   # gravity
        self.drone.pitch_acceleration = 0
        thruster_left = self.drone.thrust_left  # thrust_left
        thruster_right = self.drone.thrust_right # thrust_right

        if action == 0: # stand still 
            thruster_left = self.drone.thrust_left  # thrust_left
            thruster_right = self.drone.thrust_right # thrust_right
        # if the thrust_L&R is < 0.5 we go down
        elif action == 1:   # going up
            thruster_left += self.drone.thruster_amplitude
            thruster_right += self.drone.thruster_amplitude
        elif action == 2:   # going down
            thruster_left -= self.drone.thruster_amplitude
            thruster_right -= self.drone.thruster_amplitude
        elif action == 3:   # going right
            thruster_left += self.drone.diff_amplitude
            thruster_right -= self.drone.diff_amplitude
        elif action == 4:   # going left
            thruster_left -= self.drone.diff_amplitude
            thruster_right += self.drone.diff_amplitude
        # Calculating accelerations with Newton's laws of motions
        
        # here we calculate the total thrust & torque 
        total_thrust = thruster_right + thruster_left
        total_torque = (thruster_left - thruster_right)*self.drone.turning_constant # momentum, torque
        # we calculate the x, y and theta accelation components 
        thrust_vec_x = np.sin(self.drone.pitch) # for doc is y 
        thrust_vec_y = np.cos(self.drone.pitch) # for doc is z
    
        acc_x_h = (total_thrust * thrust_vec_x) / self.drone.mass - self.drone.velocity_drag * self.drone.velocity_x              
        acc_y_h = (total_thrust * thrust_vec_y) / self.drone.mass + self.drone.g - self.drone.velocity_drag * self.drone.velocity_y
        theta_acc_h = (total_torque) / self.drone.mass - self.drone.pitch_drag_constant * np.abs(self.drone.pitch_velocity)

        # Drive the speed and the possition of the drone using small aproximations: cos(theta) = 1, and sin(theta) = theta
        # new component is = as old component + accelaration * time : Euler method
        delta_time = 0.01
        # First part of delta time
        vel_x_h = self.drone.velocity_x + acc_x_h * delta_time/2                      # new x velocity
        vel_y_h = self.drone.velocity_y + acc_y_h * delta_time/2                      # new y velocity
        theta_vel_h = self.drone.pitch_velocity + theta_acc_h * delta_time/ 2        # new theta velocity

        x_h = self.drone.x + vel_x_h * delta_time / 2                                 # new x possition
        y_h = self.drone.y + vel_y_h * delta_time / 2                                 # new y possition
        theta_h = self.drone.pitch + theta_vel_h * delta_time / 2                     # new  theta angle

        # we calculate the x, y and theta accelation components 
        thrust_vec_x = np.sin(theta_h)
        thrust_vec_y = np.cos(theta_h)
        velocity_size = np.sqrt(vel_x_h*vel_x_h+vel_y_h*vel_y_h)                # this is not used

        acc_x_f = (total_thrust * thrust_vec_x) / self.drone.mass - self.drone.velocity_drag * self.drone.velocity_x
        acc_y_f = (total_thrust * thrust_vec_y) / self.drone.mass + self.drone.g - self.drone.velocity_drag * self.drone.velocity_y
        theta_acc_f = (total_torque) / self.drone.mass - self.drone.pitch_drag_constant * np.abs(self.drone.pitch_velocity)

        # First part of delta time
        self.drone.velocity_x = vel_x_h + acc_x_f * delta_time/2                      # new x velocity
        self.drone.velocity_y = vel_y_h + acc_y_f * delta_time/2                      # new y velocity
        self.drone.pitch_velocity = theta_vel_h + theta_acc_f * delta_time / 2        # new theta velocity

        self.drone.x = x_h + self.drone.velocity_x * delta_time / 2                         # new x possition
        self.drone.y = y_h + self.drone.velocity_y * delta_time / 2                         # new y possition
        self.drone.pitch = theta_h + self.drone.pitch_velocity * delta_time / 2             # new theta angle

      
        reward, done = self.calculate_reward()
        info = {}
        return (
            self.get_state(), # current state 
            self.reward,    # reward
            done,
            info,
            #(thruster_left,thruster_right)   
        )

    def calculate_distance_to_target(self):
         # Updates the target list
        target_point = self.drone.get_next_target()
        # Calculate the Euclidean distance to the target
        distance_x = self.drone.x - target_point[0]
        distance_y = self.drone.y - target_point[1]
        distance_to_target = np.sqrt(distance_x*distance_x+distance_y*distance_y)
        return distance_to_target

    def calculate_angle_to_target(self):
        # Calculate the angle between the drone's current orientation and the direction to the target
        target_point = self.drone.get_next_target()
        angle_to_target = np.arctan2(target_point[1] - self.drone.y, target_point[0] - self.drone.x)
        return angle_to_target - self.drone.pitch
    
    def calculate_angle_target_and_velocity(self):
        # Calculate the angle between the direction to the target and the drone's velocity
        
        angle_to_target = self.calculate_angle_to_target()
        angle_target_and_velocity = angle_to_target - np.arctan2(self.drone.velocity_y, self.drone.velocity_x)
        return angle_target_and_velocity
    
    def get_state(self):
        """
        Calculates the observations
        Returns:
            np.ndarray: The normalized observations:
            -angle_to_up: The angle between the drone and the upward vector, used to observe gravity.
            -velocity: The speed of the drone.
            -angle_velocity: The angle of the velocity vector.
            -distance_to_target: The distance from the drone to the target.
            -angle_to_target: The angle between the drone's orientation and the vector pointing towards the target.
            -angle_target_and_velocity: The angle between the to_target vector and the velocity vector.
        """
        """Calculates the drone states from the drone variable."""
        angle_to_up = self.drone.pitch / 180 * pi  
        velocity = np.sqrt(self.drone.velocity_x**2 + self.drone.velocity_y**2)
        angle_velocity = self.drone.pitch_velocity
        distance_to_target = self.calculate_distance_to_target()
        angle_to_target = self.calculate_angle_to_target()
        angle_target_and_velocity = self.calculate_angle_target_and_velocity()
        return np.array([angle_to_up, 
                         velocity, 
                         angle_velocity, 
                         distance_to_target, 
                         angle_to_target, 
                         angle_target_and_velocity])
