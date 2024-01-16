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
    
        self.time_limit = 20 # TODO has to be checked!!!!

        


    def reset(self):
        # Reset the accelaration variable
        self.drone.acceleration_x = 0
        self.drone.acceleration_y = 0
        self.drone.pitch_acceleration = 0

        # Reset the velocity variable
        self.drone.velocity_x = 0       # new x velocity
        self.drone.velocity_y = 0       # new y velocity
        self.drone.pitch_velocity = 0   # new theta velocity

        # Reset the possition variable
        self.drone.x = 200                # new x possition
        self.drone.y = 200                # new y possition
        self.drone.pitch = 0            # new  theta angle

        self.drone.target_coordinates = self.drone.get_next_target()
        
        self.drone.reward = 0
        self.drone.t = 0

        self.state = self.get_state()
        return self.state


    def step(self, action):
            # step loop
        self.drone.reward = 0.00
        self.drone.action_space = int(action)

        # Act every 5 frames
        for _ in range(5):
        
            self.drone.t += 1 / self.drone.FPS

            # Initialize accelerations
            self.drone.acceleration_x = 0              # 
            self.drone.acceleration_y = self.drone.g   # acceleration   # gravity
            self.drone.pitch_acceleration = 0

            # TODO: set the thrust to 0.5 stay still before we calculate what action we should take. 

            thruster_left = self.drone.thrust_left  # thrust_left
            thruster_right = self.drone.thrust_right # thrust_right

            # Take an action, calculate the thrust to apply to the motors

            if action == 0: # stand still 

                thruster_left = self.drone.thrust_left  # thrust_left
                thruster_right = self.drone.thrust_right # thrust_right

            elif action == 1:   # going up                # Thrust_L&R is > 0.5 we go up
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
            
            # Step 1: we calculate the total thrust & torque 
            total_thrust = thruster_right + thruster_left
            total_torque = (thruster_left - thruster_right)*self.drone.turning_constant # momentum, torque

            # Step 2: calculate acceleration x y and theta
            
            thrust_vec_x = np.sin(self.drone.get_pitch_rad()) # for doc is y 
            thrust_vec_y = np.cos(self.drone.get_pitch_rad()) # for doc is z
    
            self.drone.acceleration_x += (total_thrust * thrust_vec_x) / self.drone.mass 
            self.drone.acceleration_y += (total_thrust * thrust_vec_y) / self.drone.mass 
            self.drone.pitch_acceleration += self.drone.length_arm_drone *  (total_torque) / self.drone.mass 

            # Drive the speed and the possition of the drone using small aproximations: cos(theta) = 1, and sin(theta) = theta

            # new component is = as old component + accelaration * time : Euler method

            # Calculate velocity a y and theta
            self.drone.velocity_x += self.drone.acceleration_x                      # new x velocity
            self.drone.velocity_y += self.drone.acceleration_y                      # new y velocity
            self.drone.pitch_velocity += self.drone.pitch_acceleration              # new theta velocity

            # Calculate possition a y and angle theta
            self.drone.x += self.drone.velocity_x                                   # new x possition
            self.drone.y += self.drone.velocity_y                                   # new y possition
            self.drone.pitch += self.drone.pitch_velocity                           # new  theta angle

                
            self.drone.reward, done = self.calculate_reward()
        info = {}

        return (
            self.get_state(), # current state 
            self.drone.reward,    # reward
            done,
            info,
            self.drone.t
            #(thruster_left,thruster_right)   
        )

    def calculate_reward(self):
        # Definition of tje  reward
        distance_to_target = self.calculate_distance_to_target()

         # Reward per step survived 
        self.drone.reward += 1 / self.drone.FPS

        # Penalty according to the distance to target
        self.drone.reward -= distance_to_target / (100 * 60)

        if distance_to_target < self.drone.game_target_size:
            # The drone is close to target so we generate a new target
            self.drone.get_next_target()
            # Reward if close to target
            self.drone.reward += 100
        # If out of time
        if self.drone.t > self.time_limit:
            done = True
            
        # If the drone exceeds a certain distance from the target, it results in a crash - reward a big penalty
        elif distance_to_target > 1000:
            self.drone.reward -= 1000
            done = True
            
        else:
            done = False
        return self.drone.reward, done

    def calculate_distance_to_target(self):
         # Updates the target list
        target_point = self.drone.target_coordinates 
        # Calculate the Euclidean distance to the target
        distance_x = self.drone.x - target_point[0]
        distance_y = self.drone.y - target_point[1]
        distance_to_target = np.sqrt(distance_x*distance_x + distance_y*distance_y)

        return distance_to_target

    def calculate_angle_to_target(self):
        # Calculate the angle between the drone's current orientation and the direction to the target
        target_point = self.drone.target_coordinates
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
