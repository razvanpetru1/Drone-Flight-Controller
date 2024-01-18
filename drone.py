import numpy as np
import gym
from typing import Tuple
from gym import spaces
from random import uniform
from math import pi

class Drone():


    def __init__(self):
        # Initialize variables
        self.x = 0
        self.y = 0
        self.pitch = 0
        
        self.velocity_y = 0
        self.velocity_x = 0
        self.pitch_velocity = 0
        
        self.acceleration_x = 0
        self.acceleration_y = 0
        self.pitch_acceleration = 0
        
        # Command for the motors if thrust_left & _right is 0.5 the drone is standing still.
        self.thrust_left = 0.5
        self.thrust_right = 0.5 
        
        self.thruster_amplitude = 0.000005
        self.diff_amplitude = 0.000006              
        # The target x,y coordinates the drone is trying to reach
          
        
        # self.target_coordinates = np.random.randint(480, 721, size=(2,))
        
        # Physics constants
        self.velocity_drag = 1.0
        self.pitch_drag_constant = 0.3
        self.target_coordinates = [] 
        self.max_thrust = 1.0
        self.turning_constant = 1.0
        self.mass = 1.0
        self.g = -1.0/self.mass         # gravity
	        
        self.length_arm_drone = 1      # we try length of 1
              
        self.t = 0                      # the time of the simulation (reseted at the start of every new episod)
        self.FPS = 60                   # frames per second
        self.game_target_size = 0.1     # size of the target on the map
        self.has_reached_target_last_update = False
        self.target_counter = 0         # we want to collect the number of targets collected. 
        
        # 5 actions: Nothing, Up, Down, Right, Left .New
        self.action_space = gym.spaces.Discrete(5)
        
        # 6 observations: angle_to_up, velocity, angle_velocity, distance_to_target, angle_to_target, angle_target_and_velocity
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,))
        # self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        #print("\n",type(self.observation_space))
        
        self.reward = 0                 
        
    def add_target_coordinate(self, point: Tuple[float, float]):
        self.target_coordinates.append(point)

    
    def get_pitch_rad(self):
        return  self.pitch * pi / 180 
    
    def get_pitch(self):
        return self.pitch


    def set_thrust(self, thrust_percentage: Tuple[float, float]):
        assert(len(thrust_percentage) == 2)
        assert(0<=thrust_percentage[0]<=1)
        assert(0<=thrust_percentage[1]<=1)
        self.thrust_left = thrust_percentage[0] * self.max_thrust
        self.thrust_right = thrust_percentage[1] * self.max_thrust

    
    def get_next_target(self) -> Tuple[float, float]:
        #return (0,0) if len(self.target_coordinates)==0 else self.target_coordinates[0]

        self.add_target_coordinate((uniform(-0.4, 0.5), uniform(-0.4, 0.5)))       
        return self.target_coordinates


    def step_simulation(self, delta_time: float):
        # Set the target reached flag to false
        self.has_reached_target_last_update = False
        self.t += delta_time

        # here we calculate the total thrust & torque 
        total_thrust = self.thrust_left+self.thrust_right
        total_torque = (self.thrust_left - self.thrust_right)*self.turning_constant # momentum, torque

        # we calculate the x, y and theta accelation components 
        thrust_vec_x = np.sin(self.pitch) # for doc is y 
        thrust_vec_y = np.cos(self.pitch) # for doc is z
        
        acc_x_h = (total_thrust * thrust_vec_x) / self.mass - self.velocity_drag * self.velocity_x              
        acc_y_h = (total_thrust * thrust_vec_y) / self.mass + self.g - self.velocity_drag * self.velocity_y
        theta_acc_h = (total_torque) / self.mass - self.pitch_drag_constant * np.abs(self.pitch_velocity)
        

        velocity_size = np.sqrt(self.velocity_x*self.velocity_x+self.velocity_y*self.velocity_y) # this is not used 

        # Drive the speed and the possition of the drone using small aproximations: cos(theta) = 1, and sin(theta) = theta
        # new component is = as old component + accelaration * time : Euler method

        # First part of delta time
        vel_x_h = self.velocity_x + acc_x_h * delta_time/2                      # new x velocity
        vel_y_h = self.velocity_y + acc_y_h * delta_time/2                      # new y velocity
        theta_vel_h = self.pitch_velocity + theta_acc_h * delta_time / 2        # new theta velocity

        x_h = self.x + vel_x_h * delta_time / 2                                 # new x possition
        y_h = self.y + vel_y_h * delta_time / 2                                 # new y possition
        theta_h = self.pitch + theta_vel_h * delta_time / 2                     # new  theta angle

        # we calculate the x, y and theta accelation components 
        thrust_vec_x = np.sin(theta_h)
        thrust_vec_y = np.cos(theta_h)
        velocity_size = np.sqrt(vel_x_h*vel_x_h+vel_y_h*vel_y_h)                # this is not used

        acc_x_f = (total_thrust * thrust_vec_x) / self.mass - self.velocity_drag * self.velocity_x
        acc_y_f = (total_thrust * thrust_vec_y) / self.mass + self.g - self.velocity_drag * self.velocity_y
        theta_acc_f = (total_torque) / self.mass - self.pitch_drag_constant * np.abs(self.pitch_velocity)

        # First part of delta time
        self.velocity_x = vel_x_h + acc_x_f * delta_time/2                      # new x velocity
        self.velocity_y = vel_y_h + acc_y_f * delta_time/2                      # new y velocity
        self.pitch_velocity = theta_vel_h + theta_acc_f * delta_time / 2        # new theta velocity

        self.x = x_h + self.velocity_x * delta_time / 2                         # new x possition
        self.y = y_h + self.velocity_y * delta_time / 2                         # new y possition
        self.pitch = theta_h + self.pitch_velocity * delta_time / 2             # new theta angle

        # Updates the target list
        target_point = self.get_next_target()
        distance_x = self.x - target_point[0]
        distance_y = self.y - target_point[1]
        distance_to_target = np.sqrt(distance_x*distance_x+distance_y*distance_y)

        if distance_to_target < self.game_target_size:
            # if True that means we reached the target
            if len(self.target_coordinates) > 0:
                self.target_coordinates.pop(0)
                self.has_reached_target_last_update = True






    
