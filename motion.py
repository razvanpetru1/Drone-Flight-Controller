import numpy as np
from flight_controller import FlightController
from drone import Drone
from typing import Tuple
from drone import Drone

class Controller(FlightController):
    # need to set the parameter properly again
    def __init__(self):
        # vertical
        self.mass = 1
        self.gravity = 1
        self.target_altitude = 10 
        self.current_altitude = 0  
        self.vertical_velocity = 0  
        self.k = 1  # k and b are just positive constants
        self.b = 1 
        self.epsilon_min = 0  
        self.epsilon_max = 1 
        self.target_y = 10
        self.current_y = 0
        
        self.t_eq = None
        self.mg = None
        
        # horizontal
        self.theta = 0 # angle
        self.target_pitch = 10    
        self.k_theta = 0
        self.b_theta = 0
        self.theta_velocity = 0
        self.gamma_min = 0
        self.gamma_max= 0
        self.t_max = 10
    
    # Example of vertical motion 
    def get_vertical_thrust(self) -> Tuple[float, float, float]:
        epsilon = self.get_epsilon()
        self.mg = self.mass *self.gravity
        # equilibrium thrust 
        self.t_eq = 0.5 * self.mg
        
        vertical_t1 = self.t_eq + epsilon / 2
        vertical_t2 = self.t_eq - epsilon / 2
        f = vertical_t1 + vertical_t2 - self.mg
        return (vertical_t1, vertical_t2, f)
    
    def get_epsilon(self):
        altitude_error= self.target_y- self.current_y
        epsilon = min(self.epsilon_max, max(-self.k * altitude_error - self.b * self.vertical_velocity, self.epsilon_min))
        return epsilon
    
    # Example of horizontal motion
    def get_horizontal_thrust(self)-> Tuple[float, float, float]:
        pitch_error = self.target_pitch - self.theta
        gamma = np.clip(-self.k_theta * pitch_error - self.b_theta * self.theta_velocity, self.gamma_min, self.gamma_max)
        torque = 2 * gamma
        
        #in the case of eliminate vertical motion
        self.mg = 2 * self.t_eq *np.cos(self.theta)
        # equilibrium thrust with eliminate vertical motion
        self.t_eq = np.sqrt((2 * self.t_max * np.cos(self.theta)) / np.cos(self.target_pitch))

        horizontal_t1 = np.clip(self.t_eq + gamma, 0, self.t_max)
        horizontal_t2 = np.clip(self.t_eq - gamma, 0, self.t_max)
        return (horizontal_t1, horizontal_t2, torque)

    #TODO: On Working
    # Example of final motion
    def get_thrusts(self, drone: Drone) -> Tuple[float, float]:
        target_point = drone.get_next_target()
        dx = target_point[0] - drone.x
        dy = target_point[1] - drone.y
        pass
        

