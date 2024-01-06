from flight_controller import FlightController
from drone import Drone
from typing import Tuple
import numpy as np
from math import sin, cos, pi, sqrt
from random import randrange



class CustomController(FlightController):

    def __init__(self):
        pass



    def train(self):
        pass    
    def get_thrusts(self, drone: Drone) -> Tuple[float, float]:
        return (0.5, 0.5) # Replace this with your custom algorithm
    def load(self):
        pass
    def save(self):
        pass
    
    def reset(self):
        # Reset variables
        self.x = 0 # randrange(480, 720)
        self.y = 0 # randrange(480, 720)
        self.pitch = 0
        
        self.velocity_y = 0
        self.velocity_x = 0
        self.pitch_velocity = 0

        self.acceleration_y = 0
        self.acceleration_x = 0
        self.pitch_acceleration = 0

        self.target_coordinate(0.35, 0.3)

        self.thrust_left = 0.5
        self.thrust_right = 0.5               

        self.target_counter = 0
        self.reward = 0
        self.t = 0

        return self.get_obs()
    def get_observation(self) -> np.ndarray:
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
        
        angle_to_up = self.a / 180 * pi                                              # we use radians for theta
        velocity = sqrt(self.velocity_x*self.velocity_x + self.velocity_y**2)        # absolut value of the velocity vector
        angle_velocity = self.pitch_velocity
        
        target_point = self.get_next_target()
        distance_x = self.x - target_point[0]
        distance_y = self.y - target_point[1]
        distance_to_target = np.sqrt(distance_x*distance_x+distance_y*distance_y)
        
        angle_to_target = np.arctan2( self.target_point[1] - self.y , self.target_point[0] - self.x ) # opposide side /adjacent side
        
        # Angle between the to_target vector and the velocity vector
        angle_target_and_velocity = angle_to_target - np.arctan2(self.velocity_y, self.velocity_x)
    
        
        return np.array(
            [
                angle_to_up,
                velocity,
                angle_velocity,
                distance_to_target,
                angle_to_target,
                angle_target_and_velocity,
            ]
        ).astype(np.float32)