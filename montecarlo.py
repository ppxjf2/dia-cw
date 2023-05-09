import numpy as np
import math
from typing import Tuple
from pacman_controller import MonteCarlo



def is_training() -> bool:
    return True # <--- Replace this with True if you want to train, false otherwise
def is_saving() -> bool:
    return True # <--- Replace this with True if you want to save the results of training, false otherwise


def main(controller: FlightController):

    # Initialise pygame
    pygame.init()
    clock = pygame.time.Clock()
    
    # Initalise the drone
    drone = controller.init_drone()
    
    simulation_step_counter = 0
    max_simulation_steps = controller.get_max_simulation_steps()
    delta_time = controller.get_time_interval()


    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False


        # Makes sure that the simulation runs at a target 60FPS
        clock.tick(60)

        # Checks whether to reset the current drone
        simulation_step_counter+=1
        if (simulation_step_counter>max_simulation_steps):
            drone = controller.init_drone() # Reset the drone
            simulation_step_counter = 0
            
if __name__ == "__main__":

    controller = generate_controller()
    if is_training():
        controller.train()
        if is_saving():
            controller.save()        
    else:
        controller.load()
    
    main(controller)