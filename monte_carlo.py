import datetime
from math import floor, log
from typing import Tuple

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pos_actions=5

class MonteCarlo:

    def __init__(self):
        self.pacman_x = 0
        self.pacman_y = 0
        self.red_x = 0
        self.red_y = 0
        self.orange_x = 0
        self.orange_y = 0
        self.blue_x = 0
        self.blue_y = 0
        self.pink_x = 0
        self.pink_y = 0
        self.cherry_x = 0
        self.cherry_y = 0
        self.G = []
        
        # values= np.random.random([,pos_actions]) 
        self.value_function = values
        self.steps
        
        self.pillMap = self.mapGen()
        self.ghostMap = self.mapGen()
        
    def mapGen(self):
        # board[y][x]
        board = np.loadtxt("small_map_test.txt", dtype="uint8", delimiter=' ')
        
        return board
        

    def value_calc(self, state):
        
        if (np.random.rand() > 0.99):
            return np.random.randint(pos_actions)
        else:  
            max1 = max(self.value_function[state[0]][state[1]][state[2]][state[3]])
            i = 0
            for n in range(pos_actions):
                if(max1 == self.value_function[state[0]][state[1]][state[2]][state[3]][n]):
                    i=n
            return i

    def reward(self):
        reward = 0
        return reward
    

    def action_selection(self, states):

        self.pacman_x 
        self.pacman_y
                
        
        return action

    def behaviour(self):
        
            
        return

    def train(self):
        self.values= np.random.random([20,40,30,10,pos_actions]); 
        
        epochs = 1
        average_returns = np.empty([20,40,30,10,pos_actions])
        average_returns_count = np.zeros([20,40,30,10,pos_actions])
        Q = np.empty([20,40,30,10,pos_actions])

        total_rewards = []
        total_targets = []



        # --- Code snipped provided for guidance only --- #
        for n in range(epochs):
            # 1) modify parameters
            self.states = []
            self.actions = []
            self.rewards = []

            # 2) create a new drone simulation
            drone = self.init_drone()
            self.nexttarget=drone.get_next_target()
            self.counter = 1

            # 3) run simulation
            for j in range(self.steps):

                action = self.action_selection()

                observation, reward, terminated, truncated, info = self.env.step(action)
                
                self.pacman_x = observation[17]
                self.pacman_y = observation[17]
                
                self.red_x = observation[10]
                self.red_y = observation[16]
                self.orange_x = observation[7]
                self.orange_y = observation[13]
                self.blue_x = observation[8]
                self.blue_y = observation[14]
                self.pink_x = observation[9]
                self.pink_y = observation[15]
                self.cherry_x = observation[12]
                self.cherry_y = observation[18]
                
                
                
                fitness += reward

                if terminated or truncated:
                    observation, info = self.env.reset()
                    break
                #self.env.close()
            
            
            for t in range(self.get_max_simulation_steps()):
                drone.set_thrust(self.get_thrusts(drone))
                drone.step_simulation(self.get_time_interval())
            
            states, actions, rewards = self.states, self.actions, self.rewards

            # 4) measure change in quality
            total_new_rewardsblip = np.sum(rewards)
            print(n, total_new_rewardsblip,self.counter)
            total_rewards.append(total_new_rewardsblip)
            total_targets.append(self.counter)

            G = 0
            # 5) update parameters according to algorithm
            for i in reversed(range(len(actions))):
                k = len(actions)-1
                total_new_rewardsblip
                G = np.average(rewards[(k-i):])

                cur_reward = self.value_function[states[k-i][0]][states[k-i][1]][states[k-i][2]][states[k-i][3]][actions[k-i]]

                average_returns[states[k-i][0]][states[k-i][1]][states[k-i][2]][states[k-i][3]][actions[k-i]] += G
                average_returns_count[states[k-i][0]][states[k-i][1]][states[k-i][2]][states[k-i][3]][actions[k-i]] += 1

                Q[states[k-i][0]][states[k-i][1]][states[k-i][2]][states[k-i][3]][actions[k-i]] = average_returns[states[k-i][0]][states[k-i][1]][states[k-i][2]][states[k-i][3]][actions[k-i]] / average_returns_count[states[k-i][0]][states[k-i][1]][states[k-i][2]][states[k-i][3]][actions[k-i]]


                self.value_function[states[k-i][0]][states[k-i][1]][states[k-i][2]][states[k-i][3]][actions[k-i]] = Q[states[k-i][0]][states[k-i][1]][states[k-i][2]][states[k-i][3]][actions[k-i]]

        pass

        from datetime import datetime
        import csv

        variablename = {"total_reward":total_rewards, "targets_hit":total_targets}
        df = pd.DataFrame(variablename)
        now = datetime.now()
        time = now.strftime(f"Drone-epochs {epochs} %d-%m-%Y_%H-%M-%S")
        df.to_csv(f'{time}.csv')

        # plt.figure()
        # plt.plot(total_rewards, label="Sampled Mean Return", alpha=1)
        # plt.xlabel("Epochs")
        # plt.ylabel("Avg Return")
        # plt.show()
        

    def load(self):
        """Load the parameters of this flight controller from disk.
        """
        try:
            parameter_array = np.load('pacman_controller_parameters.npy')
            self.value_function = parameter_array[0]
            print((parameter_array[0]))
        except:
            print("Could not load parameters, sticking with default parameters.")

    def save(self):
        """Save the parameters of this flight controller to disk.
        """
        parameter_array = np.array([self.value_function])
        np.save('pacman_controller_parameters.npy', parameter_array)