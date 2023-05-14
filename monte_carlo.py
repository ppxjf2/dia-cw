import datetime
import math
from typing import Tuple


# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pos_actions=5

class MonteCarlo:

    def __init__(self, env):
        self.pacman_x = 0
        self.pacman_y = 0
        
        self.orange_x = 0
        self.blue_x = 0
        self.pink_x = 0
        self.red_x = 0
        
        self.orange_y = 0
        self.blue_y = 0
        self.pink_y = 0
        self.red_y = 0
        
        self.cherry_x = 0
        self.cherry_y = 0
        
        self.G = []
        self.env = env
        
        # values= np.random.random([,pos_actions]) 
        # self.value_function = values
        # self.steps
        
        # 12y 8x is one map square 
        self.defaultMap = self.mapGen()
        self.pillMap = self.pillGen()
        self.ghostMap = self.mapGen()
        
    def mapGen(self):
        # board[y][x]
        board = np.loadtxt("small_map_test.txt", dtype="uint8", delimiter=' ')
        return board
        
    def pillGen(self):
        # board[y][x]
        board = np.loadtxt("pill_map.txt", dtype="uint8", delimiter=' ')
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

    def reward(self, score):
        
        # negative if ghosts too close
        # negative on death
        # 
        
        
        reward = ghost_distance
        
        
        
        return reward
    
    def ghostMapReset(self):
        # orange 3, blue 4, pink 5, red 6
        self.ghostMap = self.defaultMap
        
        self.ghostMap[self.orange_y][self.orange_x] = 3
        self.ghostMap[self.blue_y][self.blue_x] = 4
        self.ghostMap[self.pink_y][self.pink_x] = 5
        self.ghostMap[self.red_y][self.red_x] = 6
        
    def closestGhost(self):
        
        x = 0
        y = 0
        
        return x, y
         

    def action_selection(self, states, action):
        
        ghost_x, ghost_y = self.closestGhost()
        
        state = (
            int(self.pacman_x),
            int(self.pacman_y),
            int(),
            int()
        )
        self.states.append(state)
        
        # Action
        self.action = self.value_calc(state)
        self.actions.append(self.action)

        # Reward
        reward = self.reward()
        self.rewards.append(reward)


        up = self.defaultMap[self.pacman_y-1][self.pacman_x]
        down = self.defaultMap[self.pacman_y+1][self.pacman_x]
        left = self.defaultMap[self.pacman_y][self.pacman_x-1]
        right = self.defaultMap[self.pacman_y][self.pacman_x+1]

        if(action == 0):
            # run from ghosts
        
            # if(self.ghost):
        
        
            # if(up == 0):
            
            
            
            return 1
        
        elif(action == 1):
            # collect pills
            
            return 1
        
        elif(action == 2):
            # chase blue ghosts 
            
            return 1

        
        
        return action
    
    def setCoordinates(self, observation):
        self.pacman_x = math.floor((observation[10]+2)/8)-1
        self.pacman_y = math.floor((observation[16]+10)/12)
        
        self.orange_x = math.floor((observation[6]+2)/8)-1
        self.blue_x = math.floor((observation[7]+2)/8)-1
        self.pink_x = math.floor((observation[8]+2)/8)-1
        self.red_x = math.floor((observation[9]+2)/8)-1
        
        self.orange_y = math.floor((observation[12]+10)/12)
        self.blue_y = math.floor((observation[13]+10)/12)
        self.pink_y = math.floor((observation[14]+10)/12)
        self.red_y = math.floor((observation[15]+10)/12)
        
        self.cherry_x = math.floor((observation[11]+2)/8)-1
        self.cherry_y = math.floor((observation[17]+10)/12)
        

    def train(self, epochs):
        # self.values= np.random.random([20,40,30,10,pos_actions]); 
        
        # average_returns = np.empty([20,40,30,10,pos_actions])
        # average_returns_count = np.zeros([20,40,30,10,pos_actions])
        # Q = np.empty([20,40,30,10,pos_actions])

        # total_rewards = []
        # total_targets = []



        # --- Code snipped provided for guidance only --- #
        for n in range(epochs):
            
            self.env.reset()
            
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
                self.setCoordinates(observation)
                
                
                
                self.pillMap[self.pacman_y][self.pacman_x] = 0
                
                fitness += reward

                if terminated or truncated:
                    observation, info = self.env.reset()
                    break            
                
                
                
            
            # for t in range(self.get_max_simulation_steps()):
            #     drone.set_thrust(self.get_thrusts(drone))
            #     drone.step_simulation(self.get_time_interval())
            
            # states, actions, rewards = self.states, self.actions, self.rewards

            # # 4) measure change in quality
            # total_new_rewardsblip = np.sum(rewards)
            # print(n, total_new_rewardsblip,self.counter)
            # total_rewards.append(total_new_rewardsblip)
            # total_targets.append(self.counter)

            # G = 0
            # # 5) update parameters according to algorithm
            # for i in reversed(range(len(actions))):
            #     k = len(actions)-1
            #     total_new_rewardsblip
            #     G = np.average(rewards[(k-i):])

            #     cur_reward = self.value_function[states[k-i][0]][states[k-i][1]][states[k-i][2]][states[k-i][3]][actions[k-i]]

            #     average_returns[states[k-i][0]][states[k-i][1]][states[k-i][2]][states[k-i][3]][actions[k-i]] += G
            #     average_returns_count[states[k-i][0]][states[k-i][1]][states[k-i][2]][states[k-i][3]][actions[k-i]] += 1

            #     Q[states[k-i][0]][states[k-i][1]][states[k-i][2]][states[k-i][3]][actions[k-i]] = average_returns[states[k-i][0]][states[k-i][1]][states[k-i][2]][states[k-i][3]][actions[k-i]] / average_returns_count[states[k-i][0]][states[k-i][1]][states[k-i][2]][states[k-i][3]][actions[k-i]]


            #     self.value_function[states[k-i][0]][states[k-i][1]][states[k-i][2]][states[k-i][3]][actions[k-i]] = Q[states[k-i][0]][states[k-i][1]][states[k-i][2]][states[k-i][3]][actions[k-i]]

        pass
        

    def load(self):
        try:
            parameter_array = np.load('pacman_controller_parameters.npy')
            self.value_function = parameter_array[0]
            print((parameter_array[0]))
        except:
            print("Could not load parameters, sticking with default parameters.")

    def save(self):
        parameter_array = np.array([self.value_function])
        np.save('pacman_controller_parameters.npy', parameter_array)