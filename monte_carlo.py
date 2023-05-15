import datetime
import math
from typing import Tuple


# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pos_actions=3

class MonteCarlo:

    def __init__(self, env):        
        self.pacman_x = 10
        self.pacman_y = 9
        
        self.orange_x = 10
        self.blue_x = 10
        self.pink_x = 10
        self.red_x = 10
        
        self.orange_y = 7
        self.blue_y = 7
        self.pink_y = 7
        self.red_y = 5
        
        self.cherry_x = 0
        self.cherry_y = 0
        
        self.G = []
        self.env = env
        
        self.current_action = 3
        self.counter = 0
        self.action = 0
        self.score = 0
        values = np.random.random([15, 21, 15, 21, pos_actions]) 
        self.value_function = values
        self.steps = 20000
        
        # 12y 8x is one map square 
        self.defaultMap = self.mapGen()
        self.pillMap = self.pillMapGen()
        self.ghostMap = self.ghostMapGen()
        self.ghostMapReset()
        
        self.graph = self.graphGen()
        
    def graphGen(self):
        graph = {}

        for y in range(1, 15):
            for x in range(1, 20):
                coord = []
                # print("(" + str(y) + "," + str(x) + ")")
                for dy, dx in [(-1, 0), (0, +1), (+1, 0), (0, -1)]:
                    a, b = y + dy, x + dx
                    if(self.defaultMap[a][b] != 1):
                        coord.append((a,b))                            
                    
                graph[(y,x)] = coord
        
        # graph[(5,0)] = [(5,1),(5,20)]
        # graph[(9,0)] = [(9,1),(9,20)]
        
        # graph[(5,20)] = [(5,19),(5,0)]
        # graph[(9,20)] = [(9,19),(9,0)]
        
        
        graph[(5,0)] = [(5,1)]
        graph[(9,0)] = [(9,1)]
        graph[(5,20)] = [(5,19)]
        graph[(9,20)] = [(9,19)]
        
        return graph
        
    def mapGen(self):
        # board[y][x]
        board = np.loadtxt("maps/map.txt", dtype="uint8", delimiter=' ')
        return board
        
    def pillMapGen(self):
        # board[y][x]
        board = np.loadtxt("maps/pill_map.txt", dtype="uint8", delimiter=' ')
        return board
    
    def ghostMapGen(self):
        # board[y][x]
        board = np.loadtxt("maps/ghost_map.txt", dtype="uint8", delimiter=' ')
        return board

    def value_calc(self, state):
        
        if (np.random.rand() > 0.99):
            return np.random.randint(pos_actions)
        else:  
            maxValue = max(self.value_function[state[0]][state[1]][state[2]][state[3]])
            i = 0
            for n in range(pos_actions):
                if(maxValue == self.value_function[state[0]][state[1]][state[2]][state[3]][n]):
                    i=n
            return i

    def reward(self, state, ghost_dist):
        # negative if ghosts too close
        # negative on death
        
        reward = -ghost_dist + self.score
        
        return reward
    
    def ghostMapReset(self):
        # orange 3, blue 4, pink 5, red 6
        self.ghostMap = self.defaultMap
        
        self.ghostMap[self.orange_y][self.orange_x] = 3
        self.ghostMap[self.blue_y][self.blue_x] = 4
        self.ghostMap[self.pink_y][self.pink_x] = 5
        self.ghostMap[self.red_y][self.red_x] = 6
        
    def closestGhost(self):
        route = self.bfs("ghost", self.graph, (self.pacman_y, self.pacman_x))        
        ghost_y, ghost_x = route[-1]

        x = int(ghost_x)
        y = int(ghost_y)
        dist = len(route)
        
        return y, x, dist

    # found solution for returning path from stack overflow answered by SeasonalShot
    # https://stackoverflow.com/questions/8922060/how-to-trace-the-path-in-a-breadth-first-search
    def bfs(self, map_type, graph, start):
              
        if(map_type == "ghost"):
            board = self.ghostMap
        elif(map_type == "pill"):
            board = self.pillMap

            
        queue = [(start,[start])]
        visited = []
    
        while queue:
            vertex, path = queue.pop(0)
            visited.append(vertex)
            
            for node in graph[vertex]:
                if(board[node[0]][node[1]] > 0):
                    return path + [node]
                else:
                    if node not in visited:
                        visited.append(node)
                        queue.append((node, path + [node]))
                        
        return [(1,1)]
    
    def state_machine(self):
        ghost_y, ghost_x, ghost_dist = self.closestGhost()
        
        # State
        state = (
            int(self.pacman_y),
            int(self.pacman_x),
            int(ghost_y),
            int(ghost_x)
        )
        self.states.append(state)
        
        action = self.action_selection()
        
        # Action
        self.action = self.value_calc(state)
        # self.action = 0
        self.actions.append(self.action)

        # Reward
        reward = self.reward(state, ghost_dist)
        self.rewards.append(reward)
        
        return action

    def action_selection(self):
        # pacman_y = state[0]
        # pacman_x = state[1]   

        if(self.action == 0):
            # collect pills
            route = self.bfs("pill", self.graph, (self.pacman_y, self.pacman_x))
            
            
            if(len(route) != 1):
                direction = route[1]
         
                y = direction[0] - self.pacman_y
                x = direction[1] - self.pacman_x
            else:
                x = 1
                y = 1
            
            
            if(x != 0):
                if(x > 0):
                    action = 2 # right
                else: 
                    action = 3 # left
            
            elif(y !=0):
                if(y > 0):
                    action = 4 # down
                else: 
                    action = 1 # up
        
        elif(self.action == 1):
            # run from ghosts    
            
            route = self.bfs("ghost", self.graph, (self.pacman_y, self.pacman_x))
            if(len(route) != 1):
                direction = route[1]
         
                y = direction[0] - self.pacman_y
                x = direction[1] - self.pacman_x
            else:
                x = 1
                y = 1
            
            if(x != 0):
                if(x > 0):
                    action = 3 # left
                else: 
                    action = 2 # right
            
            elif(y !=0):
                if(y > 0):
                    action = 1 # up
                else: 
                    action = 4 # down
            
            # action = 1
        
        elif(self.action == 2):
            # chase ghosts 
            route = self.bfs("ghost", self.graph, (self.pacman_y, self.pacman_x))
            if(len(route) != 1):
                direction = route[1]
         
                y = direction[0] - self.pacman_y
                x = direction[1] - self.pacman_x
            else:
                x = 1
                y = 1
            
            if(x != 0):
                if(x > 0):
                    action = 2 # right
                else: 
                    action = 3 # left
            
            elif(y !=0):
                if(y > 0):
                    action = 4 # down
                else: 
                    action = 1 # up
            
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
        self.values= np.random.random([15,21,15,21,pos_actions])
        
        average_returns = np.empty([15,21,15,21,pos_actions])
        average_returns_count = np.zeros([15,21,15,21,pos_actions])
        Q = np.empty([15,21,15,21,pos_actions])

        total_rewards = []
        
        # --- Code snipped provided for guidance only --- #
        for n in range(epochs):
            
            self.env.reset()
            
            # 1) modify parameters
            self.states = []
            self.actions = []
            self.rewards = []

            self.counter = 1
                        
            # 3) run simulation
            for j in range(self.steps):
                self.pillMap[self.pacman_y][self.pacman_x] = 0
                
                action = self.state_machine()
                # print(action)

                # observation, reward, terminated, truncated, info = self.env.step(self.current_action)
                observation, reward, terminated, truncated, info = self.env.step(action)
                self.setCoordinates(observation)
                
                
                self.score = reward
                
                if terminated or truncated:
                    observation, info = self.env.reset()
                    break            
                                        
            states, actions, rewards = self.states, self.actions, self.rewards

            # 4) measure change in quality
            total_new_rewardsblip = np.sum(rewards)
            print(n, total_new_rewardsblip,self.counter)
            total_rewards.append(total_new_rewardsblip)

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