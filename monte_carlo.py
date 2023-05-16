import datetime
import math
from typing import Tuple
import matplotlib.pyplot as plt



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
        
        
        self.states = []
        self.actions = []
        self.rewards = []
        
        self.G = []
        self.env = env
        self.lives = 2
        self.current_action = 3
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
        
        graph[(5,19)] = [(5,18)]
        graph[(9,19)] = [(9,18)]
        
        graph[(5,1)] = [(5,2)]
        graph[(9,1)] = [(9,2)]
        
        # graph[(9,10)] = [(9,9),(9,11)]
        
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

    # This function was used in a prior piece of work for MLiS part 1 
    def value_calc(self, state):
        
        if (np.random.rand() > 0.98):
            return np.random.randint(pos_actions)
        else:  
            maxValue = max(self.value_function[state[0]][state[1]][state[2]][state[3]])
            i = 0
            for n in range(pos_actions):
                if(maxValue == self.value_function[state[0]][state[1]][state[2]][state[3]][n]):
                    i=n
            return i

    def reward(self, ghost_dist):
        # negative if ghosts too close
        
        too_close = 0
        if(ghost_dist < 2):
            too_close = -500
        elif(ghost_dist < 3):
            too_close = -300
        elif(ghost_dist < 4):
            too_close = -100
        
        
        life_bonus = self.lives * 10
        
        pillsLeft = np.count_nonzero(self.pillMap == 2)*2

        reward = (ghost_dist)*10 + (self.score*50) + too_close - pillsLeft + life_bonus
        
        return reward
    
    def ghostMapReset(self):
        # orange 3, blue 4, pink 5, red 6
        self.ghostMap = np.loadtxt("maps/ghost_map.txt", dtype="uint8", delimiter=' ')

        
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
        
        
        # Action
        self.action = self.value_calc(state)
        self.actions.append(self.action)

        action = self.action_selection()


        # Reward
        reward = self.reward(ghost_dist)
        self.rewards.append(reward)
        
        return action
    
    def direction(self, x, y):
        
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

    def action_selection(self):
                
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

            action = self.direction(x, y)
                    
        elif(self.action == 1):
            # run from ghosts    
            route = self.bfs("ghost", self.graph, (self.pacman_y, self.pacman_x))            
            directions = []
            
            for i in self.graph[(self.pacman_y, self.pacman_x)]:
                directions.append(i)
            
            if(len(directions) != 1):
                directions.remove(route[1])      
                                    
            if(len(route) != 1):
                direction = directions[0]

                y = direction[0] - self.pacman_y
                x = direction[1] - self.pacman_x
            else:
                x = 1
                y = 1
        
            action = self.direction(x, y)
                        
        elif(self.action == 2):
            # Chase ghosts 
            route = self.bfs("ghost", self.graph, (self.pacman_y, self.pacman_x))
            if(len(route) != 1):
                direction = route[1]
         
                y = direction[0] - self.pacman_y
                x = direction[1] - self.pacman_x
            else:
                x = 1
                y = 1

            action = self.direction(x, y)
                        
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
            self.pillMap = self.pillMapGen()

            # 1) modify parameters
            self.states = []
            self.actions = []
            self.rewards = []
                        
            # 3) run simulation
            for j in range(self.steps):
                self.ghostMapReset()

                self.pillMap[self.pacman_y][self.pacman_x] = 0
                
                action = self.state_machine()
                # print(action)
                # self.current_action = self.state_machine()

                # observation, reward, terminated, truncated, info = self.env.step(self.current_action)
                observation, reward, terminated, truncated, info = self.env.step(action)
                self.setCoordinates(observation)
                self.lives = observation[123]
                
                self.score = reward
                
                if terminated or truncated:
                    observation, info = self.env.reset()
                    break            
                                        
            states, actions, rewards = self.states, self.actions, self.rewards

            # 4) measure change in quality
            total_new_rewardsblip = np.sum(rewards)
            print(n, total_new_rewardsblip)
            total_rewards.append(total_new_rewardsblip)

            G = 0
            
            # This function was used in a prior piece of work for MLiS part 1 
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
    
    def run(self, env):
        env.reset()
        self.pillMap = self.pillMapGen()
        rewardSum = 0
        graphScore = []
        # movement starts at 66 frames
        for i in range(20000):
            self.ghostMapReset()

            self.pillMap[self.pacman_y][self.pacman_x] = 0
            action = self.state_machine()
            # print(self.action)

            observation, reward, terminated, truncated, info = env.step(action)
            self.setCoordinates(observation)
            self.lives = observation[123]
            
            self.score = reward
            
            
            rewardSum += reward
            
            graphScore.append(rewardSum)
            
            if terminated or truncated:
                observation, info = env.reset()
                print("Survived: " + str(i))
                break            

        env.close()
        
        df = pd.DataFrame(graphScore)
        df.to_csv("MCScore.csv", index=False)       
        
    def runPillCollect(self, env):
        env.reset()
        self.pillMap = self.pillMapGen()
        rewardSum = 0
        graphScore = []
        # movement starts at 66 frames
        for i in range(20000):
            self.ghostMapReset()

            self.pillMap[self.pacman_y][self.pacman_x] = 0
            self.action = 0
            action = self.action_selection()
            # print(self.action)

            observation, reward, terminated, truncated, info = env.step(action)
            self.setCoordinates(observation)
            self.lives = observation[123]
            
            self.score = reward
            
            
            rewardSum += reward
            
            graphScore.append(rewardSum)
            
            if terminated or truncated:
                observation, info = env.reset()
                print("Survived: " + str(i))
                break            

        env.close()
        
        df = pd.DataFrame(graphScore)
        df.to_csv("MCPillScore.csv", index=False)
        
    def runGhostEvade(self, env):
        env.reset()
        self.pillMap = self.pillMapGen()
        rewardSum = 0
        graphScore = []
        # movement starts at 66 frames
        for i in range(20000):
            self.ghostMapReset()

            self.pillMap[self.pacman_y][self.pacman_x] = 0
            self.action = 1
            action = self.action_selection()
            # print(self.action)

            observation, reward, terminated, truncated, info = env.step(action)
            self.setCoordinates(observation)
            self.lives = observation[123]
            
            self.score = reward
            
            
            rewardSum += reward
            
            graphScore.append(rewardSum)
            
            if terminated or truncated:
                observation, info = env.reset()
                print("Survived: " + str(i))    
                break            

        env.close()
        
        df = pd.DataFrame(graphScore)
        df.to_csv("MCGhostScore.csv", index=False)

        
    # This function was used in a prior piece of work for MLiS part 1 
    def load(self):
        try:
            parameter_array = np.load('pacman_controller_parameters.npy')
            self.value_function = parameter_array[0]
            print((parameter_array[0]))
        except:
            print("Could not load parameters, sticking with default parameters.")
            
    # This function was used in a prior piece of work for MLiS part 1 
    def save(self):
        parameter_array = np.array([self.value_function])
        np.save('pacman_controller_parameters.npy', parameter_array)