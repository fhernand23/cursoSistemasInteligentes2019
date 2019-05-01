import gym
from gym import spaces
from enum import Enum
from collections import defaultdict

class Action(Enum):
    REST = 0
    CLIMB = 1
    DOWN = 2

class Position(Enum):
    BOTTOM = 0
    MIDDLE = 1
    TOP = 2

class BottomTopEnv(gym.Env):
    """
    BottomTop environment.
    You are an agent on an 3 state network with actions / rewards / next state:
    - BOTTOM (REST 0.1 BOTTOM; CLIMB 0 TOP;   DOWN -1 BOTTOM)
    - MIDDLE (REST -1 MIDDLE;  CLIMB 0.3 TOP; DOWN 1 BOTTOM)
    - TOP    (REST 0.1 TOP;    CLIMB -1 TOP;  DOWN 0.2 MIDDLE)
    """
    def __init__(self):
        # environment
        self.shape = defaultdict();
        self.shape[Position.BOTTOM] = [[Action.REST,0.1,Position.BOTTOM],[Action.CLIMB,0,Position.TOP],[Action.DOWN,-1,Position.BOTTOM]]
        self.shape[Position.MIDDLE] = [[Action.REST,-1,Position.MIDDLE],[Action.CLIMB,0.3,Position.TOP],[Action.DOWN,1,Position.BOTTOM]]
        self.shape[Position.TOP] = [[Action.REST,0.1,Position.TOP],[Action.CLIMB,-1,Position.TOP],[Action.DOWN,0.2,Position.MIDDLE]]

        # initial position
        self.state = Position.BOTTOM.value

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Discrete(3)

    def getPossibleAction(self, current):
        return self.shape[current]

    def step(self, action):
        assert self.action_space.contains(action)

        # reward based on current state and action
        reward = self.shape[Position(self.state)][action][1]
        # next state based on current state and action
        self.state = self.shape[Position(self.state)][action][2]

        done = False
        return self.state, reward, done, {}

    def reset(self):
        self.state = Position.BOTTOM
        return self.state
