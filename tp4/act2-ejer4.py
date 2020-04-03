import gym
import numpy as np
import math
from collections import deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

class doubleqlearning_mountaincar():
    def __init__(self, episodes=500, min_alpha=0.1, min_epsilon=0.1, gamma=1.0):
        # down-scaling feature space to discrete range
        self.buckets = (12, 12, 8, 1)
        # training episodes
        self.episodes = episodes
        # minimun learning rate
        self.min_alpha = min_alpha
        # minimun exploration rate
        self.min_epsilon = min_epsilon
        # discount factor
        self.gamma = gamma
        # set environment
        self.env = gym.make('MountainCarContinuous-v0')

        # initialising double Q-table
        self.Q1 = np.zeros(self.buckets)
        self.Q2 = np.zeros(self.buckets)

    # Discretize input space to make Q-table
    def discretize(self, obs):
        upper_bounds = [self.env.observation_space.high[0], self.env.observation_space.high[1]]
        lower_bounds = [self.env.observation_space.low[0], self.env.observation_space.low[1]]
        ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
        new_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        return tuple(new_obs)

    # Adaptive learning of Exploration Rate
    def get_epsilon(self, t):
        return max(self.min_epsilon, min(1, 1.0 - math.log10((t + 1) / 25)))

    # Adaptive learning of Learning Rate
    def get_alpha(self, t):
        return max(self.min_alpha, min(1.0, 1.0 - math.log10((t + 1) / 25)))

    # Choosing action based on epsilon-greedy policy
    def choose_action(self, state, epsilon):
        cont_action = self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax([item1 + item2 for item1, item2 in zip(self.Q1[state], self.Q2[state])])

        return self.encode_action(cont_action)

    def encode_action(self, cont_action):
        if cont_action in [0,1,2,3,4,5,6,7]:
            disc_action = [cont_action]
        elif cont_action < -0.75:
            disc_action = [0]
        elif cont_action < -0.50:
            disc_action = [1]
        elif cont_action < -0.25:
            disc_action = [2]
        elif cont_action < 0.0:
            disc_action = [3]
        elif cont_action < 0.25:
            disc_action = [4]
        elif cont_action < 0.50:
            disc_action = [5]
        elif cont_action < 0.75:
            disc_action = [6]
        elif cont_action <= 1.0:
            disc_action = [7]
        else:
            print('invalid cont_action {}'.format(cont_action))

        return disc_action

    def decode_action(self, disc_action):
        if disc_action[0] == 0:
            action = [-0.875]
        elif disc_action[0] == 1:
            action = [-0.625]
        elif disc_action[0] == 2:
            action = [-0.375]
        elif disc_action[0] == 3:
            action = [-0.125]
        elif disc_action[0] == 4:
            action = [0.125]
        elif disc_action[0] == 5:
            action = [0.375]
        elif disc_action[0] == 6:
            action = [0.625]
        elif disc_action[0] == 7:
            action = [0.875]
        else:
            print('invalid disc_action {}'.format(disc_action))

        return action

    # Updating Q table using double Q-value
    def update_q(self, state_old, action, reward, state_new, alpha):
        if np.random.binomial(1, 0.5) == 1:
            active_q = self.Q1
            target_q = self.Q2
        else:
            active_q = self.Q2
            target_q = self.Q1

        best_action = np.random.choice([action_ for action_, value_ in enumerate(active_q[state_new]) if value_ == np.max(active_q[state_new])])
        active_q[state_old][action] += alpha * (reward + self.gamma * target_q[state_new][best_action] - active_q[state_old][action])


    def run(self):

        rewards = []
        for e in range(self.episodes):
            plt.ylabel('Episodes')
            # As states are continuous, discretize them into buckets
            current_state = self.discretize(self.env.reset())

            # Get adaptive learning alpha and epsilon decayed over time
            alpha = self.get_alpha(e)
            epsilon = self.get_epsilon(e)
            done = False
            cumReward = 0

            while not done:
                # Render environment
                # self.env.render()

                # Choose action according to greedy policy and take it
                disc_action = self.choose_action(current_state, epsilon)
                cont_action = self.decode_action(disc_action)
                # print('cont_action {} disc_action {}'.format(cont_action, disc_action))
                obs, reward, done, _ = self.env.step(cont_action)
                new_state = self.discretize(obs)

                # Update Q-Table
                self.update_q(current_state, disc_action, reward, new_state, alpha)
                current_state = new_state
                cumReward += reward
            rewards.append(cumReward)
            if (e%20==0):
                print('Episode {} reward {} alpha {} epsilon {}'.format(e, cumReward, alpha, epsilon))

        # plot scores by episode
        smoothing_window = 10
        rewards_smoothed = pd.Series(rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
        plt.plot(rewards_smoothed)
        plt.xlabel("Episode")
        plt.ylabel("Episode Reward (Smoothed)")
        plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))

        plt.savefig('./images/figure_2_4.png')
        plt.close()


# Make an instance of CartPole class
print("Starting Mountain Car Runner")
qrunner = doubleqlearning_mountaincar(episodes=200)
qrunner.run()
print("Mountain Car Runner Finished")
