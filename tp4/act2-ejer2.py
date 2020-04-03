import gym
import numpy as np
import math
from collections import deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

class doubleqlearning_cart_pole():
    def __init__(self, episodes=500, min_alpha=0.1, min_epsilon=0.1, gamma=1.0):
        # down-scaling feature space to discrete range
        self.buckets = (1, 1, 6, 12,)
        # training episodes
        self.episodes = episodes
        # minimun learning rate
        self.min_alpha = min_alpha
        # minimun exploration rate
        self.min_epsilon = min_epsilon
        # discount factor
        self.gamma = gamma
        # set environment
        self.env = gym.make('CartPole-v1')

        # initialising double Q-table
        self.Q1 = np.zeros(self.buckets + (self.env.action_space.n,))
        self.Q2 = np.zeros(self.buckets + (self.env.action_space.n,))

    # Discretize input space to make Q-table
    def discretize(self, obs):
        upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], math.radians(50)]
        lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2], -math.radians(50)]
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
        return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax([item1 + item2 for item1, item2 in zip(self.Q1[state], self.Q2[state])])

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
            i = 0

            while not done:
                # Render environment
                self.env.render()

                # Choose action according to greedy policy and take it
                action = self.choose_action(current_state, epsilon)
                obs, reward, done, _ = self.env.step(action)
                new_state = self.discretize(obs)

                # Update Q-Table
                self.update_q(current_state, action, reward, new_state, alpha)
                current_state = new_state
                i += 1
            rewards.append(i)
            if (e%20==0):
                print('Episode {} score: {} alpha {} epsilon {}'.format(e, i, alpha, epsilon))

        # plot scores by episode
        smoothing_window = 10
        rewards_smoothed = pd.Series(rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
        plt.plot(rewards_smoothed)
        plt.xlabel("Episode")
        plt.ylabel("Episode Reward (Smoothed)")
        plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))

        plt.savefig('./images/figure_2_2.png')
        plt.close()


# Make an instance of CartPole class
print("Starting Cart Pole Runner")
qrunner = doubleqlearning_cart_pole(episodes=200)
qrunner.run()
print("Cart Pole Runner Finished")
