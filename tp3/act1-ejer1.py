import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

matplotlib.use('Agg')

class Bandit:
    # @k_arm: # of arms
    # @epsilon: probability for exploration in epsilon-greedy algorithm
    # @initial: initial estimation for each action
    # @step_size: constant step size for updating estimations
    # @sample_averages: if True, use sample averages to update estimations instead of constant step size
    # @epsilon_variable: if True, use variable epsilon as the steps increase
    # @eps_1: if epsilon_variable, epsilon value for 1st qarter of steps
    # @eps_2: if epsilon_variable, epsilon value for 2nd qarter of steps
    # @eps_3: if epsilon_variable, epsilon value for 3rd qarter of steps
    # @eps_4: if epsilon_variable, epsilon value for 4rt qarter of steps
    def __init__(self, k_arm=10, epsilon=0., initial=0., step_size=0.1, sample_averages=False, true_reward=0.,
                 epsilon_variable=False, eps_1=0, eps_2=0, eps_3=0, eps_4=0):
        self.k = k_arm
        self.step_size = step_size
        self.sample_averages = sample_averages
        self.indices = np.arange(self.k)
        self.time = 0
        self.average_reward = 0
        self.true_reward = true_reward
        self.epsilon = epsilon
        self.initial = initial
        self.epsilon_variable = epsilon_variable
        self.eps_1 = eps_1
        self.eps_2 = eps_2
        self.eps_3 = eps_3
        self.eps_4 = eps_4

    def reset(self):
        # real reward for each action
        self.q_true = np.random.randn(self.k) + self.true_reward

        # estimation for each action
        self.q_estimation = np.zeros(self.k) + self.initial

        # # of chosen times for each action
        self.action_count = np.zeros(self.k)

        self.best_action = np.argmax(self.q_true)

    # get an action for this bandit
    def act(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.indices)

        q_best = np.max(self.q_estimation)
        return np.random.choice([action for action, q in enumerate(self.q_estimation) if q == q_best])

    # take an action, update estimation for this action
    def step(self, action):
        # generate the reward under N(real reward, 1)
        reward = np.random.randn() + self.q_true[action]
        self.time += 1
        self.action_count[action] += 1
        self.average_reward += (reward - self.average_reward) / self.time

        if self.sample_averages:
            # update estimation using sample averages
            self.q_estimation[action] += (reward - self.q_estimation[action]) / self.action_count[action]
        else:
            # update estimation with constant step size
            self.q_estimation[action] += self.step_size * (reward - self.q_estimation[action])
        return reward


def simulate(runs, time, bandits):
    best_action_counts = np.zeros((len(bandits), runs, time))
    rewards = np.zeros(best_action_counts.shape)
    for i, bandit in enumerate(bandits):
        for r in trange(runs):
            bandit.reset()
            for t in range(time):
                if bandit.epsilon_variable:
                    if time == 250:
                        bandit.epsilon = bandit.eps_2
                    elif time == 500:
                        bandit.epsilon = bandit.eps_3
                    elif time == 750:
                        bandit.epsilon = bandit.eps_4
                action = bandit.act()
                reward = bandit.step(action)
                rewards[i, r, t] = reward
                if action == bandit.best_action:
                    best_action_counts[i, r, t] = 1
    mean_best_action_counts = best_action_counts.mean(axis=1)
    mean_rewards = rewards.mean(axis=1)
    return mean_best_action_counts, mean_rewards


def generate_reward_distrib():
    plt.violinplot(dataset=np.random.randn(200, 10) + np.random.randn(10))
    plt.xlabel("Action")
    plt.ylabel("Reward distribution")
    plt.savefig('./images/fig1_1_reward_distrib.png')
    plt.close()


def run_comparison(runs=2000, time=1000):
    epsilons = ['0.1', '0.01', 'variable']
    bandits = [Bandit(epsilon=0.1, sample_averages=True),
               Bandit(epsilon=0.01, sample_averages=True),
               Bandit(epsilon=0.1, sample_averages=True, epsilon_variable=True, eps_1=0.1, eps_2=0.05, eps_3=0.01, eps_4=0.005)]
    best_action_counts, rewards = simulate(runs, time, bandits)

    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    for eps, rewards in zip(epsilons, rewards):
        plt.plot(rewards, label='epsilon = ' + eps)
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    for eps, counts in zip(epsilons, best_action_counts):
        plt.plot(counts, label='epsilon = ' + eps)
    plt.xlabel('steps')
    plt.ylabel('% optimal action')
    plt.legend()

    plt.savefig('./images/fig1_1_comparison.png')
    plt.close()

if __name__ == '__main__':
    generate_reward_distrib()
    run_comparison()
