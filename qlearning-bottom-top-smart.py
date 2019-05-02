import itertools
import matplotlib.style
import numpy as np
import pandas as pd
from gymEnvBottonTop import BottomTopEnv
from matplotlib import pyplot as plt
from collections import defaultdict, namedtuple

matplotlib.style.use('ggplot')

# variable used for statistics
EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])

# Create gym environment.
env = BottomTopEnv()

# Make the $\epsilon$-smart policy.
def createEpsilonSmartPolicy(Q, epsilon, num_actions):
    """
    Creates an epsilon-smart policy based on a given Q-function and epsilon.

    Returns a function that takes the state & quantity of times that the agent is in te same state
    as an input and returns the probabilities for each action in the form of a numpy array of
    length of the action space(set of possible actions).
    """
    def policyFunction(state, stateRepeats):
        # set probabilities 1 - ( epsilon * stateRepeats)
        # ej. stateRepeats = 0 => 1 - epsilon
        # ej. stateRepeats = 1 => 1 - (epsilon*2)
        # ej. stateRepeats = 2 => 1 - (epsilon*3)
        epsilonModified = epsilon*(stateRepeats+1)
        # max value of epsilonModified (0.5)
        if (epsilonModified >= 0.5):
            epsilonModified = 0.5
        Action_probabilities = np.ones(num_actions, dtype=float) * epsilonModified / num_actions
        # get best action
        best_action = np.argmax(Q[state])
        Action_probabilities[best_action] += (1.0 - epsilonModified)
        return Action_probabilities

    return policyFunction

# Build Q-Learning Model
def qLearning(env, num_episodes, discount_factor=1.0,
              alpha=0.6, epsilon=0.1):
    """
    Q-Learning algorithm: Off-policy TD control.
    Finds the optimal greedy policy while improving following an epsilon-greedy policy
    """

    # Action value function
    # A nested dictionary that maps
    # state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    print("env.action_space.n %s" % env.action_space.n)

    # Keeps track of useful statistics
    stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # Create an epsilon greedy policy function
    # appropriately for environment action space
    policy = createEpsilonSmartPolicy(Q, epsilon, env.action_space.n)

    # For every episode
    for ith_episode in range(num_episodes):
        print("Episode %s" % ith_episode)
        print(dict(Q))

        # Reset the environment and pick the first action
        state = env.reset()
        # Counter of moves that remain the agent in the same state
        stateRepeats = 0

        for t in itertools.count():
            print("---------------")

            # get probabilities of all actions from current state
            action_probabilities = policy(state, stateRepeats)

            # choose action according to
            # the probability distribution
            action = np.random.choice(np.arange(
                len(action_probabilities)),
                p=action_probabilities)
            print("state " + str(state) + " action " + str(action))

            # take action and get reward, transit to next state
            next_state, reward, done, _ = env.step(action)
            print("next state " + str(next_state) + " reward " + str(reward))

            # Update statistics
            stats.episode_rewards[ith_episode] += reward
            stats.episode_lengths[ith_episode] = t

            # TD Update
            best_next_action = np.argmax(Q[next_state])
            print("next best_next_action " + str(best_next_action))
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            print("td_target " + str(td_target))
            td_delta = td_target - Q[state][action]
            print("td_delta " + str(td_delta))
            Q[state][action] += alpha * td_delta
            print("Q[state][action] " + str(Q[state][action]))

            # episode terminated if env return Done or after 50 movements
            if (done or t == 50):
                break

            # if state unchanged => incremante stateRepeats counter
            # if state changed => reset stateRepeats counter
            if (state == next_state):
                stateRepeats += 1
            else:
                stateRepeats = 0
            state = next_state

    return Q, stats

def plot_episode_stats(stats, smoothing_window=10, noshow=False):
    # Plot the episode reward over time
    fig1 = plt.figure(figsize=(10,5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title("Episode Reward over Time".format(smoothing_window))
    if noshow:
        plt.close(fig1)
    else:
        plt.show(fig1)

    return fig1


# Train the model 100 times with differents discount factors
Q1, stats1 = qLearning(env, 100, discount_factor=1.0)
Q2, stats2 = qLearning(env, 100, discount_factor=0.8)
Q3, stats3 = qLearning(env, 100, discount_factor=0.6)

# plot important statistics
plot_episode_stats(stats1)
plot_episode_stats(stats2)
plot_episode_stats(stats3)

