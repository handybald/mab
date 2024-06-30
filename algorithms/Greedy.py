# Greedy Algorithm

import numpy as np


class Greedy:
    def __init__(self, Q1, mab):
        self.Q1 = Q1
        self.mab = mab
        self.num_arms = mab.num_arms
        self.Q_a = np.full((self.num_arms), self.Q1)
        self.N_a = np.zeros(self.num_arms)
        self.total_reward = 0
        self.observed_times = [[] for _ in range(self.num_arms)]
        self.observed_rewards = [[] for _ in range(self.num_arms)]

    def run(self, max_steps, debug=False):
        # Store reward at each t
        self.reward_t = np.zeros(max_steps)
        num_steps = 1
        while(num_steps < max_steps):
            arm_to_pull = self.select_action()
            current_reward = self.mab.pull_arm(
                arm_to_pull, num_steps)
            self.total_reward += current_reward
            self.reward_t[num_steps] = current_reward
            if(debug):
                print(f'Current reward : {self.total_reward}')
            self.observed_times[arm_to_pull].append(num_steps)
            self.observed_rewards[arm_to_pull].append(current_reward)
            self.N_a[arm_to_pull] += 1
            self.Q_a[arm_to_pull] = self.Q_a[arm_to_pull] + \
                (1 / self.N_a[arm_to_pull]) * \
                (current_reward - self.Q_a[arm_to_pull])
            num_steps += 1
        return self.reward_t, self.total_reward

    def select_action(self):
        # Exploit
        # Arm with highest reward
        arm_to_pull = np.argmax(self.Q_a)
        return arm_to_pull

    def getParameters(self):
        return self.Q_a, self.N_a, self.total_reward, self.reward_t, self.observed_times, self.observed_rewards

    # Clear vars
    def clear(self):
        self.Q_a = np.full((self.num_arms), self.Q1)
        self.N_a = np.zeros(self.num_arms)
        self.total_reward = 0
        self.observed_times = [[] for _ in range(self.num_arms)]
        self.observed_rewards = [[] for _ in range(self.num_arms)]
