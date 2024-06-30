# Thompson Sampling Algorithm

import numpy as np
import math

class ThompsonSampling:
    def __init__(self, maxSteps, mab):
        self.maxSteps = maxSteps
        self.mab = mab
        self.num_arms = mab.num_arms
        self.alpha = np.ones(self.num_arms)
        self.beta = np.ones(self.num_arms)
        self.Q_a = np.zeros(self.num_arms)
        self.N_a = np.zeros(self.num_arms)
        self.total_reward = 0
        self.reward_t = np.zeros(self.maxSteps)
        self.observed_times = [[] for _ in range(self.num_arms)]
        self.observed_rewards = [[] for _ in range(self.num_arms)]
    
    def run(self, maxSteps, debug=False):
        for t in range(maxSteps):
            # Step 1: Sample from the posterior for each arm
            sampled_theta = np.random.beta(self.alpha, self.beta)
            
            # Step 2: Select the arm with the highest sampled value
            arm_selected = self.select_action(sampled_theta)
            
            # Step 3: Pull the selected arm and observe the reward
            reward = self.mab.pull_arm(arm_selected, t)
            self.total_reward += reward
            self.reward_t[t] = reward
            
            # Step 4: Update the posterior distribution for the selected arm
            if reward > 0:
                self.alpha[arm_selected] += 1
            else:
                self.beta[arm_selected] += 1
            
            # Optional: Store observed times and rewards for analysis
            self.observed_times[arm_selected].append(t)
            self.observed_rewards[arm_selected].append(reward)
            
            if debug:
                print(f"Step {t+1}: Arm {arm_selected} selected, reward: {reward}")
            
        return self.reward_t, self.total_reward
    
    def select_action(self, sampled_theta):
        arm_selected = np.argmax(sampled_theta)
        return arm_selected
    
    def getParameters(self):
        return self.Q_a, self.N_a, self.total_reward, self.reward_t, self.observed_times, self.observed_rewards

    # Clear vars
    def clear(self):
        self.alpha = np.ones(self.num_arms)
        self.beta = np.ones(self.num_arms)
        self.Q_a = np.zeros(self.num_arms)
        self.N_a = np.zeros(self.num_arms)
        self.total_reward = 0
        self.reward_t = np.zeros(self.maxSteps)
        self.observed_times = [[] for _ in range(self.num_arms)]
        self.observed_rewards = [[] for _ in range(self.num_arms)]


