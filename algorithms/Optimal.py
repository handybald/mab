# Optimal Algorithm

import numpy as np

class OptimalAlgorithm:
    def __init__(self, optimalArm, maxSteps, mab):
        self.optimalArm = optimalArm
        self.mab = mab
        self.maxSteps = maxSteps
        self.total_reward = 0
        self.reward_t = np.zeros(self.maxSteps)
        self.Q_a = np.zeros(1)
        self.N_a = np.zeros(1)
        self.observed_times = [[]]
        self.observed_rewards = [[]]
    
    def run(self, maxSteps, debug=False):
        for t in range(maxSteps):
            # Pull the optimal arm and observe the reward
            reward = self.mab.pull_arm(self.optimalArm, t, True)
            self.total_reward += reward
            self.reward_t[t] = reward
            self.N_a[0] += 1
            self.Q_a[0] = self.Q_a[0] + (1 / self.N_a[0]) * (reward - self.Q_a[0])
            
            # Optional: Store observed times and rewards for analysis
            self.observed_times[0].append(t)
            self.observed_rewards[0].append(reward)
            
            if debug:
                print(f"Step {t+1}: Reward: {reward}")
        
        return self.reward_t, self.total_reward

    def getParameters(self):
        return self.Q_a, self.N_a, self.total_reward, self.reward_t, self.observed_times, self.observed_rewards
    
    # Clear vars
    def clear(self):
        self.total_reward = 0
        self.reward_t = np.zeros(self.maxSteps)
        self.Q_a = np.zeros(1)
        self.N_a = np.zeros(1)
        self.observed_times = [[]]
        self.observed_rewards = [[]]