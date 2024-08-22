# New Confidence Bound Algorihm

import numpy as np
import math

class NewConfidenceBound:
    def __init__(self, mab, period, max_steps):
        self.mab = mab
        self.num_arms = mab.num_arms
        self.max_steps = max_steps
        self.period = period
        self.Q_a = np.zeros(self.num_arms)
        self.N_a = np.zeros(self.num_arms)
        self.fourier_coefficients = np.zeros((self.num_arms, 3))  # a0, a1, b1 for each arm
        self.total_reward = 0
        self.reward_t = np.zeros(max_steps)
        self.observed_times = [[] for _ in range(self.num_arms)]  # Initialize as lists of lists
        self.observed_rewards = [[] for _ in range(self.num_arms)]
        self.phase_shifts = np.zeros(self.num_arms)

    def run(self, max_steps, debug=False):
        num_steps = 0
        while num_steps < max_steps:  # Assuming a maximum of 10,000 steps
            arm_to_pull = self.select_action(num_steps)
            current_reward = self.mab.pull_arm(arm_to_pull, num_steps)
            self.total_reward += current_reward
            self.reward_t[num_steps] = current_reward
            if(debug):
                print(f'Current reward: {self.total_reward}')
            self.N_a[arm_to_pull] += 1
            self.Q_a[arm_to_pull] = self.Q_a[arm_to_pull] + \
                (1 / self.N_a[arm_to_pull]) * \
                (current_reward - self.Q_a[arm_to_pull])
            self.update_fourier_coefficients(arm_to_pull, current_reward, num_steps)
            num_steps += 1
        return self.reward_t, self.total_reward

    def select_action(self, t):
        predicted_rewards = []
        for arm in range(self.num_arms):
            a0, a1, b1 = self.fourier_coefficients[arm]
            predicted_reward = a0 + a1 * np.cos(2 * np.pi * t / self.period + self.phase_shifts[arm]) + b1 * np.sin(2 * np.pi * t / self.period + self.phase_shifts[arm])
            predicted_rewards.append(predicted_reward)
        arm_to_pull = np.argmax(predicted_rewards)
        return arm_to_pull

    def update_fourier_coefficients(self, arm, reward, t):
        # Assuming self.fourier_coefficients is already initialized in the class constructor
        # and is intended to store the Fourier coefficients for each arm.
        
        # Check if this is the first time updating coefficients for this arm
        if self.N_a[arm] == 1:
            # Initialize lists to store times and rewards for the arm
            self.observed_times[arm] = [t]
            self.observed_rewards[arm] = [reward]
        else:
            # Append the new time and reward to the lists for the arm
            self.observed_times[arm].append(t)
            self.observed_rewards[arm].append(reward)

        # Convert observed times and rewards for the arm into numpy arrays for calculation
        times = np.array(self.observed_times[arm])
        rewards = np.array(self.observed_rewards[arm])

        # Normalize times by the period to bring them within a single period cycle
        normalized_times = 2 * np.pi * times / (self.period + self.phase_shifts[arm])
        # Calculate Fourier coefficients
        # a0: Average of the rewards (DC component)
        a0 = np.mean(rewards)

        # a1: Cosine coefficient
        a1 = (2 / len(times)) * np.sum(rewards * np.cos(normalized_times))

        # b1: Sine coefficient
        b1 = (2 / len(times)) * np.sum(rewards * np.sin(normalized_times))

        # Update the Fourier coefficients for the arm
        self.fourier_coefficients[arm] = [a0, a1, b1]
        
        if a1 != 0:
            self.phase_shifts[arm] = np.arctan(b1 / a1)
        else:
            # Handle the case where a1 is zero to avoid division by zero
            self.phase_shifts[arm] = 0  # or some other logic as needed
    
    def getParameters(self):
        return self.Q_a, self.N_a, self.total_reward, self.reward_t, self.observed_times, self.observed_rewards

    # Clear vars
    def clear(self):
        self.Q_a = np.zeros(self.num_arms)
        self.N_a = np.zeros(self.num_arms)
        self.T_a = np.ones(self.num_arms)
        self.total_reward = 0
        self.reward_t = np.zeros(self.max_steps)
        self.observed_times = [[] for _ in range(self.num_arms)]
        self.observed_rewards = [[] for _ in range(self.num_arms)]
        self.phase_shifts = np.zeros(self.num_arms)

