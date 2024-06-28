import numpy as np
import math


class MultiArmBandit:
    def __init__(self, num_arms, period, debug=False):
        self.num_arms = num_arms
        self.possible_actions = np.zeros(num_arms)
        self.period = period
        self.q_a = np.zeros(num_arms)

        # Ini mean values
        for i in np.arange(num_arms):
            self.q_a[i] = np.random.normal(loc=0, scale=math.sqrt(1))

    def pull_arm(self, index_arm, t, debug=False):
        T = self.period
        # Adjust phase shift and amplitude based on arm index for variability
        phase_shift = index_arm * (2 * np.pi / self.num_arms)
        amplitude = 1 + 0.5 * np.sin(2 * np.pi * index_arm / self.num_arms)
        
        # Periodic reward function with adjusted amplitude and phase shift
        r_t = amplitude * np.sin((2 * np.pi / T) * t + phase_shift) #+ self.q_a[index_arm]
        
        # Adjust noise level
        noise = np.random.normal(loc=0, scale=0.1)  # Consider experimenting with the scale
        r_t += noise
        return r_t

    def print_actions(self):
        print(f'Possible actions - {self.possible_actions}')
