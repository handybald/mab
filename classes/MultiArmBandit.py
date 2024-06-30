# MAB Constructor

import numpy as np
import math


class MultiArmBandit:
    def __init__(self, num_arms, period, optimalArm, numOfFrauds = 0, debug=False):
        self.num_arms = num_arms
        self.possible_actions = np.zeros(num_arms)
        self.period = period
        self.q_a = np.zeros(num_arms)
        self.optimalArm = optimalArm
        self.numOfFrauds = numOfFrauds

        # Ini mean values
        for i in np.arange(num_arms):
            self.q_a[i] = np.random.normal(loc=0, scale=math.sqrt(1))

    def pull_arm(self, index_arm, t, optimalAlgPulling=False, debug=False):
        T = self.period
        # Adjust phase shift and amplitude based on arm index for variability
        phase_shift = index_arm * (2 * np.pi / self.num_arms)
        amplitude = 1 #+ 0.5 * np.sin(2 * np.pi * index_arm / self.num_arms)
        noise = np.random.normal(loc=0, scale=0.1)  # Consider experimenting with the scale

        if self.numOfFrauds != 0:
            fraudArms = np.arange(0,self.numOfFrauds+1,1)
            zeroArms = np.arange(max(fraudArms)+1,self.num_arms,1)
            if self.optimalArm in zeroArms:
                zeroArms = np.delete(zeroArms, np.where(zeroArms == self.optimalArm))
            if self.optimalArm in fraudArms:
                fraudArms = np.delete(fraudArms, np.where(fraudArms == self.optimalArm))
            if (index_arm == self.optimalArm) and (optimalAlgPulling):
                r_t = amplitude * np.sin((2 * np.pi / T) * t + phase_shift)
                r_t = abs(r_t)
            elif (index_arm == self.optimalArm):
                r_t = amplitude * np.sin((2 * np.pi / T) * t + phase_shift)
                r_t = abs(r_t)
                r_t += noise
            elif index_arm in fraudArms:
                r_t = amplitude * np.sin((2 * np.pi / T) * t + phase_shift)
                r_t = -abs(r_t)
                r_t += noise
            elif index_arm in zeroArms:
                r_t = 0.1 * np.sin((2 * np.pi / T) * t + phase_shift)
                r_t = -abs(r_t)
                r_t += noise
        else:
            if (index_arm == self.optimalArm) and (optimalAlgPulling):
                r_t = amplitude * np.sin((2 * np.pi / T) * t + phase_shift)
                r_t = abs(r_t)
            elif (index_arm == self.optimalArm):
                r_t = amplitude * np.sin((2 * np.pi / T) * t + phase_shift)
                r_t = abs(r_t)
                r_t += noise
            elif index_arm % 2 == 0:
                r_t = amplitude * np.sin((2 * np.pi / T) * t + phase_shift)
                r_t = -abs(r_t)
                r_t += noise
            else:
                r_t = 0.1 * np.sin((2 * np.pi / T) * t + phase_shift)
                r_t = -abs(r_t)
                r_t += noise

        
        

        return r_t

    def print_actions(self):
        print(f'Possible actions - {self.possible_actions}')
