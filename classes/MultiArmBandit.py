# MAB Constructor

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors


class MultiArmBandit:
    def __init__(self, num_arms, period, optimalArm, numOfFrauds = 0, numOfTrueArms = 0, noNoise=False, times = 0, debug=False):
        self.num_arms = num_arms
        self.possible_actions = np.zeros(num_arms)
        self.period = period
        self.q_a = np.zeros(num_arms)
        self.optimalArm = optimalArm
        self.numOfFrauds = numOfFrauds
        self.noNoise = noNoise
        self.numOfTrueArms = numOfTrueArms
        self.times = times

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
        elif self.numOfTrueArms == 0 and self.numOfFrauds == 0:
            if (index_arm == self.optimalArm) and (optimalAlgPulling):
                r_t = amplitude * np.sin((2 * np.pi / T) * t + phase_shift)
                r_t = abs(r_t)
            elif (index_arm == self.optimalArm):
                r_t = amplitude * np.sin((2 * np.pi / T) * t + phase_shift)
                r_t = abs(r_t)
                if not self.noNoise:
                    r_t += noise
            else:
                r_t = 0.1 * np.sin((2 * np.pi / T) * t + phase_shift)
                r_t = abs(r_t)
                if not self.noNoise:
                    r_t += noise
        elif self.numOfTrueArms != 0:
            trueArms = np.arange(0,self.numOfTrueArms,1)
            if (index_arm in trueArms) and (optimalAlgPulling):
                r_t = amplitude * np.sin((2 * np.pi / T) * t + phase_shift)
                r_t = abs(r_t)
            elif (index_arm in trueArms):
                r_t = amplitude * np.sin((2 * np.pi / T) * t + phase_shift)
                r_t = abs(r_t)
                if not self.noNoise:
                    r_t += noise
            else:
                r_t = 0.1 * np.sin((2 * np.pi / T) * t + phase_shift)
                r_t = -abs(r_t)
                if not self.noNoise:
                    r_t += noise
        return r_t

    def getFlatReward(self):
        reward = []
        for t in range(self.times):
            arm_rewards = [self.pull_arm(i, t) for i in range(self.num_arms)]
            closest_reward = min(arm_rewards, key=lambda x: min(abs(x - 1), abs(x + 1), abs(x)))
            reward.append(closest_reward)
        
        reward = np.array(reward)


        return reward


    def print_actions(self):
        print(f'Possible actions - {self.possible_actions}')

    def plotReward(self, numOfArms, fraudArm, optimalArm, maxSteps):
            #plot the reward function at time t for each arm
        # fig, ax = plt.subplots()

        # #create color palette for the arms, 30 arms, 30 colors, set the color for the optimal arm to orange and color of the arm 20 inverse of the optimal arm use pastel colors
        # colors = plt.cm.get_cmap('Pastel1', numOfArms)(range(numOfArms))

        # # Set the 15th index to orange
        # colors[optimalArm] = matplotlib.colors.to_rgba('orange')

        # # Calculate the inverse of orange
        # orange_inverse = np.array(matplotlib.colors.to_rgba('orange'))
        # orange_inverse[:3] = 1.0 - orange_inverse[:3]  # Invert the RGB components, ignore the alpha

        # # Set the 20th index to the inverse of orange
        # colors[fraudArm] = orange_inverse

        # for i in range(numOfArms):
        reward = []
        for t in range(maxSteps):
            reward.append(self.pull_arm(optimalArm, t, optimalAlgPulling=True))
        #     ax.plot(reward, label=f'Arm {i}',
        #         color = colors[i], linestyle="-", linewidth=2)
        #     if i == optimalArm :
        #         ax.text(maxSteps, reward[-1] * 1.02, f'Optimal Arm',
        #                 color=colors[i], fontweight="normal", horizontalalignment="left", verticalalignment="center")
        #     if i == 20:
        #         ax.text(maxSteps, (reward[-1] - 0.5) * 1.02, f'Fraud Arm',
        #                 color=colors[i], fontweight="normal", horizontalalignment="left", verticalalignment="center")
        #     if i == 21:
        #         ax.text(maxSteps, (reward[-1]- 0.2) * 1.02, f'Zero Reward Arms',
        #                 color=colors[i], fontweight="normal", horizontalalignment="left", verticalalignment="center")
            
        # ax.spines["right"].set_visible(False)
        # ax.spines["left"].set_visible(False)
        # ax.spines["top"].set_visible(False)
        # ax.yaxis.set_ticks_position("left")
        # ax.xaxis.set_ticks_position("bottom")
        # ax.spines["bottom"].set_bounds(0,maxSteps-1)
        # ax.set_ylabel(f'Instantaneous Reward at each arm')
        # ax.set_xlabel(f'Steps')
        # fig.tight_layout()
        # plt.show()

        return reward
