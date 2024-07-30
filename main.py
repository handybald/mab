
from algorithms.Egreedy import Egreedy
from algorithms.Greedy import Greedy
from algorithms.UpperConfidenceBound import UpperConfidenceBound
from algorithms.NewConfidenceBound import NewConfidenceBound
from algorithms.Optimal import OptimalAlgorithm
from algorithms.ThompsonSampling import ThompsonSampling
from classes.MultiArmBandit import MultiArmBandit
from helpers.ExperimentRunner import ExperimentRunner
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.colors


def main(part1, part2, part3, part4):
    np.random.seed(42)
    

    #mab.plotReward(numOfArms,fraudArm,optimalArm,maxSteps)
    if part1 == True:
        numOfArms = 30
        numOfRuns = 1000
        maxSteps = 101
        optimalArm = 15
        period = 30
        fraudArm = 20
        experiment_runner = ExperimentRunner(num_runs=numOfRuns, max_steps=maxSteps, optimalAlgName='Optimal', figSaveDir="30Arm_15thOptArm_Period30_Steps101_Run100_01ScaleNoise_mod2Fraud_01AmpNoiseReward")
        mab = MultiArmBandit(numOfArms, period=period, optimalArm=optimalArm, numOfFrauds=0, numOfTrueArms=0)
        new_bound_conf = NewConfidenceBound(mab=mab, period=period, max_steps=maxSteps)
        experiment_runner.runExperiments_part1(alg=new_bound_conf, alg_name=f'ECAD')
        ucb_bound_conf = UpperConfidenceBound(c=2, mab=mab, max_steps=maxSteps)
        experiment_runner.runExperiments_part1(alg=ucb_bound_conf, alg_name=f'UCB')
        optimalArg = OptimalAlgorithm(optimalArm, maxSteps, mab=mab)
        experiment_runner.runExperiments_part1(alg=optimalArg, alg_name=f'Optimal')
        q1 = 1
        greedy_optimistic_ini = Greedy(Q1=q1, mab=mab)
        experiment_runner.runExperiments_part1(alg=greedy_optimistic_ini, alg_name=f'Greedy')
        eps = 0.1
        e_greedy = Egreedy(eps=eps, mab=mab)
        experiment_runner.runExperiments_part1(alg=e_greedy, alg_name=f"$\epsilon$-Greedy")
        thompson_sampling = ThompsonSampling(maxSteps, mab)
        experiment_runner.runExperiments_part1(alg=thompson_sampling, alg_name=f'Thompson Sampling')

        experiment_runner.plot_part1()
        experiment_runner.plotInstantaneousRegret()
        experiment_runner.plotCumulativeRegret()

    # Part 2 - Performance of the algorithms on increasing fraud rewards

    if part2 == True:
        numOfArms = 30
        numOfRuns = 1000
        maxSteps = 101
        optimalArm = 15
        period = 30
        experiment_runner = ExperimentRunner(num_runs=numOfRuns, max_steps=maxSteps, optimalAlgName='Optimal', figSaveDir="Fraud")
        mab = MultiArmBandit(numOfArms, period=period, optimalArm=optimalArm, numOfFrauds=0, numOfTrueArms=0)

        optimalArg = OptimalAlgorithm(optimalArm, maxSteps, mab=mab)
        experiment_runner.runExperiments_part1(alg=optimalArg, alg_name=f'Optimal')
        fraudNum = np.arange(1, 31, 1)
        for fraud in fraudNum: 
            mab = MultiArmBandit(numOfArms, period=period, optimalArm=optimalArm, numOfFrauds=fraud, numOfTrueArms=0)
            new_bound_conf = NewConfidenceBound(mab=mab, period=period, max_steps=maxSteps)
            experiment_runner.runExperiments_part1(alg=new_bound_conf, alg_name=f'ECAD{fraud}')
            ucb_bound_conf = UpperConfidenceBound(c=2, mab=mab, max_steps=maxSteps)
            experiment_runner.runExperiments_part1(alg=ucb_bound_conf, alg_name=f'UCB{fraud}')
            q1 = 1
            greedy_optimistic_ini = Greedy(Q1=q1, mab=mab)
            experiment_runner.runExperiments_part1(alg=greedy_optimistic_ini, alg_name=f'QGreedy{fraud}')
            eps = 0.1
            e_greedy = Egreedy(eps=eps, mab=mab)
            experiment_runner.runExperiments_part1(alg=e_greedy, alg_name=f'$\epsilon$-Greedy{fraud}')
            thompson_sampling = ThompsonSampling(maxSteps, mab)
            experiment_runner.runExperiments_part1(alg=thompson_sampling, alg_name=f'TS{fraud}')

        
        experiment_runner.plotFraud(30)
        experiment_runner.plotRegretFraud(30)
    
    if part3 == True:
        numOfArms = 30
        numOfRuns = 1000
        maxSteps = 101
        optimalArm = 15
        period = 30
        fraudArm = 20
        armRewards = np.zeros((numOfArms, maxSteps))
        cadTimes = np.arange(1, maxSteps+1, 1)
        mab = MultiArmBandit(numOfArms, period=period, optimalArm=optimalArm, numOfFrauds=0, noNoise=True, numOfTrueArms=0)
        for i in range(numOfArms):
            reward = []
            for t in range(maxSteps):
                reward.append(mab.pull_arm(i, t))
            armRewards[i,:] = np.array(reward)
        
        # set the armRewards array element to 0 if the reward is bigger than -0.5 and smaller than 0.5
        for i in range(numOfArms):
            for t in range(maxSteps):
                if armRewards[i,t] > -0.99 and armRewards[i,t] < 0.99:
                    armRewards[i,t] = 0
                elif armRewards[i,t] > 0.99:
                    armRewards[i,t] = 1
                else:
                    armRewards[i,t] = -1

                if t % 2 == 0:
                    armRewards[i,t] = 0
        
        # plot the arm rewards
        fig, ax = plt.subplots()
        ax.set_title("Arm Rewards")
        ax.set_xlabel("Time")
        ax.set_ylabel("Arm")
        cax = ax.matshow(armRewards, cmap='coolwarm', aspect='auto')
        fig.colorbar(cax)
        plt.show()

        worConsumption = 0.091460267

        cumWorConsumptionForECAD = np.zeros(maxSteps)
        cumWorConsumptionForCAD = np.zeros(maxSteps)

        for i in range(maxSteps):
            if armRewards[optimalArm,i] == 1:
                cumWorConsumptionForECAD[i] = worConsumption
                cumWorConsumptionForCAD[i] = worConsumption
            if armRewards[fraudArm,i] == -1:
                cumWorConsumptionForCAD[i] = worConsumption
            
        for i in range(1, maxSteps):
            cumWorConsumptionForECAD[i] = cumWorConsumptionForECAD[i-1] + cumWorConsumptionForECAD[i]
            cumWorConsumptionForCAD[i] = cumWorConsumptionForCAD[i-1] + cumWorConsumptionForCAD[i]
        
        colors = ["#EBA33B", "#507A99"]
        fig, ax = plt.subplots()
        ax.set_xlabel("Time")
        ax.set_ylabel("mAh")
        ax.plot(cumWorConsumptionForECAD, label=f'ECAD',
            color = colors[0], linestyle="-", linewidth=2)
        ax.text(maxSteps, cumWorConsumptionForECAD[-1] * 1.02, f'ECAD',
            color=colors[0], fontweight="normal", horizontalalignment="left", verticalalignment="center")
        ax.plot(cumWorConsumptionForCAD, label=f'CAD',
            color = colors[1], linestyle="-", linewidth=2)
        ax.text(maxSteps, cumWorConsumptionForCAD[-1] * 1.02, f'CAD',
            color=colors[1], fontweight="normal", horizontalalignment="left", verticalalignment="center")
        
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")
        ax.spines["bottom"].set_bounds(0,maxSteps-1)
        ax.set_ylabel(f'Power Consumption (mAh)')
        ax.set_xlabel(f'Steps')
        fig.tight_layout()
        fig.savefig("plots/PowerConsumption.png")


    if part4==True:
        numOfArms = 30
        numOfRuns = 10000
        maxSteps = 101
        optimalArm = 15
        period = 30

        numOfArms = [10, 20, 30, 40, 50, 60, 70, 90, 100]
        periods =   [10, 20, 30, 40, 50, 60, 70, 90, 100]
        optimalArms = [random.randint(0, numOfArms[i]-1) for i in range(len(numOfArms))]

        #create experiment runner
        experiment_runner = ExperimentRunner(num_runs=numOfRuns, max_steps=maxSteps, optimalAlgName='Optimal', figSaveDir="IncreasingNumOfArms")

        mab = MultiArmBandit(30, period=30, optimalArm=optimalArm, numOfFrauds=0, numOfTrueArms=0)
        optimalArg = OptimalAlgorithm(optimalArm, maxSteps, mab=mab)
        experiment_runner.runExperiments_part1(alg=optimalArg, alg_name=f'Optimal')

        for i in range(len(numOfArms)):
            mab = MultiArmBandit(numOfArms[i], period=periods[i], optimalArm=optimalArms[i], numOfFrauds=0, numOfTrueArms=0)
            new_bound_conf = NewConfidenceBound(mab=mab, period=periods[i], max_steps=maxSteps)
            experiment_runner.runExperiments_part1(alg=new_bound_conf, alg_name=f'ECAD{numOfArms[i]}')
            eps = 0.1
            e_greedy = Egreedy(eps=eps, mab=mab)
            experiment_runner.runExperiments_part1(alg=e_greedy, alg_name=f'$\epsilon$-Greedy{numOfArms[i]}')
        
        experiment_runner.plotIncreasingNumOfArmsReward(numOfArms[-1], 2*len(numOfArms), numOfArms)
        experiment_runner.plotIncreasingNumOfArmsRegret(numOfArms[-1], 2*len(numOfArms), numOfArms)
        
main(part1=True, part2=True, part3=True, part4=True)