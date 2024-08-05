
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
from scipy.signal import find_peaks
import matplotlib.colors

def detect_spikes(data, threshold):
    """
    Detect spikes in the data array based on a specified threshold.
    
    Parameters:
    data (np.array): The input array containing data points.
    threshold (float): The threshold value to detect spikes.
    
    Returns:
    np.array: Indices of the detected spikes.
    """
    # Calculate the difference between consecutive elements
    diff = np.diff(data)
    
    # Identify where the difference exceeds the threshold
    spikes = np.where(np.abs(diff) > threshold)[0]
    
    # Return the indices of the spikes
    return spikes

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
        #TODO: fix here, CAD will take the reward at the air at any time t
        #TODO: ecad will take its selections
        #TODO: plot the power consumption by mapping the reward to 1,0,-1 and convert them to the power consumptions
        numOfArms = 30
        numOfRuns = 1000
        maxSteps = 301
        optimalArm = 15
        period = 20
        fraudArm = 20
        armRewards = np.zeros((numOfArms, maxSteps))
        cadTimes = np.arange(1, maxSteps+1, 1)
        mab = MultiArmBandit(numOfArms, period=period, optimalArm=optimalArm, numOfFrauds=0, noNoise=False, numOfTrueArms=0, times=maxSteps)
        tx = mab.plotReward(numOfArms,fraudArm,optimalArm,maxSteps)
        peaks, _ = find_peaks(tx, height=0.95)

        # Set spikes to 1 and remaining values to 0
        txTimes = np.zeros(maxSteps)
        txTimes[peaks] = 1  # Set spike values to 1
        experiment_runner = ExperimentRunner(num_runs=numOfRuns, max_steps=maxSteps, optimalAlgName='Optimal', figSaveDir="PowerConsumption")
        ECADConsumption = []
        CADConsumption = []
        ncb = NewConfidenceBound(mab=mab, period=period, max_steps=maxSteps)
        experiment_runner.runExperiments_part1(alg=ncb, alg_name=f'ECAD')
        resECAD = experiment_runner.all_results_part1.loc['ECAD']
        resECAD = np.array(resECAD)
        CADConsumption = np.array(mab.getFlatReward())
        fig, ax = plt.subplots()
        plt.plot(CADConsumption)
        plt.plot(resECAD)
        plt.show()

        for i in range(len(CADConsumption)):
            if CADConsumption[i] > 0.99:
                CADConsumption[i] = 1
            if CADConsumption[i] < -0.99:
                CADConsumption[i] = -1

        for i in range(len(resECAD)):
            resECAD[i] = min([1, -1, 0], key=lambda x: abs(resECAD[i] - x))

        for i in range(len(resECAD)):
            if resECAD[i] > 0.99:
                resECAD[i] = 1
        
        reduced_CADConsumption = []
        last_value = None
        for value in CADConsumption:
            if value == -1 and last_value == -1:
                continue
            reduced_CADConsumption.append(value)
            last_value = value

        CADConsumption = np.array(reduced_CADConsumption)

        reduced_resECAD = []
        last_value = None
        for value in resECAD:
            if value == 1 and last_value == 1:
                reduced_resECAD.append(0)
                continue
            reduced_resECAD.append(value)
            last_value = value

        resECAD = np.array(reduced_resECAD)

        for i in range(len(CADConsumption)):
            if CADConsumption[i] == 1:
                # Find the closest index in resECAD that is 1
                closest_index = None
                min_distance = float('inf')
                for j in range(len(resECAD)):
                    if resECAD[j] == 1 and abs(i - j) < min_distance:
                        closest_index = j
                        min_distance = abs(i - j)
                
                if closest_index is not None:
                    # Set the value at the closest index to 1 and the previous index to 0
                    CADConsumption[closest_index] = 1
                    if closest_index > 0:
                        CADConsumption[i] = 0
        
        CADReception = []
        ECADReception = []

        SuccessfulCADReception = []
        SuccessfulECADReception = []
        CADFailedReception = []
        ECADFailedReception = []
        CADActivity = []
        ECADActivity = []
        
        for i in range(len(CADConsumption)):
            if CADConsumption[i] == 1:
                CADConsumption[i] = 1 * (4.6 + 40 + 0.03794)
                CADReception.append(1)
                SuccessfulCADReception.append(1)
                CADFailedReception.append(0)
                CADActivity.append(0)
            elif CADConsumption[i] == -1:
                CADConsumption[i] = 1 * (4.6 + 40 + 0.03794)
                CADReception.append(1)
                SuccessfulCADReception.append(0)
                CADFailedReception.append(1)
                CADActivity.append(0)
            else:
                CADConsumption[i] = 0 + 0.002 + 0.0012 + 0.03794
                CADReception.append(0)
                SuccessfulCADReception.append(0)
                CADFailedReception.append(0)
                CADActivity.append(1)
        for i in range(len(resECAD)):
            if resECAD[i] == 1:
                ECADConsumption.append(1 * (4.6 + 40 + 0.03794))
                ECADReception.append(1)
                SuccessfulECADReception.append(1)
                ECADFailedReception.append(0)
                ECADActivity.append(0)
            elif resECAD[i] == -1:
                ECADConsumption.append(1 * (4.6 + 40 + 0.03794))
                ECADReception.append(1)
                SuccessfulECADReception.append(0)
                ECADFailedReception.append(1)
                ECADActivity.append(0)
            else:
                ECADConsumption.append(0 + 0.002 + 0.0012 + 0.03794)
                ECADReception.append(0)
                SuccessfulECADReception.append(0)
                ECADFailedReception.append(0)
                ECADActivity.append(1)

        # ECADConsumption = np.cumsum(ECADConsumption)
        # CADConsumption = np.cumsum(CADConsumption)

        # ECADReception = np.cumsum(ECADReception)
        # CADReception = np.cumsum(CADReception)
        SuccessfulCADReception = np.cumsum(SuccessfulCADReception)
        SuccessfulECADReception = np.cumsum(SuccessfulECADReception)
        # CADFailedReception = np.cumsum(CADFailedReception)
        # ECADFailedReception = np.cumsum(ECADFailedReception)
        # CADActivity = np.cumsum(CADActivity)
        # ECADActivity = np.cumsum(ECADActivity)
        # txTimes = np.cumsum(txTimes)


        colors = ["#EBA33B", "#D28D46", "#B97852", "#A0635E", "#874E6A",
                  "#6E3976", "#553482", "#3C2E8E", "#23389A", "#0A43A6",
                  "#3BA1EB"
        ]
        # fig, ax = plt.subplots()
        # ax.set_xlabel("Time")
        # ax.set_ylabel("mAh")
        # ax.plot(ECADConsumption, label=f'ECAD',
        #     color = colors[0], linestyle="-", linewidth=2)
        # ax.text(maxSteps, ECADConsumption[-1] * 1.02, f'ECAD',
        #     color=colors[0], fontweight="normal", horizontalalignment="left", verticalalignment="center")
        # ax.plot(CADConsumption, label=f'CAD',
        #     color = colors[1], linestyle="-", linewidth=2)
        # ax.text(maxSteps, CADConsumption[-1] * 1.02, f'CAD',
        #     color=colors[1], fontweight="normal", horizontalalignment="left", verticalalignment="center")

        lineSize = [0.5, 0.5, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
        fig, ax = plt.subplots()

        # ax.eventplot(ECADConsumption, lineoffsets=0, linelengths=lineSize[0], color=colors[0], label='ECAD Power Consumption')
        # ax.eventplot(CADConsumption, lineoffsets=1, linelengths=lineSize[1], color=colors[1], label='CAD Power Consumption')
        # ax.plot(txTimes, color=colors[2], label='Transmission Times')
        # ax.plot(CADReception, color=colors[3], label='CAD Reception')
        # ax.plot(ECADReception, color=colors[4], label='ECAD Reception')
        ax.plot(SuccessfulCADReception, color=colors[5], label='Successful CAD Reception')
        ax.plot(SuccessfulECADReception, color=colors[6], label='Successful ECAD Reception')
        # ax.plot(CADFailedReception, color=colors[7], label='CAD Failed Reception')
        # ax.plot(ECADFailedReception, color=colors[8], label='ECAD Failed Reception')
        # ax.plot(CADActivity, color=colors[9], label='CAD Activity')
        # ax.plot(ECADActivity, color=colors[10], label='ECAD Activity')

        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")
        ax.spines["bottom"].set_bounds(0,maxSteps-1)
        ax.set_ylabel(f'Power Consumption (mAh)')
        ax.set_xlabel(f'Steps')
        fig.tight_layout()
        plt.show()
        #fig.savefig("plots/PowerConsumption.png")


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
        
main(part1=False, part2=False, part3=True, part4=False)