
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
from scipy.signal import find_peaks

def find_peaks_custom(arr):
    peaks = []
    peak_indices = []

    for i in range(1, len(arr) - 1):
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1] and arr[i] > 0.95:
            peaks.append(arr[i])
            peak_indices.append(i)
        elif arr[i] < arr[i - 1] and arr[i] < arr[i + 1] and arr[i] < -0.95:
            peaks.append(arr[i])
            peak_indices.append(i)    

    return peaks, peak_indices

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
        mab = MultiArmBandit(numOfArms, period=period, optimalArm=optimalArm, numOfFrauds=1, numOfTrueArms=0, noNoise=True)
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
        period = 25
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
        maxSteps = 301
        optimalArm = 15
        period = 25
        mab = MultiArmBandit(numOfArms, period=period, optimalArm=optimalArm, numOfFrauds=0, noNoise=False, numOfTrueArms=1, times=maxSteps)
        experiment_runner = ExperimentRunner(num_runs=numOfRuns, max_steps=maxSteps, optimalAlgName='Optimal', figSaveDir="PowerConsumption")
        ECADConsumption = []
        CADConsumption = []
        ncb = NewConfidenceBound(mab=mab, period=period, max_steps=maxSteps)
        experiment_runner.runExperiments_part1(alg=ncb, alg_name=f'ECAD')
        resECAD = experiment_runner.all_results_part1.loc['ECAD']
        resECAD = np.array(resECAD)
        CADConsumption = np.array(mab.getFlatReward())

        _, peak_indices = find_peaks_custom(resECAD)
        _, cad_peak_indices = find_peaks_custom(CADConsumption)

        peakCAD = np.zeros(len(CADConsumption))
        peakECAD = np.zeros(len(resECAD))
        peakCAD[cad_peak_indices] = 1
        peakECAD[peak_indices] = 1

        # for i in range (len(resECAD)):
        #     if i in peak_indices and resECAD[i] > 0.85:
        #         resECAD[i] = 1
        #     else:
        #         resECAD[i] = 0        
        
        # for i in range (len(CADConsumption)):
        #     if  CADConsumption[i] > 0.55:
        #         CADConsumption[i] = 1
        #     elif CADConsumption[i] < -0.55:
        #         CADConsumption[i] = -1
        #     else:
        #         CADConsumption[i] = 0
        

        reduced_resECAD = []
        last_value = None
        for value in peakECAD:
            if value == 1 and last_value == 1:
                reduced_resECAD.append(0)
                continue
            reduced_resECAD.append(value)
            last_value = value

        peakECAD = np.array(reduced_resECAD)

        for i in range(len(peakCAD)):
            if peakCAD[i] == 1:
                # Find the closest index in resECAD that is 1
                closest_index = None
                min_distance = float('inf')
                for j in range(len(resECAD)):
                    if peakECAD[j] == 1 and abs(i - j) < min_distance:
                        closest_index = j
                        min_distance = abs(i - j)
                
                if closest_index is not None:
                    # Set the value at the closest index to 1 and the previous index to 0
                    peakCAD[closest_index] = 1
                    if closest_index > 0:
                        peakCAD[i] = 0



        ECADRx = np.insert(peakECAD, 0, 0)
        CADRx = np.insert(peakCAD, 0, 0)

        CADConsumption = peakCAD
        ECADConsumption = peakECAD

        for i in range(len(CADConsumption)):
            if CADConsumption[i] == 1:
                CADConsumption[i] = 1 * (4.6 + 40 + 0.03794)
            elif CADConsumption[i] == -1:
                CADConsumption[i] = 1 * (4.6 + 40 + 0.03794)
            else:
                CADConsumption[i] = 0 + 0.002 + 0.0012 + 0.03794
        for i in range(len(resECAD)):
            if ECADConsumption[i] == 1:
                ECADConsumption[i] = 1 * (4.6 + 40 + 0.03794)
            elif ECADConsumption[i] == -1:
                ECADConsumption[i] = 1 * (4.6 + 40 + 0.03794)
            else:
                ECADConsumption[i] = 0 + 0.002 + 0.0012 + 0.03794
        
        ECADConsumption = np.cumsum(ECADConsumption)
        CADConsumption = np.cumsum(CADConsumption)
        ECADRx = np.cumsum(ECADRx)
        CADRx = np.cumsum(CADRx)
        colors_consumption = ["#EBA33B", "#507A99"]
        colors_rx = ["#FF5733", "#33FF57"]

        fig, ax1 = plt.subplots()

        # Plot ECADConsumption and CADConsumption on the primary y-axis
        ax1.plot(ECADConsumption, label='ECAD',
            color = colors_consumption[0], linestyle="-", linewidth=3)
        ax1.text(maxSteps, ECADConsumption[-1] * 1.01, 'ECAD',
            color=colors_consumption[0], fontweight="normal", horizontalalignment="left", verticalalignment="center")
        ax1.plot(CADConsumption, label='CAD',
            color = colors_consumption[1], linestyle="-", linewidth=3)
        ax1.text(maxSteps, CADConsumption[-1] * 1.1, 'CAD',
            color=colors_consumption[1], fontweight="normal", horizontalalignment="left", verticalalignment="center")
        ax1.spines["right"].set_visible(False)
        ax1.spines["left"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        ax1.yaxis.set_ticks_position("left")
        ax1.xaxis.set_ticks_position("bottom")
        ax1.spines["bottom"].set_bounds(0, maxSteps)
        ax1.set_xlabel("Time Slot")
        ax1.set_ylabel("Power Consumption (mAh)", color=colors_consumption[0])
        ax1.tick_params(axis='y', labelcolor=colors_consumption[0])

        # Create a secondary y-axis
        ax2 = ax1.twinx()
        ax2.set_ylabel("Number of Packets Received")
        ax2.plot(ECADRx, label='ECAD',
            color = colors_rx[0], linestyle="-", linewidth=1)
        ax2.plot(CADRx, label='CAD',
            color = colors_rx[1], linestyle="-", linewidth=1)
        ax2.spines["right"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.yaxis.set_ticks_position("left")
        ax2.xaxis.set_ticks_position("bottom")
        ax2.spines["bottom"].set_bounds(0, maxSteps)
        ax1.set_xlabel("Time Slot")

        fig.tight_layout()
        #plt.show()
        plt.savefig("plots/PowerConsumption.png")


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
        
main(part1=True, part2=False, part3=False, part4=False)