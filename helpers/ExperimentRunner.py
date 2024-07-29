# Experiment Runner

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


class ExperimentRunner:
    def __init__(self, num_runs, max_steps, optimalAlgName, figSaveDir):
        self.all_results_part1 = pd.DataFrame()
        self.all_results_part2 = pd.DataFrame()
        self.fraudResults = pd.DataFrame()
        self.num_runs = num_runs
        self.max_steps = max_steps
        self.optimalAlgName = optimalAlgName
        self.figSaveDir = figSaveDir

    def runExperiments_part1(self, alg, alg_name=""):
        results_t_list = pd.DataFrame()
        for i in np.arange(self.num_runs):
            results_t, total_results = alg.run(self.max_steps)
            results_t_list = pd.concat(
                [results_t_list, pd.DataFrame(results_t)], axis=1)
            alg.clear()
        mean_per_timestep = results_t_list.mean(axis=1)
        mean_per_timestep = mean_per_timestep.reset_index().T.drop('index')
        mean_per_timestep['alg_name'] = alg_name
        mean_per_timestep = mean_per_timestep.set_index('alg_name')
        self.all_results_part1 = pd.concat(
            [self.all_results_part1, pd.DataFrame(mean_per_timestep)], axis=0)
        
    
    def compareWithCADandPlot(self):
        cadScanTimes = np.arange(1, self.max_steps, 1)
        

        

    

    def plot_part1(self):
        fig, ax = plt.subplots(
        )  # This sets the figure size to 6 inches wide by 5 inches high
        ax.plot(self.all_results_part1.loc[self.optimalAlgName], label=self.optimalAlgName,
                color="#706759", linestyle="--", linewidth=1)
        ax.text(self.max_steps, self.all_results_part1.loc[self.optimalAlgName].iloc[-1]*1.1, self.optimalAlgName,
                color="#706759", fontweight="bold", horizontalalignment="left", verticalalignment="center")
        colors = ["#EBA33B", "#507A99", "#AB8855", "#3BA1EB"]
        #remove optimal from the list
        dropped = self.all_results_part1.drop(self.optimalAlgName)
        for i in range(4):
            ax.plot(dropped.iloc[i], label=dropped.index[i],
                    color = colors[i], linestyle="-", linewidth=2)
            ax.text(self.max_steps, dropped.iloc[i].iloc[-1] * 1.02, dropped.index[i],
                    color=colors[i], fontweight="normal", horizontalalignment="left", verticalalignment="center")
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")
        ax.spines["bottom"].set_bounds(0,self.all_results_part1.shape[1])
        ax.set_ylabel(f'Expected reward over {self.num_runs} runs')
        ax.set_xlabel(f'Steps')
        fig.tight_layout()
        #create the directory if it does not exist
        directory = "plots/" + self.figSaveDir
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig("plots/" + self.figSaveDir + "/ExpectedReward.png", dpi=300)
        

    def plotInstantaneousRegret(self):
        fig, ax = plt.subplots()
        colors = ["#EBA33B", "#507A99", "#AB8855", "#3BA1EB"]
        regret = pd.DataFrame()
        regret = self.all_results_part1.drop(self.optimalAlgName)
        regret = self.all_results_part1.loc[self.optimalAlgName] - regret
        
        for i in range(4):
            ax.plot(regret.iloc[i], label=regret.index[i],
                    color = colors[i], linestyle="-", linewidth=2)
            ax.text(self.max_steps, regret.iloc[i].iloc[-1] * 1.02, regret.index[i],
                    color=colors[i], fontweight="normal", horizontalalignment="left", verticalalignment="center")
        
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")
        ax.spines["bottom"].set_bounds(0,regret.shape[1])
        ax.set_ylabel(f'Instantaneous Regret')
        ax.set_xlabel(f'Steps')
        fig.tight_layout()
        plt.savefig("plots/" + self.figSaveDir + "/InstantaneousRegret.png", dpi=300)

    def plotCumulativeRegret(self):
        fig, ax = plt.subplots()
        colors = ["#EBA33B", "#507A99", "#AB8855", "#3BA1EB"]
        regret = pd.DataFrame()
        regret = self.all_results_part1.drop(self.optimalAlgName)
        regret = self.all_results_part1.loc[self.optimalAlgName] - regret
        for i in range(4):
            regret.iloc[i] = regret.iloc[i].cumsum()
            ax.plot(regret.iloc[i], label=regret.index[i],
                    color = colors[i], linestyle="-", linewidth=2)
            ax.text(self.max_steps, regret.iloc[i].iloc[-1] * 1.02, regret.index[i],
                    color=colors[i], fontweight="normal", horizontalalignment="left", verticalalignment="center")
        
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")
        ax.spines["bottom"].set_bounds(0,regret.shape[1])
        ax.set_ylabel(f'Instantaneous Regret')
        ax.set_xlabel(f'Steps')
        fig.tight_layout()
        plt.savefig("plots/" + self.figSaveDir + "/CumulativeRegret.png", dpi=300)
    
    def plotFraud(self, numFraud):
        fig, ax = plt.subplots(
        )
        optSum = self.all_results_part1.loc[self.optimalAlgName].sum()
        ax.axhline(y=optSum, color="#706759", linestyle="--", linewidth=1)
        ax.text(numFraud*1.07, optSum*1.01, self.optimalAlgName,
                color="#706759", fontweight="bold", horizontalalignment="left", verticalalignment="center")
        colors = ["#EBA33B", "#507A99", "#AB8855", "#3BA1EB", "#AB8855"]
        #remove optimal from the list
        dropped = self.all_results_part1.drop(self.optimalAlgName)
        
        # find the cumsum of each reward array
        cumsumArrayNCB = []
        cumsumArrayUCB = []
        cumsumArrayGreedy = []
        cumsumArrayEpsGreedy = []
        cumsumArrayTS = []
        for i in range(numFraud*5):
            if dropped.index[i].find("ECAD") != -1:
                cumsumArrayNCB.append(dropped.iloc[i].sum())
            elif dropped.index[i].find("UCB") != -1:
                cumsumArrayUCB.append(dropped.iloc[i].sum())
            elif dropped.index[i].find("QGreedy") != -1:
                cumsumArrayGreedy.append(dropped.iloc[i].sum())
            elif dropped.index[i].find(f"$\epsilon$-Greedy") != -1:
                cumsumArrayEpsGreedy.append(dropped.iloc[i].sum())
            elif dropped.index[i].find("TS") != -1:
                cumsumArrayTS.append(dropped.iloc[i].sum())
        
        ax.plot(cumsumArrayNCB, label="ECAD",
            color = colors[0], linestyle="-", linewidth=2)
        ax.text(numFraud, cumsumArrayNCB[-1] - 0.01, "ECAD",
            color=colors[0], fontweight="normal", horizontalalignment="left", verticalalignment="center")
        ax.plot(cumsumArrayUCB, label="UCB",
            color = colors[1], linestyle="-", linewidth=2)
        ax.text(numFraud, cumsumArrayUCB[-1] * 1.01, "UCB",
            color=colors[1], fontweight="normal", horizontalalignment="left", verticalalignment="center")
        ax.plot(cumsumArrayGreedy, label="Greedy",
            color = colors[2], linestyle="-", linewidth=2)
        ax.text(numFraud, cumsumArrayGreedy[-1] * 1.01, "Greedy",
            color=colors[2], fontweight="normal", horizontalalignment="left", verticalalignment="center")
        ax.plot(cumsumArrayEpsGreedy, label="$\epsilon$-Greedy",
            color = colors[3], linestyle="-", linewidth=2)
        ax.text(numFraud, cumsumArrayEpsGreedy[-1] * 1.01, "$\epsilon$-Greedy",
            color=colors[3], fontweight="normal", horizontalalignment="left", verticalalignment="center")
        ax.plot(cumsumArrayTS, label="TS",
            color = colors[4], linestyle="-", linewidth=2)
        ax.text(numFraud, cumsumArrayTS[-1] * 1.01, "TS",
            color=colors[4], fontweight="normal", horizontalalignment="left", verticalalignment="center")
        
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")
        ax.spines["bottom"].set_bounds(0,numFraud)
        ax.set_ylabel(f'Total Reward over {self.num_runs} runs')
        ax.set_xlabel(f'Number of Fraud End Devices')
        fig.tight_layout()
        #create the directory if it does not exist
        directory = "plots/" + self.figSaveDir
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig("plots/" + self.figSaveDir + "/TotalReward.png", dpi=300)

    
    def plotRegretFraud(self, numFraud):
        fig, ax = plt.subplots(
        )
        colors = ["#EBA33B", "#507A99", "#AB8855", "#3BA1EB", "#AB8855"]
        #remove optimal from the list
        dropped = self.all_results_part1.drop(self.optimalAlgName)
        
        # find the cumsum of each reward array
        cumsumArrayNCB = []
        cumsumArrayUCB = []
        cumsumArrayGreedy = []
        cumsumArrayEpsGreedy = []
        cumsumArrayTS = []
        for i in range(numFraud*5):
            if dropped.index[i].find("ECAD") != -1:
                cumsumArrayNCB.append((self.all_results_part1.loc[self.optimalAlgName] - dropped.iloc[i]).sum())
            elif dropped.index[i].find("UCB") != -1:
                cumsumArrayUCB.append((self.all_results_part1.loc[self.optimalAlgName] - dropped.iloc[i]).sum())
            elif dropped.index[i].find("QGreedy") != -1:
                cumsumArrayGreedy.append((self.all_results_part1.loc[self.optimalAlgName] - dropped.iloc[i]).sum())
            elif dropped.index[i].find(f"$\epsilon$-Greedy") != -1:
                cumsumArrayEpsGreedy.append((self.all_results_part1.loc[self.optimalAlgName] - dropped.iloc[i]).sum())
            elif dropped.index[i].find("TS") != -1:
                cumsumArrayTS.append((self.all_results_part1.loc[self.optimalAlgName] - dropped.iloc[i]).sum())
        
        ax.plot(cumsumArrayNCB, label="ECAD",
            color = colors[0], linestyle="-", linewidth=2)
        ax.text(numFraud, cumsumArrayNCB[-1] - 0.01, "ECAD",
            color=colors[0], fontweight="normal", horizontalalignment="left", verticalalignment="center")
        ax.plot(cumsumArrayUCB, label="UCB",
            color = colors[1], linestyle="-", linewidth=2)
        ax.text(numFraud, cumsumArrayUCB[-1] * 1.01, "UCB",
            color=colors[1], fontweight="normal", horizontalalignment="left", verticalalignment="center")
        ax.plot(cumsumArrayGreedy, label="Greedy",
            color = colors[2], linestyle="-", linewidth=2)
        ax.text(numFraud, cumsumArrayGreedy[-1] * 1.01, "Greedy",
            color=colors[2], fontweight="normal", horizontalalignment="left", verticalalignment="center")
        ax.plot(cumsumArrayEpsGreedy, label="$\epsilon$-Greedy",
            color = colors[3], linestyle="-", linewidth=2)
        ax.text(numFraud, cumsumArrayEpsGreedy[-1] * 1.01, "$\epsilon$-Greedy",
            color=colors[3], fontweight="normal", horizontalalignment="left", verticalalignment="center")
        ax.plot(cumsumArrayTS, label="TS",
            color = colors[4], linestyle="-", linewidth=2)
        ax.text(numFraud, cumsumArrayTS[-1] * 1.01, "TS",
            color=colors[4], fontweight="normal", horizontalalignment="left", verticalalignment="center")
        
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")
        ax.spines["bottom"].set_bounds(0,numFraud)
        ax.set_ylabel(f'Total Regret over {self.num_runs} runs')
        ax.set_xlabel(f'Number of Fraud End Devices')
        fig.tight_layout()
        #create the directory if it does not exist
        directory = "plots/" + self.figSaveDir
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig("plots/" + self.figSaveDir + "/TotalRegret.png", dpi=300)
        
            

    def runExperiments_part2(self, algs, exp_range, max_steps=1000,  alg_name=""):
        self.max_steps = max_steps
        results_list = []
        all_mean_value_per_alg = []
        for alg_impl in algs:
            results_t, total_results = alg_impl.run(max_steps)
            mean_per_alg = results_t.mean(axis=0)
            all_mean_value_per_alg.append(mean_per_alg)
        all_mean_value_per_alg_df = pd.DataFrame(all_mean_value_per_alg)
        all_mean_value_per_alg_df.reset_index(
            inplace=True, drop=True)
        all_mean_value_per_alg_df = all_mean_value_per_alg_df.T
        all_mean_value_per_alg_df.columns = [exp_range]
        return all_mean_value_per_alg_df

    def plot_part2(self, df1, df2, df3, df4):
        plt.plot(list(
            df1.columns.get_level_values(0)), df1.values.flatten(), label="e-greedy")
        plt.plot(list(
            df2.columns.get_level_values(0)), df2.values.flatten(), label="greedy")
        plt.plot(list(
            df3.columns.get_level_values(0)), df3.values.flatten(), label="ucb")
        plt.plot(list(
            df4.columns.get_level_values(0)), df4.values.flatten(), label="ECAD")
        ax = plt.gca()
        ax.set_title(
            'Summary of Results for Multi-Armed Bandit with different algorithms')
        ax.set_ylabel(f'Average reward over {self.max_steps} steps')
        ax.set_xlabel(
            f'Different Hyperparameters - (Eps-greedy : eps, Greedy : Q0, ucb: c) ')
        plt.legend()
        ax.set_xscale('log')
        plt.show()
