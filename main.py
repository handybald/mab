
from algorithms.Egreedy import Egreedy
from algorithms.Greedy import Greedy
from algorithms.UpperConfidenceBound import UpperConfidenceBound
from algorithms.NewConfidenceBound import NewConfidenceBound
from algorithms.Optimal import OptimalAlgorithm
from algorithms.ThompsonSampling import ThompsonSampling
from classes.MultiArmBandit import MultiArmBandit
from helpers.ExperimentRunner import ExperimentRunner
import numpy as np



def main():
    np.random.seed(42)
    numOfArms = 30
    numOfRuns = 100
    maxSteps = 101
    optimalArm = 15
    period = 30
    experiment_runner = ExperimentRunner(num_runs=numOfRuns, max_steps=maxSteps, optimalAlgName='Optimal', figSaveDir="30Arm_15thOptArm_Period30_Steps101_Run100_01ScaleNoise_mod2Fraud_01AmpNoiseReward")
    mab = MultiArmBandit(numOfArms, period=period, optimalArm=optimalArm, numOfFrauds=0)
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

    # you already opened space in mab for fraud and zero rewards, use it next
    # then plot for different noise and amplitude?? levels (maybe not needed)

    # Part 2 - Performance of the algorithms on increasing fraud rewards

    numOfArms = 30
    numOfRuns = 100
    maxSteps = 101
    optimalArm = 15
    period = 30
    experiment_runner = ExperimentRunner(num_runs=numOfRuns, max_steps=maxSteps, optimalAlgName='Optimal', figSaveDir="Fraud")
    mab = MultiArmBandit(numOfArms, period=period, optimalArm=optimalArm, numOfFrauds=0)

    optimalArg = OptimalAlgorithm(optimalArm, maxSteps, mab=mab)
    experiment_runner.runExperiments_part1(alg=optimalArg, alg_name=f'Optimal')
    fraudNum = np.arange(1, 31, 1)
    for fraud in fraudNum: 
        mab = MultiArmBandit(numOfArms, period=period, optimalArm=optimalArm, numOfFrauds=fraud)
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


    # Part 3 - Performance of the algorithms on increasing 0 rewards




if __name__ == '__main__':
    main()


# def main():
#     np.random.seed(42)

#     parser = argparse.ArgumentParser(description='Multi-Armed Bandit')
#     parser.add_argument('--num-arms', type=int, default=10, metavar='N',
#                         help='Number of arms of MAB.')
#     parser.add_argument('--num-runs', type=int, default=10, metavar='N',
#                         help='Number of runs to repeat and average for each algorithm.')
#     args = parser.parse_args()
#     period = 5
#     mab = MultiArmBandit(args.num_arms, period=period)
#     experiment_runner = ExperimentRunner()

#     # Part 1 - A plot of reward over time (averaged over 100 runs each) on the
#     # same axes, for 𝜖-greedy with 𝜖 = 0.1, greedy with 𝑄1 = 5, and
#     # UCB with 𝑐 = 2
#     eps = 0.1
#     eps_greedy = Egreedy(eps=eps, mab=mab)
#     experiment_runner.runExperiments_part1(alg=eps_greedy,
#                                            max_steps=1000, num_runs=300, alg_name=f'e-greedy with e={eps}')

#     q1 = 5
#     greedy_optimistic_ini = Greedy(Q1=q1, mab=mab)
#     experiment_runner.runExperiments_part1(alg=greedy_optimistic_ini,
#                                            max_steps=1000, num_runs=300, alg_name=f'Greedy with Q1={q1}')

#     c = 2
#     upper_bound_conf = UpperConfidenceBound(c=c, mab=mab)
#     experiment_runner.runExperiments_part1(alg=upper_bound_conf,
#                                            max_steps=1000, num_runs=300, alg_name=f'UCB with c={c}')
    
#     new_bound_conf = NewConfidenceBound(mab=mab, period=5, max_steps=1000)
#     experiment_runner.runExperiments_part1(alg=new_bound_conf,
#                                            max_steps=1000, num_runs=300, alg_name=f'NCB')

#     experiment_runner.plot_part1()
#     # Part 2 - A summary comparison plot of rewards over first 1000 steps for
#     # the three algorithms with different values of the hyperparameters

#     # E - greedy
#     eps_greedy_range = [1/128, 1/64, 1/32, 1/16, 1/4, 1/2]
#     eps_greedy_algs = []
#     for eps in eps_greedy_range:
#         eps_greedy = Egreedy(eps=eps, mab=mab)
#         eps_greedy_algs.append(eps_greedy)
#     eps_greedy_results = experiment_runner.runExperiments_part2(algs=eps_greedy_algs, exp_range=eps_greedy_range,
#                                                                 max_steps=1000,  alg_name='e-greedy')

#     # Greedy with 𝑄1
#     q1_range = [1/32, 1/16, 1/8, 1/4, 1/2, 1, 2]
#     greedy_algs = []
#     for q1 in q1_range:
#         greedy_optimistic_ini = Greedy(Q1=q1, mab=mab)
#         greedy_algs.append(greedy_optimistic_ini)
#     greedy_results = experiment_runner.runExperiments_part2(algs=greedy_algs, exp_range=q1_range,
#                                                             max_steps=1000,  alg_name='greedy')
#     # UCB
#     c_range = [1/16, 1/8, 1/4, 1/2, 1, 2, 4]
#     upper_bound_conf_algs = []
#     for c in c_range:
#         upper_bound_conf = UpperConfidenceBound(c=c, mab=mab)
#         upper_bound_conf_algs.append(upper_bound_conf)
#     ucb_results = experiment_runner.runExperiments_part2(algs=upper_bound_conf_algs, exp_range=c_range,
#                                                          max_steps=1000, alg_name='ucb')
#     #NCB
#     p_range = [5, 10, 20, 50, 100]
#     new_bound_conf_algs = []
#     for p in p_range:
#         new_bound_conf = NewConfidenceBound(mab=mab, period=p, max_steps=1000)
#         new_bound_conf_algs.append(new_bound_conf)
#     ncb_results = experiment_runner.runExperiments_part2(algs=new_bound_conf_algs, exp_range=p_range, max_steps=1000, alg_name='ncb')
#     # plot
#     experiment_runner.plot_part2(
#         eps_greedy_results, greedy_results, ucb_results, ncb_results)
#     plt.show()


# if __name__ == '__main__':
#     main()
