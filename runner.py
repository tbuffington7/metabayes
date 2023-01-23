import numpy as np

from src.gibbs_sampler import GibbsSampler

np.random.seed(777)
num_experiments = 100
true_control_means = np.random.beta(100, 150, size=num_experiments)
true_effects = np.random.normal(.05, 0.03, size=num_experiments)
true_treatment_means = true_control_means * (1 + true_effects)

num_trials = np.random.randint(1000, 5000)
control_successes = np.random.binomial(num_trials, true_control_means)
treatment_successes = np.random.binomial(num_trials, true_treatment_means)

control_means = control_successes / num_trials
treatment_means = treatment_successes / num_trials

var_control_means = (control_means * (1 - control_means)) / num_trials
var_treatment_means = (treatment_means * (1 - treatment_means)) / num_trials

gs = GibbsSampler(control_means, treatment_means, var_control_means, var_treatment_means)
gs.run_model()

x = 3
