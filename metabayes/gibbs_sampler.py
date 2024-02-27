from typing import Dict

import numpy as np
from numpy import ndarray
from tqdm import tqdm

from metabayes.priors import Prior


class GibbsSampler:
    def __init__(self,
                 control_means: ndarray,
                 treatment_means: ndarray,
                 control_mean_variances: ndarray,
                 treatment_mean_variances: ndarray,
                 prior,
                 num_iterations: int = 10000) -> None:
        """
        Initialize the GibbsSampler with experiment data and parameters.

        Parameters
        ----------
        control_means : ndarray
            The means of the control group across experiments.
        treatment_means : ndarray
            The means of the treatment group across experiments.
        control_mean_variances : ndarray
            The variances of the control group means.
        treatment_mean_variances : ndarray
            The variances of the treatment group means.
        prior: Prior
            The prior object
        num_iterations : int, optional
            The number of iterations for the Gibbs sampling, default is 1000.
        """
        self._validate_inputs(control_means, treatment_means, control_mean_variances, treatment_mean_variances)
        self.prior = prior
        self.parameters = self._build_data_dict(control_means, treatment_means, control_mean_variances,
                                                treatment_mean_variances)
        self.simulation_settings = self._build_simulation_setting_dict(num_iterations)
        self.traces = self._initialize_traces()

    @classmethod
    def from_binary_counts(cls, control_sample_size: int, control_successes: int,
                           treatment_sample_size: int, treatment_successes: int,
                           prior: float, num_iterations: int = 10000) -> 'GibbsSampler':
        """
        Class method to create a GibbsSampler instance from binary count data.

        Parameters
        ----------
        control_sample_size : int
            The size of the control sample.
        control_successes : int
            The number of successes in the control sample.
        treatment_sample_size : int
            The size of the treatment sample.
        treatment_successes : int
            The number of successes in the treatment sample.
        prior : float
            The prior value for the GibbsSampler.
        num_iterations : int, optional
            The number of iterations for the Gibbs sampler, default is 10000.

        Returns
        -------
        GibbsSampler
            An instance of the GibbsSampler class initialized with the calculated means and variances
            from the control and treatment samples, along with the provided prior and number of iterations.
        """

        control_means = control_successes / control_sample_size
        treatment_means = treatment_successes / treatment_sample_size

        control_mean_variances = (control_means * (1 - control_means)) / control_sample_size
        treatment_mean_variances = (treatment_means * (1 - treatment_means)) / treatment_sample_size

        return cls(control_means, treatment_means, control_mean_variances, treatment_mean_variances, prior,
                   num_iterations)

    @staticmethod
    def _validate_inputs(control_mean: ndarray, treatment_mean: ndarray,
                         var_control_mean: ndarray, var_treatment_mean: ndarray) -> None:
        num_experiments = len(control_mean)
        for input_array in [treatment_mean, var_control_mean, var_treatment_mean]:
            assert len(input_array) == num_experiments, "The input arrays must all be the same length"

    def _build_data_dict(self, control_mean: ndarray, treatment_mean: ndarray,
                         var_control_mean: ndarray, var_treatment_mean: ndarray) -> Dict[str, any]:
        data = {
            'num_experiments': len(control_mean),
            'control_mean': control_mean,
            'treatment_mean': treatment_mean,
            'var_control_mean': var_control_mean,
            'var_treatment_mean': var_treatment_mean,
            'percent_deltas': self._calculate_relative_lift(control_mean, treatment_mean),
            'omega': self._calculate_omega(control_mean, treatment_mean, var_control_mean, var_treatment_mean)
        }
        return data

    @staticmethod
    def _build_simulation_setting_dict(num_iterations: int) -> Dict[str, int]:
        simulation_settings = {'num_iterations': num_iterations}
        return simulation_settings

    @staticmethod
    def _calculate_relative_lift(control_mean: ndarray, treatment_mean: ndarray) -> ndarray:
        percent_delta = (treatment_mean - control_mean) / control_mean
        return percent_delta

    @staticmethod
    def calculate_relative_delta_variance(control_mean: ndarray, treatment_mean: ndarray,
                                          var_control_mean: ndarray, var_treatment_mean: ndarray) -> ndarray:
        var_relative_delta = (1 / control_mean ** 2 * var_treatment_mean +
                              treatment_mean ** 2 / control_mean ** 4 * var_control_mean)
        return var_relative_delta

    def _calculate_omega(self, control_mean: ndarray, treatment_mean: ndarray,
                         var_control_mean: ndarray, var_treatment_mean: ndarray) -> ndarray:
        relative_delta_variance = self.calculate_relative_delta_variance(control_mean,
                                                                         treatment_mean,
                                                                         var_control_mean,
                                                                         var_treatment_mean)

        omega = 1 / relative_delta_variance
        return omega

    def _initialize_traces(self) -> Dict[str, ndarray]:
        traces = {
            'tau': np.zeros(self.simulation_settings['num_iterations']),
            'nu': np.zeros(self.simulation_settings['num_iterations']),
            'theta': np.zeros((self.simulation_settings['num_iterations'], self.parameters['num_experiments']))
        }

        traces['tau'][0] = np.random.gamma(self.prior.alpha, scale=1 / self.prior.beta)
        traces['nu'][0] = np.random.normal(0, np.sqrt(1 / self.prior.epsilon))

        traces['theta'][0, :] = np.random.normal(traces['nu'][0],
                                                 np.sqrt(1 / traces['tau'][0]), size=self.parameters['num_experiments'])
        return traces

    def _update_theta(self, iteration_num: int) -> None:
        precision = self.parameters['omega'] + self.traces['tau'][iteration_num - 1]
        mean = (self.parameters['omega'] * self.parameters['percent_deltas'] +
                self.traces['tau'][iteration_num - 1] * self.traces['nu'][iteration_num - 1]) / precision
        self.traces['theta'][iteration_num] = np.random.normal(mean, np.sqrt(1 / precision))

    def _update_nu(self, iteration_num: int) -> None:
        precision = (self.parameters['num_experiments'] * self.traces['tau'][iteration_num - 1]
                     + self.prior.epsilon)

        mean = (self.parameters['num_experiments'] * self.traces['tau'][iteration_num - 1]
                * np.mean(self.traces['theta'][iteration_num - 1, :])) / precision

        self.traces['nu'][iteration_num] = np.random.normal(mean, np.sqrt(1 / precision))

    def _update_tau(self, iteration_num: int) -> None:
        new_alpha = self.prior.alpha + self.parameters['num_experiments'] / 2

        new_beta = (
                self.prior.beta +
                np.sum(
                    (self.traces['theta'][iteration_num - 1, :] - self.traces['nu'][iteration_num - 1]) ** 2
                ) / 2
        )
        self.traces['tau'][iteration_num] = np.random.gamma(new_alpha, scale=1 / new_beta)

    def run_model(self) -> None:
        for i in tqdm(range(1, self.simulation_settings['num_iterations'])):
            self._update_theta(i)
            self._update_nu(i)
            self._update_tau(i)
