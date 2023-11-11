from typing import Dict

import numpy as np
from numpy import ndarray


class GibbsSampler:
    def __init__(self,
                 control_mean: ndarray,
                 treatment_mean: ndarray,
                 var_control_mean: ndarray,
                 var_treatment_mean: ndarray,
                 alpha: float = 0.1,
                 beta: float = 0.1,
                 epsilon: float = 0.1,
                 num_iterations: int = 1000) -> None:
        """
        Initialize the GibbsSampler with experiment data and parameters.

        Parameters
        ----------
        control_mean : ndarray
            The means of the control group across experiments.
        treatment_mean : ndarray
            The means of the treatment group across experiments.
        var_control_mean : ndarray
            The variances of the control group means.
        var_treatment_mean : ndarray
            The variances of the treatment group means.
        alpha : float, optional
            The alpha parameter for the gamma distribution, default is 0.1.
        beta : float, optional
            The beta parameter for the gamma distribution, default is 0.1.
        epsilon : float, optional
            The epsilon parameter for the normal distribution, default is 0.1.
        num_iterations : int, optional
            The number of iterations for the Gibbs sampling, default is 1000.
        """
        self._validate_inputs(control_mean, treatment_mean, var_control_mean, var_treatment_mean)

        self.parameters = self._build_parameter_dict(control_mean, treatment_mean, var_control_mean,
                                                     var_treatment_mean, alpha, beta, epsilon)

        self.simulation_settings = self._build_simulation_setting_dict(num_iterations)

        self.traces = self._initialize_traces()

    @staticmethod
    def _validate_inputs(control_mean: ndarray, treatment_mean: ndarray,
                         var_control_mean: ndarray, var_treatment_mean: ndarray) -> None:
        num_experiments = len(control_mean)
        for input_array in [treatment_mean, var_control_mean, var_treatment_mean]:
            assert len(input_array) == num_experiments, "The input arrays must all be the same length"

    def _build_parameter_dict(self, control_mean: ndarray, treatment_mean: ndarray,
                              var_control_mean: ndarray, var_treatment_mean: ndarray,
                              alpha: float, beta: float, epsilon: float) -> Dict[str, any]:
        parameters = {
            'control_mean': control_mean,
            'treatment_mean': treatment_mean,
            'var_control_mean': var_control_mean,
            'var_treatment_mean': var_treatment_mean,
            'num_experiments': len(control_mean),
            'alpha': alpha,
            'beta': beta,
            'epsilon': epsilon,
            'percent_deltas': self._calculate_percent_delta(control_mean, treatment_mean),
            'omega': self._calculate_omega(control_mean, treatment_mean, var_control_mean, var_treatment_mean)
        }
        return parameters

    @staticmethod
    def _build_simulation_setting_dict(num_iterations: int) -> Dict[str, int]:
        simulation_settings = {'num_iterations': num_iterations}
        return simulation_settings

    @staticmethod
    def _calculate_percent_delta(control_mean: ndarray, treatment_mean: ndarray) -> ndarray:
        ONE_HUNDRED = 100
        percent_delta = ONE_HUNDRED * (treatment_mean - control_mean) / control_mean
        return percent_delta

    @staticmethod
    def calculate_relative_delta_variance(control_mean: ndarray, treatment_mean: ndarray,
                                          var_control_mean: ndarray, var_treatment_mean: ndarray) -> ndarray:
        var_relative_delta = (1 / control_mean ** 2 * var_treatment_mean +
                              treatment_mean ** 2 / control_mean ** 4 * var_control_mean)
        return var_relative_delta

    def _calculate_omega(self, control_mean: ndarray, treatment_mean: ndarray,
                         var_control_mean: ndarray, var_treatment_mean: ndarray) -> ndarray:
        ONE_HUNDRED = 100

        relative_delta_variance = self.calculate_relative_delta_variance(control_mean,
                                                                         treatment_mean,
                                                                         var_control_mean,
                                                                         var_treatment_mean)

        omega = 1 / (relative_delta_variance * ONE_HUNDRED ** 2)
        return omega

    def _initialize_traces(self) -> Dict[str, ndarray]:
        traces = {
            'tau': np.zeros(self.simulation_settings['num_iterations']),
            'nu': np.zeros(self.simulation_settings['num_iterations']),
            'theta': np.zeros((self.simulation_settings['num_iterations'], self.parameters['num_experiments']))
        }

        traces['tau'][0] = np.random.gamma(self.parameters['alpha'], scale=1 / self.parameters['beta'])
        traces['nu'][0] = np.random.normal(0, np.sqrt(1 / self.parameters['epsilon']))

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
                     + self.parameters['epsilon'])

        mean = (self.parameters['num_experiments'] * self.traces['tau'][iteration_num - 1]
                * np.mean(self.traces['theta'][iteration_num - 1, :])) / precision

        self.traces['nu'][iteration_num] = np.random.normal(mean, np.sqrt(1 / precision))

    def _update_tau(self, iteration_num: int) -> None:
        new_alpha = self.parameters['alpha'] + self.parameters['num_experiments'] / 2

        new_beta = (
                self.parameters['beta'] +
                np.sum(
                    (self.traces['theta'][iteration_num - 1, :] - self.traces['nu'][iteration_num - 1]) ** 2
                ) / 2
        )
        self.traces['tau'][iteration_num] = np.random.gamma(new_alpha, scale=1 / new_beta)

    def run_model(self) -> None:
        for i in range(1, self.simulation_settings['num_iterations']):
            self._update_theta(i)
            self._update_nu(i)
            self._update_tau(i)
