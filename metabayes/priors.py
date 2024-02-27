import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


def _plot_percentage(plotting_method):
    def format_x_axis(self, **plot_params):
        plotting_method(self, **plot_params)
        ax = plt.gca()
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0%}'))

    return format_x_axis


class Prior:
    def __init__(self, alpha: float = .1, beta: float = 0.00001, epsilon: float = 1600):
        """
        Initializes the Prior class

        Parameters
        ----------
        alpha: float
            The alpha parameter in the inverse gamma prior over the effect variance
        beta: float
            The alpha parameter in the inverse gamma prior over the effect variance
        epsilon: float
            The prior precision for the effect
        """
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

    @_plot_percentage
    def plot_meta_effect_prior(self, num_draws=10 ** 5, lower=-.1, upper=0.1, num_bins=100, **plot_params):
        draws = np.random.normal(0, self.epsilon ** -.5, size=num_draws)
        plt.hist(draws, bins=np.linspace(lower, upper, num_bins), **plot_params)
        plt.xlabel("Meta Relative Lift %")
        plt.ylabel("Probability Density")

    @_plot_percentage
    def plot_effect_sd_prior(self, num_draws=10 ** 5, lower=0, upper=0.1, num_bins=100, **plot_params):
        draws = np.random.gamma(self.alpha, 1 / self.beta, size=num_draws)
        # We drew precisions, so we need to convert them to standard deviations
        draws = draws ** -.5
        plt.hist(draws, bins=np.linspace(lower, upper, num_bins), **plot_params)
        plt.xlim(lower, upper)
        plt.xlabel("Effect standard deviation")
        plt.ylabel("Probability Density")
