# metabayes
## Installation
To install this package, run `pip install metabayes`.

## Example usage
**Note, please see the [demonstration notebook](https://github.com/tbuffington7/bayesian-meta-analysis/blob/main/notebooks/Demonstration.ipynb) for a more detailed example**.

First, import the required modules:

```python
from metabayes import GibbsSampler, Prior
```

Then, you will need to define a `Prior` object to be used in the hierarchical model. Using the default settings:
```python
prior = Prior()
```

After your `Prior` is specified, you can run the model as follows:

```python
gs = GibbsSampler.from_binary_counts(num_trials,
                                     control_successes,
                                     num_trials,
                                     treatment_successes,
                                     prior)
gs.run_model()
```

Note that the model currently only supports binomial metircs (i.e. conversion rate).

## Modeling framework
### Overview
This repo uses a [Bayesian hierarchical model](https://en.wikipedia.org/wiki/Bayesian_hierarchical_modeling) to analyze multiple experiments. The intuition behind this methodology is that the model learns a prior from the experiments included in the dataset. This prior is used to compute a posterior for the true lift of each experiment.

The key results of the model are as follows:
- A posterior distribution for the average lift across all experiments. Intuitively, this quantifies the model's "best guess" for the true lift of a _new_ experiment not in the dataset but belonging to the same category of tests included in the dataset.
- A posterior distribution for the standard deviation of the true lifts across the experiments. Intuitively, this quantifies variation in true effects across experiments. Unlike a purely "fixed effects" approach, **the hierarchical model recognizes that the true effect of each experiment can be different.** If the model concludes that the true effect is likely consistent across all tests, the posterior distribution for the effect standard deviation will be small. If the model concludes that the true effect varies significantly across tests, the posterior distribution for the standard deviation of true effects will be large.
- A posterior distribution for the true effect of each test. This provides a useful estimate for each test's lift with an informed prior. In this sense, the model intelligently uses data from other tests to inform the lift estimate of a given test (via the informed prior), but it also recognizes that each test can have a unique true effect.

### Mathematical details
For a set of $k$ A/B (two-variant) tests that measure lifts to binomial metrics, we have the following quantities:
- The number of users who converted in the control variant of the $i\text{th}$ test, $y_{i, c}$
- The total number of users in the control variant of the $i\text{th}$ test, $n_{i, c}$
- The number of users who converted in the treatment variant of the $i\text{th}$ test, $y_{i, t}$
- The total number of users in the treatment variant of the $i\text{th}$ test, $n_{i, t}$

where $i \in \\{1, ..., k\\}$. Also note that when we say "converted," it can also refer to another binary action, such as retention. The observed conversion rate of each metric is
$$\hat{p}_{i}=\frac{y\_{i,j}}{n\_{i,j}}$$

where $j \in \\{c, t\\}$. The observed _relative_ lift, $\hat{\theta}_{i}$ is as follows: $$\hat{\theta}\_{i}=\frac{\hat{p}\_{i,t}-\hat{p}\_{i,c}}{\hat{p}\_{i,c}}$$





