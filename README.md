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
results = gs.get_posterior_samples()
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
#### The likelihood
For a set of $k$ A/B (two-variant) tests that measure lifts to binomial metrics, we have the following quantities:
- The number of users who converted in the control variant of the $i\text{th}$ test, $y_{i, c}$
- The total number of users in the control variant of the $i\text{th}$ test, $n_{i, c}$
- The number of users who converted in the treatment variant of the $i\text{th}$ test, $y_{i, t}$
- The total number of users in the treatment variant of the $i\text{th}$ test, $n_{i, t}$

where $i \in \\{1, ..., k\\}$. Also note that when we say "converted," it can also refer to another binary action, such as retention. The observed conversion rate of each metric is
$$\hat{p}_{i}=\frac{y\_{i,j}}{n\_{i,j}}$$

where $j \in \\{c, t\\}$. The observed _relative_ lift, $\hat{\theta}\_{i}$ is as follows: $$\hat{\theta}\_{i}=\frac{\hat{p}\_{i,t}-\hat{p}\_{i,c}}{\hat{p}\_{i,c}}$$ 
When the sample sizes are sufficiently large, $\hat{\theta}\_{i}$ will be approximately normally distributed because of the central limit theorem CLT. This is fairly easy to recognize for $\hat{p}\_{i,j}$ as it is a sample mean. However, it may not be as obvious how CLT helps with the relative lift, $\hat{\theta}\_{i}$. To justify its asymptotic normality, we use the delta method, which is discussed in more detail in [Deng et al. (2013)](https://alexdeng.github.io/public/files/kdd2018-dm.pdf). With this assumption, we can state:
$$\hat{\theta}\_{i} \sim N(\theta_{i}, \sigma^2\_{i})$$

where $\theta_{i}$ is the _true_ relative lift and $\sigma^2\_{i}$ is the variance of the estimated lift for the $i\text{th}$ experiment, respectively. Using the delta method, we can compute $\sigma^2\_{i}$ as follows:

$$\sigma^2\_{i} = \frac{1}{\omega_{i}} = \frac{1}{\hat{p}\_{i,c}^2}var(\hat{p}\_{i, t}) + \frac{\hat{p}\_{i,t}^2}{\hat{p}\_{i,c}^4}var(\hat{p}\_{i, c})$$

For reasons that will become clearer later, we have also defined the _precision_, $\omega_{i}$, which is simply the recriprocal of the variance. Note that $var(\hat{p}\_{i, c})$ and $var(\hat{p}\_{i, t})$ can be estimated using the following equation:
$$var(\hat{p}\_{i, j}) = \frac{\hat{p}\_{i,j}(1 - \hat{p}\_{i, j})}{n}$$

These equations give us well-defined statistical descriptions of how the _observed_ lifts are distributed given the underlying _true_ lifts. In other words, the _likelihood_ of our Bayesian model is fully defined:

$$\hat{\theta}\_{i} | \theta_{i} \sim N(\theta_{i}, \frac{1}{\omega_{i}})$$


#### The priors
Now that we have fully defined the likelihood, let's consider the priors. Given that our likelihood ($\hat{\theta}\_{i}$) is a normal distribution, it is convenient to use a normal prior to take advantage of the normal-normal [_conjugacy_](https://en.wikipedia.org/wiki/Conjugate_prior). The resulting prior is as follows:
$$\theta\_{i} \sim N(\mu, \frac{1}{\tau})$$

where $\mu$ is the average lift across all experiments and $\tau$ is the precision of lifts across all experiments. Note that the _standard deviation_ of lifts across all experiments is simply $\frac{1}{\sqrt{\tau}}$. There is an assumption here that the distribution of true lifts is normally distributed. There is no CLT justification for this assumption, but it is a common assumption in meta-analyses. I believe there is some evidence that the actual distribution of true lifts is leptokurtic (has heavier tails than a normal distribution). In practice, this means that more ideas have near-zero lifts, but some ideas have extremely large lifts relative to what a normal distribution would expect. One _could_ use a prior with heavier tails, such as a t-distribution, but the resulting math would be more complicated and would likely require a probabalistic programming software such as PyMC or Stan.

We could stop here if we wanted as we now have a fully specified Bayesian model with a prior and a likelihood and leave it to users to choose sensible values for $\mu$ and $\tau$. However, given that this a meta analysis, we can instead construct a model that learns (in a Bayesian sense) $\mu$ and $\tau$ from the data. This is what makes the model _hierarchical_-- the fact that we define priors on parameters that are used in priors. In other words, we have a hierarchy of priors. Now let's define those "top-level" priors. Again, it is convenient to use a normal prior for $\mu$:

$$\mu \sim N(0, \frac{1}{\epsilon})$$

We use a mean of 0 to reflect agnosticism about the direction of the meta-lift. The precision of this prior, $\epsilon$ is a user-specified parameter, but by default, it is set to 1600. This corresponds to a standard deviation of 2.5%, which means that we are 95% confident a-priori that the true meta effect is between -5% and 5% (+/- two standard deviations). This loosely reflects the range of true lifts observed in typical A/B tests. Users are encouraged to try different values of $\epsilon$, but I expect that the results will not be very sensitive to this parameter assuming that a sufficient number of tests are included in the analysis. At some point, I hope to do a more thorough sensitivity analysis.

We also need to define a prior for $\tau$. Again, it is convenient to take advantage of conjugacy, which suggests we use a gamma prior:

$$\tau \sim Gamma(\alpha, \beta)$$

We use default values of $\alpha$=0.1 and $\beta$=0.00001 as these numbers produce a distribution with a range that is reasonable for A/B tests. When $\tau$ is translated to a standard deviation, most of the probability mass is between 0% and 10%. Again, users are encouraged to try different values here, but the results will likely not be sensitive to these parameters as long as extreme values are not used (don't say that there's a ~100% chance that the true effect standard deviation is above 50%; we know that's not true), and enough tests are included in the meta-analysis.

![image](https://github.com/tbuffington7/bayesian-meta-analysis/assets/24952168/f09f5f7d-893e-4bd5-b204-d6195c36fcfa)

#### Gibbs Sampling
Now that we have fully defined our priors and our likelihood, we now discuss how the posterior distributions are sampled using a [Gibbs sampler](https://en.wikipedia.org/wiki/Gibbs_sampling). The end result is a _trace_ for each learned parameter, which approximates samples from the relevant posterior distribution. For example words, if you plot a histogram of the trace for $\mu$, that essentially shows the posterior distribution for $\mu$.

The Gibbs sampler draws from a computed _conditional_ posterior for each learned parameter. In the posterior calculations, we condition on the most recently drawn values of parameters that aren't relevant to the current calculation.

To compute the next value in the trace for $\theta_{i}$: we use the result from the normal-normal conjuacy, which states that the posterior mean is a precision-weighted average of the prior mean and the likelihood mean, and that the posterior precision is the sum of the prior and likelihood precisions.

$$
\theta_{i} | \mu, \hat{\theta}\_{i}, \tau \sim N(\frac{\omega\_{i}\hat{\theta}\_{i} + \tau\mu}{\omega\_{i} + \tau}, \omega\_{i} + \tau)
$$

We again use the normal-normal conjugate pair to update $\mu$:

$$
\mu | \theta\_{i}, \tau \sim N(\frac{\tau\sum\theta\_{i}}{k\tau + \epsilon}, k\tau + \epsilon)
$$

Finally, we use the normal-gamma conjugate pair to update $\tau$:

$$
\tau | \mu, \theta\_{i} \sim Gamma(\alpha+\frac{k}{2}, \beta + \frac{\sum(\theta\_{i} - \mu)^2}{2})
$$
