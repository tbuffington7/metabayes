from setuptools import setup, find_packages

VERSION = '0.0.2'
DESCRIPTION = 'A tool for Bayesian meta analysis'
LONG_DESCRIPTION = """
A hierarchical model with random effects to understand aggregated effects of experiments, and to inform priors
"""

# Setting up
setup(
    name="metabayes",
    version=VERSION,
    author="Tyler Buffington",
    author_email="tyler.c.buffington@gmail.com",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    url='https://github.com/tbuffington7/bayesian-meta-analysis',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'tqdm',
        'matplotlib'
    ]
)
