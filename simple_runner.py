from __future__ import division

import bilby
import numpy as np
import shutil
import os
import corner

class SimpleGaussianLikelihood(bilby.Likelihood):
    def __init__(self, data):
        super().__init__(parameters={'mu': None, 'sigma': None})
        self.data = data
        self.N = len(data)

    def log_likelihood(self):
        mu = self.parameters['mu']
        sigma = self.parameters['sigma']
        res = self.data - mu
        return -0.5 * (np.sum((res / sigma) ** 2) + self.N * np.log(
            2 * np.pi * sigma ** 2))


def main():
    # A few simple setup steps
    label = 'gaussian_example'
    outdir = 'outdir'

    # Making simulated data: in this case, we consider just a Gaussian
    data = np.random.normal(3, 4, 100)

    likelihood = SimpleGaussianLikelihood(data)
    priors = dict(mu=bilby.core.prior.Uniform(0, 5, 'mu'),
                  sigma=bilby.core.prior.Uniform(0, 10, 'sigma'))

    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    # And run sampler
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler='dynesty',
        npoints=1000,
        walks=10,
        outdir=outdir,
        label=label,
        sample='mrwalk',
        check_point_delta_t=100000,
        queue_size=10
    )
    corner.corner(result.posterior)


if __name__ == '__main__':
    main() 