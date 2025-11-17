import jax.numpy as jnp
import jax.random as jrng
import jax.nn as jnn
import jax.scipy as jsp
import numpy as np
import scipy as sp

import pytest

from aerieax.nested_sampling import nested_sampler_hmc


@pytest.mark.parametrize("n_dims", [1, 2])
@pytest.mark.parametrize("mu", [0, -0.5, 2.0])
@pytest.mark.parametrize(
    "xmin,xmax",
    [
        (-2, 3),
        (-4, 4),
    ],
)
def test_nested_sampler_hmc_gauss_evidence(n_dims, mu, xmin, xmax):
    def _log_like(x, sigma=1):
        y = _transform(x)
        return -jnp.sum(
            0.5 * (y - mu) ** 2 / sigma**2
            + jnp.log(sigma)
            + 0.5 * jnp.log(2.0 * jnp.pi)
        )

    def _transform(x):
        return (xmax - xmin) * jnn.sigmoid(x) + xmin

    def _inv_transform(y):
        x = (y - xmin) / (xmax - xmin)
        return jsp.special.logit(x)

    def _log_prior(x):
        sx = jnn.log_sigmoid(x)
        return jnp.sum(2 * sx - x)

    def _prior_draw(rng_key):
        y = jrng.uniform(rng_key, shape=(n_dims,), minval=xmin, maxval=xmax)
        return _inv_transform(y)

    true_logZ = n_dims * np.log(
        (sp.stats.norm.cdf(xmax - mu) - sp.stats.norm.cdf(xmin - mu)) / (xmax - xmin)
    )

    rng_key = jrng.PRNGKey(seed=21)
    n_live = 100

    (logZ, delta_logZ, samps, logw, loglike, ns_data) = nested_sampler_hmc(
        rng_key,
        _log_like,
        _log_prior,
        _prior_draw,
        n_dims,
        n_live,
        verbose=False,
    )

    print("logZ|err|true:", logZ, delta_logZ, true_logZ)
    assert np.abs(logZ - true_logZ) < 3.0 * delta_logZ


def test_nested_sampler_hmc_gauss_stats():
    mu = 0.5
    sigma = 0.25
    xmin = -3.5
    xmax = 4.5
    n_dims = 1

    def _log_like(x):
        y = _transform(x)
        return -jnp.sum(
            0.5 * (y - mu) ** 2 / sigma**2
            + jnp.log(sigma)
            + 0.5 * jnp.log(2.0 * jnp.pi)
        )

    def _transform(x):
        return (xmax - xmin) * jnn.sigmoid(x) + xmin

    def _inv_transform(y):
        x = (y - xmin) / (xmax - xmin)
        return jsp.special.logit(x)

    def _log_prior(x):
        sx = jnn.log_sigmoid(x)
        return jnp.sum(2 * sx - x)

    def _prior_draw(rng_key):
        y = jrng.uniform(rng_key, shape=(n_dims,), minval=xmin, maxval=xmax)
        return _inv_transform(y)

    true_logZ = n_dims * np.log(
        (sp.stats.norm.cdf(xmax - mu) - sp.stats.norm.cdf(xmin - mu)) / (xmax - xmin)
    )

    rng_key = jrng.PRNGKey(seed=21)
    n_live = 1000

    (logZ, delta_logZ, samps, logw, loglike, ns_data) = nested_sampler_hmc(
        rng_key,
        _log_like,
        _log_prior,
        _prior_draw,
        n_dims,
        n_live,
        verbose=False,
    )

    print("logZ|err|true:", logZ, delta_logZ, true_logZ)
    assert np.abs(logZ - true_logZ) < 3.0 * delta_logZ

    samps = _transform(samps[:, 0])
    wgts = np.exp(logw)
    wgts /= np.sum(wgts)
    mn = np.sum(samps * wgts)
    sd = np.sqrt(np.sum(wgts * (samps - mu) ** 2))
    print("mn|sd:", mn, sd)

    np.testing.assert_allclose(
        mn,
        mu,
        rtol=0.1,
        atol=0,
    )

    np.testing.assert_allclose(
        sd,
        sigma,
        rtol=0.1,
        atol=0,
    )
