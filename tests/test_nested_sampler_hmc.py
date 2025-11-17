import math

import jax
import jax.numpy as jnp
import jax.random as jrng
import jax.nn as jnn
import jax.scipy as jsp
import numpy as np
import scipy as sp

import pytest

from ornax.nested_sampling import nested_sampler_hmc
from ornax.nested_sampling.nested_sampler_hmc import _2d_array_to_nested_tuples


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

    print("\n", end="", flush=True)
    (logZ, delta_logZ, samps, logw, loglike, ns_data) = nested_sampler_hmc(
        rng_key,
        _log_like,
        _log_prior,
        _prior_draw,
        n_dims,
        n_live,
        verbose=True,
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

    print("\n", end="", flush=True)
    (logZ, delta_logZ, samps, logw, loglike, ns_data) = nested_sampler_hmc(
        rng_key,
        _log_like,
        _log_prior,
        _prior_draw,
        n_dims,
        n_live,
        verbose=True,
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


@pytest.mark.parametrize("n_dims", [1, 2])
@pytest.mark.parametrize("mu", [0, -0.5, 2.0])
@pytest.mark.parametrize(
    "xmin,xmax",
    [
        (-2, 3),
        (-4, 4),
    ],
)
def test_nested_sampler_hmc_gauss_evidence_transform(n_dims, mu, xmin, xmax):
    def _log_like(x, sigma=1):
        return -jnp.sum(
            0.5 * (x - mu) ** 2 / sigma**2
            + jnp.log(sigma)
            + 0.5 * jnp.log(2.0 * jnp.pi)
        )

    def _log_prior(x):
        return -n_dims * jnp.log(xmax - xmin)

    def _prior_draw(rng_key):
        return jrng.uniform(rng_key, shape=(n_dims,), minval=xmin, maxval=xmax)

    true_logZ = n_dims * np.log(
        (sp.stats.norm.cdf(xmax - mu) - sp.stats.norm.cdf(xmin - mu)) / (xmax - xmin)
    )

    rng_key = jrng.PRNGKey(seed=21)
    n_live = 100

    print("\n", end="", flush=True)
    (logZ, delta_logZ, samps, logw, loglike, ns_data) = nested_sampler_hmc(
        rng_key,
        _log_like,
        _log_prior,
        _prior_draw,
        n_dims,
        n_live,
        prior_domain=jnp.array([[xmin, xmax]] * n_dims),
        verbose=True,
    )

    print("logZ|err|true:", logZ, delta_logZ, true_logZ)
    assert np.abs(logZ - true_logZ) < 3.0 * delta_logZ


def test_nested_sampler_hmc_gauss_evidence_transform_mixed():
    n_dims = 2
    mu = 0.5
    xmin = -2
    xmax = 3
    xmin_auto = -3
    xmax_auto = 2

    def _transform(x):
        return jnp.array(
            [
                (xmax - xmin) * jnn.sigmoid(x[0]) + xmin,
                x[1],
            ]
        )

    def _inv_transform(y):
        x = (y[0] - xmin) / (xmax - xmin)
        return jnp.array(
            [
                jsp.special.logit(x),
                y[1],
            ]
        )

    def _log_like(x, sigma=1):
        y = _transform(x)
        return -jnp.sum(
            0.5 * (y - mu) ** 2 / sigma**2
            + jnp.log(sigma)
            + 0.5 * jnp.log(2.0 * jnp.pi)
        )

    def _log_prior(x):
        sx = jnn.log_sigmoid(x[0])
        return jnp.sum(2 * sx - x[0]) - jnp.log(xmax - xmin)

    @jax.jit
    def _prior_draw(rng_key):
        y = jrng.uniform(rng_key, shape=(n_dims,), minval=0, maxval=1)
        y = jnp.array(
            [
                (xmax - xmin) * y[0] + xmin,
                (xmax_auto - xmin_auto) * y[1] + xmin_auto,
            ]
        )
        return _inv_transform(y)

    true_logZ = np.log(
        (sp.stats.norm.cdf(xmax - mu) - sp.stats.norm.cdf(xmin - mu)) / (xmax - xmin)
    ) + np.log(
        (sp.stats.norm.cdf(xmax_auto - mu) - sp.stats.norm.cdf(xmin_auto - mu))
        / (xmax_auto - xmin_auto)
    )

    rng_key = jrng.PRNGKey(seed=21)
    n_live = 100

    print("\n", end="", flush=True)
    (logZ, delta_logZ, samps, logw, loglike, ns_data) = nested_sampler_hmc(
        rng_key,
        _log_like,
        _log_prior,
        _prior_draw,
        n_dims,
        n_live,
        prior_domain=jnp.array([[-jnp.inf, jnp.inf], [xmin_auto, xmax_auto]]),
        verbose=True,
    )

    print("logZ|err|true:", logZ, delta_logZ, true_logZ)
    assert np.abs(logZ - true_logZ) < 3.0 * delta_logZ


@pytest.mark.parametrize(
    "vals,tvals",
    [
        (((np.inf,),), ((math.inf,),)),
        (((jnp.inf,),), ((math.inf,),)),
        (((-np.inf, -jnp.inf),), ((-math.inf, -math.inf),)),
        (
            (
                (-2, 5),
                (-np.inf, -jnp.inf),
            ),
            (
                (-2, 5),
                (-math.inf, -math.inf),
            ),
        ),
    ],
)
@pytest.mark.parametrize("as_array", [True, False])
def test_nested_sampler_hmc_2d_array_to_nested_tuples(vals, tvals, as_array):
    if as_array:
        vals = jnp.array(vals)
    assert _2d_array_to_nested_tuples(vals) == tvals
