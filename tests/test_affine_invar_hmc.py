import jax
import jax.numpy as jnp
import jax.random as jrng
import numpy as np

import pytest

from aerieax.hmc import ensemble_hmc


@pytest.mark.parametrize("n_dims", [1, 2, 10])
def test_ensemble_hmc_smoke(n_dims):
    rng_key = jrng.key(10)

    def _log_like(x, sigma=1.25, mu=2):
        return -jnp.sum(
            0.5 * (x - mu) ** 2 / sigma**2
            + jnp.log(sigma)
            + 0.5 * jnp.log(2.0 * jnp.pi)
        )

    chain, acc, loglike = ensemble_hmc(
        rng_key,
        _log_like,
        n_dims=n_dims,
        n_samples=10000,
        verbose=False,
    )

    n_walkers = max(2 * n_dims, 10)

    assert chain.shape == (10000, n_walkers, n_dims)
    assert acc.shape == (10000, n_walkers)
    assert loglike.shape == (10000, n_walkers)

    print("mean|std|acc:", chain.mean(), chain.std(), acc.mean())

    assert acc.mean() > 0.9
    np.testing.assert_allclose(
        chain.mean(),
        2.0,
        rtol=0.01,
        atol=0,
    )
    np.testing.assert_allclose(
        chain.std(),
        1.25,
        rtol=0.01,
        atol=0,
    )

    np.testing.assert_allclose(
        loglike,
        jax.vmap(jax.vmap(_log_like))(chain),
        rtol=1e-6,
        atol=0,
    )
