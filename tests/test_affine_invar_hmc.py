import jax.numpy as jnp
import jax.random as jrng
import numpy as np

from aerieax.hmc import ensemble_hmc


def test_ensemble_hmc_smoke():
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
        n_dims=2,
        n_samples=10000,
        verbose=False,
    )

    assert chain.shape == (10000, 10, 2)
    assert acc.shape == (10000, 10)
    assert loglike.shape == (10000, 10)

    print("mean|std|acc:", chain.mean(), chain.std(), acc.mean())

    assert acc.mean() > 0.9
    assert np.allclose(
        chain.mean(),
        2.0,
        rtol=0.01,
        atol=0,
    )
    assert np.allclose(
        chain.std(),
        1.25,
        rtol=0.01,
        atol=0,
    )
