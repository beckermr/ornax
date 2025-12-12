# ornax
[![tests](https://github.com/beckermr/ornax/actions/workflows/tests.yml/badge.svg)](https://github.com/beckermr/ornax/actions/workflows/tests.yml) [![pre-commit.ci status](https://results.pre-commit.ci/badge/github/beckermr/ornax/main.svg)](https://results.pre-commit.ci/latest/github/beckermr/ornax/main)

Affine Invariant HMC and Nested Sampling in JAX

`ornax` is a collection of experimental ensemble Hamiltonian Monte Carlo methods base on the work of Chen ([2025][1]).

## Example

```python
import jax.numpy as jnp
import jax.random as jrng

from ornax.hmc import ensemble_hmc

n_dims = 10
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

print("mean|std|acc:", chain.mean(), chain.std(), acc.mean())
```

## References

- Chen, 2025, [arXiv:2505.02986][1], "New affine invariant ensemble samplers and their dimensional scaling"

[1]: <https://arxiv.org/abs/2505.02987> "https://arxiv.org/abs/2505.02987"
