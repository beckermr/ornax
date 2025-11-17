import functools

import jax
import jax.numpy as jnp
import jax.random as jrng
import numpy as np

from jax_tqdm import scan_tqdm


def _leapfrog_base(log_like, x, p, B, n, h, scalar):
    # we use the usual way the KDK loops
    # are unrolled
    # for four iterations we have
    #
    #   K2 D K2, K2 D K2, K2 D K2, K2 D K2
    #   K2 D K      D K      D K      D K2
    #
    # so we get a single half kick, 4-1 = 3 full
    # drifts+kicks, then a full drift + halkf kick

    gfunc = jax.grad(log_like)
    vgfunc = jax.value_and_grad(log_like)

    # first half kick
    vi, g = vgfunc(x)
    p = p - h / 2.0 * jnp.dot(B.T, -g)

    # n - 1 full drft + kick
    for _ in range(n - 1):
        if scalar:
            x = x + h * B * p
        else:
            x = x + h * jnp.dot(B, p)
        g = gfunc(x)
        p = p - h * jnp.dot(B.T, -g)

    # full drift, half kick
    if scalar:
        x = x + h * B * p
    else:
        x = x + h * jnp.dot(B, p)
    vf, g = vgfunc(x)
    p = p - h / 2 * jnp.dot(B.T, -g)

    # reverse p
    p = -p

    has_nans = (
        jnp.any(jnp.isnan(p))
        | jnp.any(jnp.isnan(p))
        | jnp.any(jnp.isnan(vi))
        | jnp.any(jnp.isnan(vf))
    )
    vf = jax.lax.cond(
        has_nans,
        lambda _x: jnp.inf,
        lambda _x: _x,
        vf,
    )
    p = jax.lax.cond(
        has_nans,
        lambda _x: jnp.zeros_like(_x),
        lambda _x: _x,
        p,
    )

    return x, p, -vi, -vf


@functools.partial(
    jax.jit,
    static_argnums=(0, 1, 2, 3, 4),
)
def _walk_step(
    log_like,
    n_steps,
    h,
    n_walkers_2,
    n_dims,
    carry,
    x_not_used,
):
    (
        params,
        rng_key,
    ) = carry

    _leapfrog = jax.jit(
        jax.vmap(
            _leapfrog_base,
            in_axes=(None, 0, 0, None, None, None, None),
            out_axes=0,
        ),
        static_argnums=(0, 4, 6),
    )

    bfac = 1.0 / np.sqrt(n_walkers_2)

    x_new = []
    acc_new = []
    nll_new = []

    for s in range(2):
        # draw p
        rng_key, nrm_key = jrng.split(rng_key)
        p = jrng.normal(nrm_key, shape=(n_walkers_2, n_walkers_2))

        if s == 0:
            x = params[:n_walkers_2, :]
            xB = params[n_walkers_2:, :]
        else:
            x = params[n_walkers_2:, :]
            # use chain from previous s loop
            xB = x_new[0]

        # make B
        mn = jnp.mean(xB, axis=0, keepdims=True)
        B = (xB - mn) * bfac
        B = B.T

        # do _leapfrog
        x_pr, p_pr, v, v_pr = _leapfrog(log_like, x, p, B, n_steps, h, False)

        # measure q = exp(-V(xn) - 0.5 * pn * pn + V(x) + 0.5 * p *p)
        logq = (
            v + 0.5 * jnp.sum(p * p, axis=1) - v_pr - 0.5 * jnp.sum(p_pr * p_pr, axis=1)
        )
        q = jnp.exp(logq)
        q = jnp.clip(q, min=0, max=1)

        # draw r from u[0,1]
        rng_key, unf_key = jrng.split(rng_key)
        r = jrng.uniform(unf_key, shape=(n_walkers_2,))

        # accept xn if r <= q else accept x
        acc_val = r <= q
        x_new.append(jnp.where(acc_val.reshape(n_walkers_2, 1), x_pr, x))
        acc_new.append(q)
        nll_new.append(jnp.where(acc_val, v_pr, v))

    x_new = jnp.concatenate(x_new)
    acc_new = jnp.concatenate(acc_new)
    nll_new = jnp.concatenate(nll_new)

    return (
        x_new,
        rng_key,
    ), (x_new, acc_new, nll_new)


def ensemble_hmc(
    rng_key,
    log_likelihood,
    n_dims,
    n_samples,
    params_init=None,
    n_walkers=None,
    leapfrog_step_size=None,
    n_leapfrog_steps=None,
    verbose=True,
):
    """Run ensemble HMC via the "walk step" algorithm from Chen
    (2025, arXiv:2505.02987).

    Parameters
    ----------
    rng_key : PRNG key
        The RNG key to use.
    log_likelihood :  callable
        A callable with signature `log_likelihood(x)`
        that returns the log-likelihood for a single sample point.
    n_dims : int
        The number of dimensions.
    n_samples : int
        The number of sampling steps to take.
    params_init : jax.numpy.ndarray, optional
        The initial starting points for the walkers w/ shape (n_walkers, n_dims).
        If `None`, then a small unit normal ball about zero is made.
    n_walkers : int, optional
        The number of walkers to use. Must be at least `2 * n_dims` and generally
        a minimum of 10 is good. The default of `None` will use
        `max(2 * n_dims, 10)` walkers.
    leapfrog_step_size : float, optional
        The step size for the leapfrog integration. The default of `None` will
        use `0.1 * (n_dims)**(-0.25)`.
    n_leapfrog_steps : int, optional
        The number of leapfrog steps to take during the inegration. If `None`, a
        default of `int(1/leapfrog_step_size)` will be used.
    verbose : bool, optional
        If True, print the progress of the chain and the acceptance rate. The default
        is True.

    Returns
    -------
    chain : jax.numpy.ndarray
        The MCMC chain w/ shape (n_samples, n_walkers, n_dims).
    acc : jax.numpy.ndarray
        The acceptance probabilities w/ shape (n_samples, n_walkers).
    loglike : jax.numpy.ndarray
        The log-likelihood at each step 2/ shape (n_samples, n_walkers).
    """
    if n_walkers is None:
        n_walkers = max(2 * n_dims, 10)

    if leapfrog_step_size is None:
        leapfrog_step_size = 0.1 * np.power(n_dims, -0.25)

    if n_leapfrog_steps is None:
        n_leapfrog_steps = int(1 / leapfrog_step_size)

    if params_init is None:
        rng_key, init_key = jrng.split(rng_key)
        params_init = jrng.normal(init_key, shape=(n_walkers, n_dims))
    else:
        assert n_dims == params_init.shape[1], (
            "The number of dimensions given by `n_dims` must match the number of "
            "dimensions implied by `params_init`. You passed `n_dims`={n_dims} and "
            f"`params_init.shape[1]={params_init.shape[1]}."
        )
        assert n_walkers == params_init.shape[0], (
            "The number of walkers given by `n_walkers` must match the number of "
            "walkers implied by `params_init`. We have `n_walkers`={n_walkers} and "
            f"`params_init.shape[0]={params_init.shape[0]}."
        )

    assert n_walkers % 2 == 0, (
        f"The number of walkers must be even! You passed n_walkers={n_walkers}."
    )
    assert n_walkers >= 2 * n_dims, (
        f"You must use at least `2 * n_dims` walkers. You passed n_walkers={n_walkers} "
        f"and `2 * n_dims`={2 * n_dims}."
    )
    n_walkers_2 = n_walkers // 2

    _local_walk_step = functools.partial(
        _walk_step,
        log_likelihood,
        n_leapfrog_steps,
        leapfrog_step_size,
        n_walkers_2,
        n_dims,
    )

    if verbose:
        _local_walk_step = scan_tqdm(
            n_samples, tqdm_type="std", ncols=80, desc="sampling"
        )(_local_walk_step)

    _, (chain, acc, nloglike) = jax.lax.scan(
        _local_walk_step, (params_init, rng_key), xs=jnp.arange(n_samples)
    )
    if verbose:
        print("acceptance rate: %0.2f%%" % (100.0 * acc.mean()))
    return chain, acc, -nloglike
