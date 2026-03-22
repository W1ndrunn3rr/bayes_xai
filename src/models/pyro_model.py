import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def BayesianModel(X: np.ndarray, y=None):
    weights = numpyro.sample(
        "weights", dist.Normal(jnp.zeros(X.shape[1]), jnp.ones(X.shape[1]))
    )
    intercept = numpyro.sample("intercept", dist.Normal(0.0, 1.0))

    logits = X @ weights + intercept
    numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=y)
