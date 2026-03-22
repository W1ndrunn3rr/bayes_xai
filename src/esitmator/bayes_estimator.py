import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
from sklearn.base import BaseEstimator, ClassifierMixin
from numpyro.infer import MCMC, NUTS
from ..models.pyro_model import BayesianModel


class BayessianClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self, n_samples: int = 2000, num_warmup: int = 1000, num_chains: int = 1
    ) -> None:
        self._model = BayesianModel
        self.rng_key = random.PRNGKey(0)
        self.rng_key, _ = random.split(self.rng_key)
        self.kernel = NUTS(self._model)
        self.n_samples_ = n_samples
        self.num_warmup_ = num_warmup
        self.num_chains_ = num_chains
        self.samples_ = None

    def _run_kernel(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> None:
        mcmc = MCMC(
            self.kernel,
            num_warmup=self.num_warmup_,
            num_samples=self.n_samples_,
            num_chains=self.num_chains_,
        )
        mcmc.run(self.rng_key, X=X, y=y)
        if verbose:
            mcmc.print_summary()
        self.samples_ = mcmc.get_samples()

    def fit(
        self, X: np.ndarray, y: np.ndarray, verbose: bool = False
    ) -> "BayessianClassifier":
        X = jnp.array(X, dtype=jnp.float32)
        y = jnp.array(y, dtype=jnp.float32)

        self._run_kernel(X, y, verbose=verbose)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = jnp.array(X, dtype=jnp.float32)

        logits = X @ self.samples_["weights"].T + self.samples_["intercept"]
        probs = jax.nn.sigmoid(logits).mean(axis=1)

        return np.array(np.stack([1 - probs, probs], axis=1))

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = jnp.array(X, dtype=jnp.float32)
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    def predict_uncertainty(self, X: np.ndarray) -> dict:
        X = jnp.array(X, dtype=jnp.float32)

        logits = X @ self.samples_["weights"].T + self.samples_["intercept"]
        probs = jax.nn.sigmoid(logits)

        return {
            "mean": np.array(probs.mean(axis=1)),
            "std": np.array(probs.std(axis=1)),
            "lower": np.array(jnp.percentile(probs, 2.5, axis=1)),
            "upper": np.array(jnp.percentile(probs, 97.5, axis=1)),
        }
