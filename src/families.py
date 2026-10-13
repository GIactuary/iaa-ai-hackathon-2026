"""Target families shared by the GLM, the trees, validation and evidence.

Convention used everywhere in src/: predictions are stored on the link scale (eta). The family converts eta to the
response scale and scores eta against the actual y. An offset (for example log exposure) is on the link scale too.

| Family        | Link on the response   | GLM fit                         | LightGBM objective | Score (lower is better)       |
|---------------|------------------------|---------------------------------|--------------------|-------------------------------|
| log_gaussian  | log(shift + y)         | least squares on log(shift + y) | regression         | RMSE on the log scale (RMSLE) |
| poisson       | log                    | IRLS                            | poisson            | mean Poisson deviance         |
| gamma         | log                    | IRLS                            | gamma              | mean gamma deviance           |
| tweedie       | log                    | IRLS, variance mu^p             | tweedie            | mean Tweedie deviance         |
| binomial      | logit                  | IRLS                            | binary             | log loss                      |
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit, xlogy

ETA_CLIP = 30.0


def _w(weight, n):
    return np.ones(n) if weight is None else np.asarray(weight, dtype=float)


class Family:
    name = ""
    link_name = ""
    score_name = ""
    lgbm_objective = ""
    closed_form = False          # True when the GLM is a single weighted least-squares fit on link(y)

    # --- link -------------------------------------------------------------------------------------------
    def link(self, mu):
        raise NotImplementedError

    def inverse(self, eta):
        raise NotImplementedError

    def dmu_deta(self, eta):
        raise NotImplementedError

    def variance(self, mu):
        raise NotImplementedError

    # --- scoring ----------------------------------------------------------------------------------------
    def row_loss(self, y, eta):
        """Per-record loss whose weighted mean, after `finalize`, is the score."""
        raise NotImplementedError

    def finalize(self, mean_loss):
        return mean_loss

    def score(self, y, eta, weight=None):
        y, eta = np.asarray(y, dtype=float), np.asarray(eta, dtype=float)
        w = _w(weight, len(y))
        return float(self.finalize(np.sum(w * self.row_loss(y, eta)) / np.sum(w)))

    # --- helpers ----------------------------------------------------------------------------------------
    def check_target(self, y):
        y = np.asarray(y, dtype=float)
        if not np.all(np.isfinite(y)):
            raise ValueError(f"{self.name}: target has missing or infinite values")
        return y

    def start_mu(self, y, weight):
        """Starting mean for IRLS: halfway between each record and the overall mean, away from the boundary."""
        w = _w(weight, len(y))
        m = np.sum(w * y) / np.sum(w)
        return (y + m) / 2

    def lgbm_label(self, y):
        return np.asarray(y, dtype=float)

    def lgbm_params(self) -> dict:
        return {"objective": self.lgbm_objective}

    def is_log_link(self) -> bool:
        return self.link_name in ("log", "log1p")

    def __repr__(self):
        return f"{type(self).__name__}()"


class LogGaussian(Family):
    """Gaussian errors on log(shift + y). shift=1 gives log1p, the scale of RMSLE."""
    name, score_name, lgbm_objective, closed_form = "log_gaussian", "RMSE on log scale", "regression", True

    def __init__(self, shift: float = 1.0):
        self.shift = float(shift)
        self.link_name = "log1p" if self.shift == 1.0 else "log"

    def link(self, mu):
        return np.log(self.shift + np.asarray(mu, dtype=float))

    def inverse(self, eta):
        return np.exp(np.asarray(eta, dtype=float)) - self.shift

    def dmu_deta(self, eta):
        return np.exp(eta)

    def variance(self, mu):
        return np.ones_like(mu)

    def row_loss(self, y, eta):
        return (self.link(y) - eta) ** 2

    def finalize(self, mean_loss):
        return np.sqrt(mean_loss)

    def check_target(self, y):
        y = super().check_target(y)
        if np.any(y <= -self.shift):
            raise ValueError(f"log_gaussian: every y must exceed {-self.shift}")
        return y

    def lgbm_label(self, y):
        return self.link(y)

    def __repr__(self):
        return f"LogGaussian(shift={self.shift})"


class _LogLink(Family):
    link_name = "log"

    def link(self, mu):
        return np.log(np.asarray(mu, dtype=float))

    def inverse(self, eta):
        return np.exp(np.clip(eta, -ETA_CLIP, ETA_CLIP))

    def dmu_deta(self, eta):
        return self.inverse(eta)


class Poisson(_LogLink):
    name, score_name, lgbm_objective = "poisson", "mean Poisson deviance", "poisson"

    def variance(self, mu):
        return mu

    def row_loss(self, y, eta):
        mu = self.inverse(eta)
        return 2 * (xlogy(y, y) - y * np.clip(eta, -ETA_CLIP, ETA_CLIP) - y + mu)

    def check_target(self, y):
        y = super().check_target(y)
        if np.any(y < 0):
            raise ValueError("poisson: counts must be non-negative")
        return y


class Gamma(_LogLink):
    name, score_name, lgbm_objective = "gamma", "mean gamma deviance", "gamma"

    def variance(self, mu):
        return mu ** 2

    def row_loss(self, y, eta):
        eta = np.clip(eta, -ETA_CLIP, ETA_CLIP)
        return 2 * (eta - np.log(y) + y * np.exp(-eta) - 1)

    def check_target(self, y):
        y = super().check_target(y)
        if np.any(y <= 0):
            raise ValueError("gamma: every amount must be positive")
        return y


class Tweedie(_LogLink):
    name, lgbm_objective = "tweedie", "tweedie"

    def __init__(self, power: float = 1.5):
        if not 1 < power < 2:
            raise ValueError("tweedie power must be strictly between 1 and 2 (compound Poisson-gamma)")
        self.power = float(power)
        self.score_name = f"mean Tweedie deviance (p={self.power})"

    def variance(self, mu):
        return mu ** self.power

    def row_loss(self, y, eta):
        p = self.power
        mu = self.inverse(eta)
        return 2 * (np.power(y, 2 - p) / ((1 - p) * (2 - p)) - y * np.power(mu, 1 - p) / (1 - p) + np.power(mu, 2 - p) / (2 - p))

    def check_target(self, y):
        y = super().check_target(y)
        if np.any(y < 0):
            raise ValueError("tweedie: losses must be non-negative")
        return y

    def lgbm_params(self):
        return {"objective": "tweedie", "tweedie_variance_power": self.power}

    def __repr__(self):
        return f"Tweedie(power={self.power})"


class Binomial(Family):
    name, link_name, score_name, lgbm_objective = "binomial", "logit", "log loss", "binary"

    def link(self, mu):
        mu = np.clip(np.asarray(mu, dtype=float), 1e-15, 1 - 1e-15)
        return np.log(mu / (1 - mu))

    def inverse(self, eta):
        return expit(np.clip(eta, -ETA_CLIP, ETA_CLIP))

    def dmu_deta(self, eta):
        m = self.inverse(eta)
        return m * (1 - m)

    def variance(self, mu):
        return mu * (1 - mu)

    def row_loss(self, y, eta):
        eta = np.clip(eta, -ETA_CLIP, ETA_CLIP)
        return np.logaddexp(0, eta) - y * eta

    def start_mu(self, y, weight):
        return (y + 0.5) / 2

    def check_target(self, y):
        y = super().check_target(y)
        if np.any((y < 0) | (y > 1)):
            raise ValueError("binomial: target must be 0/1 (or a proportion)")
        return y


def get_family(name: str, **kw) -> Family:
    table = {"log_gaussian": LogGaussian, "poisson": Poisson, "gamma": Gamma, "tweedie": Tweedie, "binomial": Binomial}
    if name not in table:
        raise ValueError(f"unknown family {name!r}; choose from {sorted(table)}")
    return table[name](**kw)
