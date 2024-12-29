"""
Optimized Multi-Armed Bandit Framework
Enhancements include modularity, performance optimization, and additional features.
"""

import numpy as np
from abc import ABC, abstractmethod
from numpy_ml.utils.testing import random_one_hot_matrix, is_number


class Bandit(ABC):
    """
    Base class for different types of Bandit environments.
    """

    def __init__(self, rewards, reward_probs, context=None):
        assert len(rewards) == len(reward_probs), "Rewards and probabilities must have the same length."
        self.step = 0
        self.n_arms = len(rewards)
        self.rewards = rewards
        self.reward_probs = reward_probs
        super().__init__()

    def __repr__(self):
        """String representation of the bandit."""
        params = ", ".join(f"{k}={v}" for k, v in self.hyperparameters.items() if k != "id")
        return f"{self.hyperparameters['id']}({params})"

    @property
    def hyperparameters(self):
        """A dictionary of bandit hyperparameters."""
        return {}

    @abstractmethod
    def oracle_payoff(self, context=None):
        """Expected reward for an optimal agent."""
        pass

    def pull(self, arm_id, context=None):
        """
        Simulates pulling a given arm and returns the reward.

        Parameters
        ----------
        arm_id : int
            Index of the arm to pull.
        context : ndarray, optional
            Context matrix for the current timestep, if applicable.
        """
        assert 0 <= arm_id < self.n_arms, "Invalid arm index."
        self.step += 1
        return self._pull(arm_id, context)

    def reset(self):
        """Reset bandit state."""
        self.step = 0

    @abstractmethod
    def _pull(self, arm_id, context=None):
        """Defines how rewards are sampled from the arm."""
        pass


class MultinomialBandit(Bandit):
    """
    Multi-armed bandit with multinomial payoff distributions.
    """

    def __init__(self, payoffs, payoff_probs):
        super().__init__(payoffs, payoff_probs)
        for r, rp in zip(payoffs, payoff_probs):
            assert len(r) == len(rp), "Payoffs and probabilities must align."
            np.testing.assert_almost_equal(sum(rp), 1.0, err_msg="Probabilities must sum to 1.")

        self.payoffs = np.array([np.array(x) for x in payoffs])
        self.payoff_probs = np.array([np.array(x) for x in payoff_probs])
        self.arm_evs = np.array([sum(p * v) for p, v in zip(payoff_probs, payoffs)])
        self.best_ev = np.max(self.arm_evs)
        self.best_arm = np.argmax(self.arm_evs)

    @property
    def hyperparameters(self):
        return {"id": "MultinomialBandit", "payoffs": self.payoffs, "payoff_probs": self.payoff_probs}

    def oracle_payoff(self, context=None):
        """Returns the expected reward for the best arm."""
        return self.best_ev, self.best_arm

    def _pull(self, arm_id, context=None):
        """Samples the reward from the arm's multinomial distribution."""
        payoffs = self.payoffs[arm_id]
        probs = self.payoff_probs[arm_id]
        return np.random.choice(payoffs, p=probs)


class BernoulliBandit(Bandit):
    """
    Multi-armed bandit with Bernoulli payoff distributions.
    """

    def __init__(self, payoff_probs):
        payoffs = [1] * len(payoff_probs)
        super().__init__(payoffs, payoff_probs)

        self.payoffs = np.array(payoffs)
        self.payoff_probs = np.array(payoff_probs)
        self.arm_evs = self.payoff_probs
        self.best_ev = np.max(self.arm_evs)
        self.best_arm = np.argmax(self.arm_evs)

    @property
    def hyperparameters(self):
        return {"id": "BernoulliBandit", "payoff_probs": self.payoff_probs}

    def oracle_payoff(self, context=None):
        return self.best_ev, self.best_arm

    def _pull(self, arm_id, context=None):
        """Samples reward as a Bernoulli random variable."""
        return int(np.random.rand() <= self.payoff_probs[arm_id])


class GaussianBandit(Bandit):
    """
    Multi-armed bandit with Gaussian payoff distributions.
    """

    def __init__(self, payoff_dists, payoff_probs):
        super().__init__(payoff_dists, payoff_probs)

        self.payoff_dists = payoff_dists
        self.payoff_probs = payoff_probs
        self.arm_evs = np.array([mu for (mu, var) in payoff_dists])
        self.best_ev = np.max(self.arm_evs)
        self.best_arm = np.argmax(self.arm_evs)

    @property
    def hyperparameters(self):
        return {"id": "GaussianBandit", "payoff_dists": self.payoff_dists, "payoff_probs": self.payoff_probs}

    def oracle_payoff(self, context=None):
        return self.best_ev, self.best_arm

    def _pull(self, arm_id, context=None):
        """Samples reward as a Gaussian random variable."""
        mean, var = self.payoff_dists[arm_id]
        return np.random.normal(mean, var)


class ContextualBernoulliBandit(Bandit):
    """
    Contextual Bernoulli bandit where each context feature is associated with
    an independent Bernoulli payoff distribution.
    """

    def __init__(self, context_probs):
        D, K = context_probs.shape
        placeholder = [None] * K
        super().__init__(placeholder, placeholder)

        self.context_probs = context_probs
        self.arm_evs = self.context_probs
        self.best_evs = self.arm_evs.max(axis=1)
        self.best_arms = self.arm_evs.argmax(axis=1)

    @property
    def hyperparameters(self):
        return {"id": "ContextualBernoulliBandit", "context_probs": self.context_probs}

    def get_context(self):
        D, K = self.context_probs.shape
        context = np.zeros((D, K))
        context[np.random.choice(D), :] = 1
        return random_one_hot_matrix(1, D).ravel()

    def oracle_payoff(self, context):
        context_id = context[:, 0].argmax()
        return self.best_evs[context_id], self.best_arms[context_id]

    def _pull(self, arm_id, context):
        D, K = self.context_probs.shape
        arm_probs = context[:, arm_id] @ self.context_probs
        arm_rwds = (np.random.rand(K) <= arm_probs).astype(int)
        return arm_rwds[arm_id]


class ContextualLinearBandit(Bandit):
    """
    Contextual linear bandit where payoffs are determined by a linear combination
    of context vectors and arm-specific coefficient vectors.
    """

    def __init__(self, K, D, payoff_variance=1):
        if is_number(payoff_variance):
            payoff_variance = [payoff_variance] * K

        self.K = K
        self.D = D
        self.payoff_variance = payoff_variance
        placeholder = [None] * K
        super().__init__(placeholder, placeholder)

        self.thetas = np.random.uniform(-1, 1, size=(D, K))
        self.thetas /= np.linalg.norm(self.thetas, axis=0)

    @property
    def hyperparameters(self):
        return {"id": "ContextualLinearBandit", "K": self.K, "D": self.D, "payoff_variance": self.payoff_variance}

    def get_context(self):
        return np.random.normal(size=(self.D, self.K))

    def oracle_payoff(self, context):
        best_arm = np.argmax(self.arm_evs)
        return self.arm_evs[best_arm], best_arm

    def _pull(self, arm_id, context):
        self.arm_evs = np.dot(context.T, self.thetas).diagonal()
        noise = np.random.normal(scale=self.payoff_variance, size=self.K)
        return (self.arm_evs + noise)[arm_id]
