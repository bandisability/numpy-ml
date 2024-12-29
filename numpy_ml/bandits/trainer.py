import warnings
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from numpy_ml.utils.testing import DependencyWarning

warnings.filterwarnings("ignore", category=DependencyWarning)

class BanditTrainer:
    """
    A modular trainer for multi-armed bandit (MAB) policies, allowing easy comparison,
    evaluation, and visualization of results.
    """

    def __init__(self, smoothing_weight=0.999, seed=None):
        """
        Initialize the trainer with configurable parameters.

        Parameters:
            smoothing_weight (float): Weight for smoothing results.
            seed (int): Random seed for reproducibility.
        """
        self.logs = {}
        self.smoothing_weight = smoothing_weight
        if seed is not None:
            np.random.seed(seed)

    def compare(self, policies, bandit, n_trials, n_duplicates, save_dir=None, verbose=True):
        """
        Compare multiple bandit policies on a given bandit environment.

        Parameters:
            policies (list): List of bandit policies.
            bandit (object): Bandit environment.
            n_trials (int): Number of trials per policy.
            n_duplicates (int): Number of duplicate runs per policy.
            save_dir (str): Directory to save plots.
            verbose (bool): Whether to print progress information.
        """
        self._initialize_logs(policies)
        results = {}
        for policy in policies:
            if verbose:
                print(f"Training policy: {policy}")
            results[policy] = self.train(policy, bandit, n_trials, n_duplicates, verbose)

        if save_dir:
            self._plot_comparisons(policies, save_dir)
        return results

    def train(self, policy, bandit, n_trials, n_duplicates, verbose=True):
        """
        Train a single policy on the bandit environment.

        Parameters:
            policy (object): Bandit policy to train.
            bandit (object): Bandit environment.
            n_trials (int): Number of trials to run.
            n_duplicates (int): Number of duplicate runs.
            verbose (bool): Whether to display progress.
        """
        policy_name = str(policy)
        self._initialize_logs(policy)

        for run in range(n_duplicates):
            if verbose:
                print(f"Run {run + 1}/{n_duplicates} for {policy_name}")
            bandit.reset()
            policy.reset()

            for trial in range(n_trials):
                reward, arm, optimal_reward, optimal_arm = self._execute_trial(policy, bandit)
                self._log_results(policy_name, trial, reward, optimal_reward, arm, optimal_arm)

        return self.logs[policy_name]

    def _execute_trial(self, policy, bandit):
        """
        Execute a single trial for the given policy on the bandit environment.

        Returns:
            tuple: (reward, chosen_arm, optimal_reward, optimal_arm)
        """
        context = bandit.get_context() if hasattr(bandit, "get_context") else None
        reward, arm = policy.act(bandit, context)
        optimal_reward, optimal_arm = bandit.oracle_payoff(context)
        return reward, arm, optimal_reward, optimal_arm

    def _initialize_logs(self, policies):
        """
        Initialize logs for each policy.

        Parameters:
            policies (list or object): List of policies or a single policy.
        """
        if not isinstance(policies, list):
            policies = [policies]

        for policy in policies:
            self.logs[str(policy)] = {
                "reward": defaultdict(list),
                "optimal_reward": defaultdict(list),
                "selected_arm": defaultdict(list),
                "optimal_arm": defaultdict(list),
                "regret": defaultdict(list),
                "cumulative_regret": defaultdict(list),
            }

    def _log_results(self, policy_name, trial, reward, optimal_reward, arm, optimal_arm):
        """
        Log trial results for a specific policy.

        Parameters:
            policy_name (str): Name of the policy.
            trial (int): Current trial number.
            reward (float): Reward obtained in the trial.
            optimal_reward (float): Optimal reward for the trial.
            arm (int): Chosen arm.
            optimal_arm (int): Optimal arm.
        """
        log = self.logs[policy_name]
        regret = optimal_reward - reward
        log["reward"][trial].append(reward)
        log["optimal_reward"][trial].append(optimal_reward)
        log["selected_arm"][trial].append(arm)
        log["optimal_arm"][trial].append(optimal_arm)
        log["regret"][trial].append(regret)
        log["cumulative_regret"][trial].append(
            regret + (log["cumulative_regret"][trial - 1][-1] if trial > 0 else 0)
        )

    def _plot_comparisons(self, policies, save_dir):
        """
        Generate and save comparison plots for policies.

        Parameters:
            policies (list): List of policies to compare.
            save_dir (str): Directory to save the plots.
        """
        for policy in policies:
            log = self.logs[str(policy)]
            plt.figure(figsize=(10, 6))

            # Plot cumulative regret
            trials = range(len(log["cumulative_regret"]))
            cumulative_regret = [np.mean(log["cumulative_regret"][t]) for t in trials]
            plt.plot(trials, cumulative_regret, label=f"{policy} - Cumulative Regret")

            # Plot smoothed rewards
            smoothed_rewards = [
                self._smooth(log["reward"][t]) for t in trials
            ]
            plt.plot(trials, smoothed_rewards, label=f"{policy} - Smoothed Rewards")

            plt.title(f"Performance of {policy}")
            plt.xlabel("Trials")
            plt.ylabel("Metrics")
            plt.legend()
            plt.savefig(os.path.join(save_dir, f"{policy}.png"))
            plt.close()

    def _smooth(self, values):
        """
        Apply exponential smoothing to a list of values.

        Parameters:
            values (list): List of numerical values.

        Returns:
            float: Smoothed value.
        """
        if not values:
            return 0
        smoothed = values[0]
        for value in values[1:]:
            smoothed = self.smoothing_weight * smoothed + (1 - self.smoothing_weight) * value
        return smoothed
