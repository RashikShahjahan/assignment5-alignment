import torch
from collections.abc import Callable
from typing import Literal


def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    rewards = []
    total_reward = 0
    total_format_reward = 0
    for rollout, ground_truth in zip(rollout_responses, repeated_ground_truths):
        reward_dict = reward_fn(rollout, ground_truth)
        rewards.append(reward_dict["reward"])
        total_reward+=reward_dict["reward"]
        total_format_reward+=reward_dict["format_reward"]

    raw_rewards = torch.tensor(rewards)
    mean_reward = total_reward/len(rewards)
    mean_format_reward = total_format_reward/len(rewards)

    return raw_rewards, {"mean_reward":mean_reward,"mean_format_reward":mean_format_reward }

def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
):
    norm_rewards = torch.zeros_like(raw_rewards)
    for i in range(raw_rewards.numel()//group_size):
        group_rewards = raw_rewards[i*group_size:(i+1)*group_size]
        mean_reward = torch.mean(group_rewards)
        std_reward = torch.std(group_rewards)
        norm_rewards[i*group_size:(i+1)*group_size] = (group_rewards-mean_reward)/(std_reward+advantage_eps)

    return norm_rewards, {"mean_reward":mean_reward}


