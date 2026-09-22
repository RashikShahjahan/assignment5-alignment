import torch
from collections.abc import Callable


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
    

