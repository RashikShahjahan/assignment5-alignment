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
        if baseline == "mean":
            baseline_reward = torch.mean(group_rewards)
        else:
            baseline_reward = 0

        if advantage_normalizer == "std":
            advnorm_reward = torch.std(group_rewards)
        elif advantage_normalizer == "mean":
            advnorm_reward = torch.mean(group_rewards)
        else:
            advnorm_reward = 0
            advantage_eps = 1


        norm_rewards[i*group_size:(i+1)*group_size] = (group_rewards-baseline_reward)/(advnorm_reward+advantage_eps)
            
   

    return norm_rewards, {}

        

    
def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    if importance_reweighting_method == "none":
        return -1 * raw_rewards_or_advantages*policy_log_probs, {}
    if importance_reweighting_method == "noclip":
        return -1 * raw_rewards_or_advantages*torch.exp(policy_log_probs-old_log_probs), {}
    if importance_reweighting_method == "grpo":
        return -1 *torch.minimum( raw_rewards_or_advantages*torch.exp(policy_log_probs-old_log_probs), raw_rewards_or_advantages*torch.clamp(torch.exp(policy_log_probs-old_log_probs),max=1+cliprange,min=1-cliprange)) , {}
    if importance_reweighting_method == "gspo":
        token_log_ratio = policy_log_probs - old_log_probs.detach()
        response_lengths = response_mask.sum(dim=1, keepdim=True)

        sequence_log_ratio = (
            token_log_ratio * response_mask
        ).sum(dim=1, keepdim=True) / response_lengths

        sequence_ratio = torch.exp(sequence_log_ratio)
        clipped_ratio = torch.clamp(
            sequence_ratio,
            1.0 - cliprange,
            1.0 + cliprange,
        )

        advantages = raw_rewards_or_advantages.reshape(-1, 1)
        sequence_loss = -torch.minimum(
            advantages * sequence_ratio,
            advantages * clipped_ratio,
        )

        per_token_loss = sequence_loss.expand_as(policy_log_probs)

        return per_token_loss, {}
def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    masked_loss = per_token_policy_gradient_loss*mask

    if loss_normalization == "sequence":
        return torch.mean(torch.sum(masked_loss,dim=1)/torch.sum(mask,dim=1), dim=0)
    if loss_normalization ==  "constant":
        return torch.sum(torch.sum(masked_loss,dim=1), dim=0)/normalization_constant


