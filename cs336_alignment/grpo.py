import torch
from collections.abc import Callable
from typing import Literal
from transformers import PreTrainedModel, PreTrainedTokenizer
from torch.optim import Optimizer
from cs336_alignment.token_utils import tokenize_prompt_and_output,get_response_log_probs
from cs336_alignment.grpo_utils import compute_policy_gradient_loss, compute_rollout_rewards, compute_group_normalized_rewards,aggregate_loss_across_microbatch



def grpo_train_step(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    optimizer: Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    out = tokenize_prompt_and_output(repeated_prompts,rollout_responses,tokenizer)
    inputs = out["input_ids"]
    labels = out["labels"]
    mask = out["response_mask"]
    loss = torch.zeros(())
    microbatch_size = len(inputs) // gradient_accumulation_steps
    for i in range(0, len(inputs), microbatch_size):
        inputs_microbatch = inputs[i:i+microbatch_size]
        labels_microbatch = labels[i:i+microbatch_size]
        repeated_ground_truths_microbatch = repeated_ground_truths[i:i+microbatch_size]
        rollout_responses_microbatch = rollout_responses[i:i+microbatch_size]
        mask_microbatch = mask[i:i+microbatch_size]
        
        raw_rewards,_ = compute_rollout_rewards(reward_fn,rollout_responses_microbatch,repeated_ground_truths_microbatch)
        advantages,_=compute_group_normalized_rewards(raw_rewards,group_size)
        log_probs=get_response_log_probs(model,inputs_microbatch,labels_microbatch)["log_probs"]

        per_token_policy_gradient_loss,_ = compute_policy_gradient_loss(advantages.unsqueeze(-1), log_probs)
        microbatch_loss = aggregate_loss_across_microbatch(per_token_policy_gradient_loss,mask_microbatch)*len(inputs_microbatch)/len(inputs)
        loss+=microbatch_loss
        microbatch_loss.backward()
    if max_grad_norm is not None:
         torch.nn.utils.clip_grad_norm_(model.parameters(),max_grad_norm)
    optimizer.step()
    optimizer.zero_grad()

    return loss,{}
