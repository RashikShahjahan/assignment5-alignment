import torch
from transformers import PreTrainedTokenizer, PreTrainedModel

def tokenize_prompt_and_output(
    prompt_strs: list[str], 
    output_strs: list[str], 
    tokenizer: PreTrainedTokenizer,
) -> dict[str, torch.Tensor]:
    prompt_ids = tokenizer(prompt_strs)["input_ids"]
    output_ids = tokenizer(output_strs)["input_ids"]

    combined_ids = []
    roles = []
    for prompt, output in zip(prompt_ids, output_ids):
        combined = torch.tensor(prompt+output,dtype=torch.long)
        role = torch.cat([torch.zeros(len(prompt), dtype=torch.bool),torch.ones(len(prompt), dtype=torch.bool)])

        roles.append(role)
        combined_ids.append(combined)

    combined_ids = torch.nn.utils.rnn.pad_sequence(combined_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    roles=torch.nn.utils.rnn.pad_sequence(roles, batch_first=True, padding_value=False)


    input_ids = combined_ids[:,:-1]
    labels = combined_ids[:,1:]
    response_mask = roles[:,1:]

    return {"input_ids":input_ids,"labels":labels, "response_mask":response_mask }


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    logits = model(input_ids).logits
    log_probs_full = torch.log_softmax(logits, dim=-1)
    
    if return_token_entropy:
        probs = torch.exp(log_probs_full)
        token_entropy = torch.sum(probs*log_probs_full, dim=-1)*-1

    else:
        token_entropy = None
        
    log_probs = torch.gather(log_probs_full, dim=-1, index= labels.unsqueeze(-1)).squeeze(-1)


    return {"log_probs":log_probs,"token_entropy":token_entropy}
