from cs336_alignment.grpo import grpo_train_step
import wandb
from cs336_alignment.modal_utils import GPU, RUN_TIMEOUT_SECONDS, app, image, wandb_secret
from cs336_alignment.vllm_utils import VLLMServer
from cs336_alignment.checkpoint import get_model_and_tokenizer
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
import xopen
import torch
import json 


n_train_examples = 6400
n_val_examples = 1024
num_rollout_steps = 200
learning_rate = 1e-5
train_batch_size = 256
group_size = 8
gradient_accumulation_steps = 32
max_grad_norm = 1.0

MODEL_ID = "allenai/OLMo-2-0425-1B"
TRAIN_DATASET_PATH = "data/gsm8k/train.jsonl"
TEST_DATASET_PATH = "data/gsm8k/test.jsonl"


def prepare_dataset(dataset_path:str):
    repeated_prompts = []
    repeated_ground_truths = []
    with xopen(dataset_path) as f:
            for line in f:
                content = json.loads(line)
                question = content["question"]
                prompt = f"A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {question} Assistant: <think>"
                repeated_prompts.append(prompt)
                ground_truth = content["answer"].split("####")[-1].strip()
                repeated_ground_truths.append(ground_truth)

    return repeated_prompts, repeated_ground_truths



@app.function(image=image, gpu=GPU, timeout=RUN_TIMEOUT_SECONDS, secrets=[wandb_secret])
def train_gsm8k() -> None:
    with wandb.init(
        project="cs336-a5-rlvr",
        job_type="train",
        config={
            "model_id": MODEL_ID,
            "n_train_examples": n_train_examples,
            "n_val_examples": n_val_examples,
            "num_rollout_steps": num_rollout_steps,
            "learning_rate": learning_rate,
            "train_batch_size": train_batch_size,
            "group_size": group_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "max_grad_norm": max_grad_norm,
        },
    ) as run:
            model,tokenizer = get_model_and_tokenizer(MODEL_ID, 'cuda:1')
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.0)
            prompts, ground_truths = prepare_dataset(TRAIN_DATASET_PATH)
            sampling_params = {"temperature": 1.0, "max_tokens": 512, "n":8, "seed":0}
            sampling_params["stop"] = ["</answer>"]
            sampling_params["include_stop_str_in_output"] = True
            vllm_server = VLLMServer(MODEL_ID, gpu=0)
            vllm_server.start()
            
            try:
                vllm_server.init_weight_sync('cuda:1')
                for i in range(0,len(prompts), train_batch_size//group_size):
                    prompts_batch = prompts[i:i+train_batch_size//group_size]
                    ground_truths_batch = ground_truths[i:i+train_batch_size//group_size]
                    vllm_server.sync_policy_weights(model)
                    rollout_responses = vllm_server.generate_completions(prompts_batch,sampling_params)
                    rollout_responses_text = []


                    for response in rollout_responses:
                        rollout_responses_text.append(response.text)

                    repeated_prompts = []
                    repeated_ground_truths = []
                    
                    for prompt, ground_truth in zip(prompts_batch,ground_truths_batch):
                        repeated_prompts.extend([prompt]*group_size)
                        repeated_ground_truths.extend([ground_truth]*group_size)
                        
                    
                    loss,_= grpo_train_step(model,tokenizer,optimizer,gradient_accumulation_steps,max_grad_norm,r1_zero_reward_fn,repeated_prompts,rollout_responses_text,repeated_ground_truths,group_size)
                    run.log({
                        "train/loss": loss.item(),
                        "train/rollout_count": len(rollout_responses_text),
                    }, step=i // (train_batch_size // group_size) + 1)
            finally:
                vllm_server.stop()
     
          

@app.local_entrypoint()
def modal_main() -> None:
    train_gsm8k.remote()
