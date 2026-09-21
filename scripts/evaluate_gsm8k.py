import json

from xopen import xopen

from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from cs336_alignment.modal_utils import RUN_TIMEOUT_SECONDS, app, image
from cs336_alignment.vllm_utils import VLLMServer


MODEL_ID = "allenai/OLMo-2-0425-1B"
DATASET_PATH = "data/gsm8k/test.jsonl"


@app.function(image=image, gpu="L4", timeout=RUN_TIMEOUT_SECONDS)
def evaluate_gsm8k() -> None:
    vllm_server = VLLMServer(MODEL_ID, gpu=0)
    vllm_server.start()

    try:
        with xopen(DATASET_PATH) as f:
            for line in f:
                content = json.loads(line)
                question = content["question"]
                answer = content["answer"].split("####")[-1].strip()

            

                question_only_prompt = (
                    f"{question} Please put your final answer within \\boxed{{}}."
                )

                zero_shot_prompt = f"A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {question} Assistant: <think>"

                few_shot_prompt = f"A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? Assistant: <think> There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. So the answer is 6. </think> <answer> 6 </answer> User: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? Assistant: <think> There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. So the answer is 5. </think> <answer> 5 </answer> User: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? Assistant: <think> Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. So the answer is 39. </think> <answer> 39 </answer> User: {question} Assistant: <think>"

                sampling_params = {"temperature": 1.0, "max_tokens": 512, "n":1, "seed":0}
                sampling_params["stop"] = ["</answer>"]
                sampling_params["include_stop_str_in_output"] = True

                question_only_response = vllm_server.generate_completions(
                    [question_only_prompt], sampling_params
                )[0].text
                question_only_reward = question_only_reward_fn(
                    question_only_response, answer
                )

                zero_shot_response = vllm_server.generate_completions(
                    [zero_shot_prompt], sampling_params
                )[0].text
                zero_shot_reward = r1_zero_reward_fn(zero_shot_response, answer)

                few_shot_response = vllm_server.generate_completions(
                    [few_shot_prompt], sampling_params
                )[0].text
                few_shot_reward = r1_zero_reward_fn(few_shot_response, answer)

                print(
                    question_only_response,
                    question_only_reward,
                    zero_shot_response,
                    zero_shot_reward,
                    few_shot_response,
                    few_shot_reward,
                )
                break
    finally:
        vllm_server.stop()


@app.local_entrypoint()
def modal_main() -> None:
    evaluate_gsm8k.remote()
