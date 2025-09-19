"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""

import argparse
import json
import os

script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
import time

import shortuuid
from datasets import Dataset, load_dataset
from PIL import Image
from tqdm import tqdm
from transformers import LlavaNextForConditionalGeneration

from ..model.choices import *
from ..model.kv_cache import initialize_past_key_values
from ..model.spec_model import SpecModel
from ..model.utils import *
from .seed_bench_prompt import build_prompt


def load_data(args):
    data = []
    with open(os.path.join(args.data_folder, "llava-seed-bench.jsonl"), "r") as f:
        lines = f.readlines()
        for l in lines:
            d = json.loads(l.strip())
            if not d["image"].startswith("SEED-Bench-image"):
                continue
            d["image"] = Image.open(os.path.join(args.data_folder, d["image"]))
            d["text"] = d["text"].partition("\n")[0]
            data.append(d)

    return Dataset.from_list(data).shuffle(seed=42).select(range(0, 100))


def baseline_forward(
    input_ids,
    model,
    tokenizer,
    tree_choices,
    logits_processor=None,
    max_steps=2048,
    **kwargs,
):
    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    # Avoid modifying the input_ids in-place
    input_ids = input_ids.clone()
    model.spec_layer.reset_kv()

    if hasattr(model, "tree_choices") and model.tree_choices == tree_choices:
        tree_buffers = model.tree_buffers
    else:
        try:
            tree_buffers = generate_tree_buffers(
                tree_choices,
                device=model.base_model.model.layers[-1].self_attn.q_proj.weight.device,
            )
            tree_buffers["retrieve_indices_head"] = tree_buffers["retrieve_indices"].to(
                model.base_model.lm_head.weight.device
            )
        except:
            tree_buffers = generate_tree_buffers(
                tree_choices,
                device=model.base_model.language_model.model.layers[
                    -1
                ].self_attn.q_proj.weight.device,
            )
            tree_buffers["retrieve_indices_head"] = tree_buffers["retrieve_indices"].to(
                model.base_model.language_model.lm_head.weight.device
            )
    model.tree_buffers = tree_buffers
    model.tree_choices = tree_choices

    # Initialize the past key and value states
    if hasattr(model, "past_key_values"):
        past_key_values = model.past_key_values
        past_key_values_data = model.past_key_values_data
        current_length_data = model.current_length_data
        # Reset the past key and value states
        current_length_data.zero_()
    else:
        try:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(model.base_model)
        except:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(model.base_model.language_model)
        model.past_key_values = past_key_values
        model.past_key_values_data = past_key_values_data
        model.current_length_data = current_length_data

    input_len = input_ids.shape[1]
    reset_tree_mode(model)

    outputs = model.base_model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
        **kwargs,
    )

    new_token = 0

    torch.cuda.synchronize()
    start_time = time.time()

    for idx in range(max_steps):
        if logits_processor is not None:
            logits = outputs.logits[:, -1]
            logits = logits_processor(None, logits)
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            input_id = torch.multinomial(probabilities, 1)
        else:
            input_id = outputs.logits[:, -1:].argmax(dim=-1)
        outputs = model.base_model(
            input_id, use_cache=True, past_key_values=past_key_values
        )
        input_ids = torch.cat([input_ids, input_id], dim=-1)

        if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            break
        if new_token > 1024:
            break
        # if input_ids.shape[1] > 1960:
        #     break
    torch.cuda.synchronize()
    end_time = time.time()

    return input_ids, new_token, idx, end_time - start_time


def run_eval(
    base_model_path,
    spec_model_path,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
    temperature,
    tree_choices,
):
    data = load_data(args)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(data) // (num_gpus_total // num_gpus_per_model)  # // 2
    ans_handles = []
    for i in range(0, len(data), chunk_size):
        ans_handles.append(
            get_answers_func(
                base_model_path,
                spec_model_path,
                model_id,
                data.select(range(i, i + chunk_size)),
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                temperature,
                tree_choices,
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
    base_model_path,
    spec_model_path,
    model_id,
    data,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    max_gpu_memory,
    temperature,
    tree_choices,
):
    # temperature = 0.0

    model = SpecModel.from_pretrained(
        base_model_path=base_model_path,
        spec_model_path=spec_model_path,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        # load_in_8bit=True,
        device_map="auto",
    )

    tokenizer = model.get_tokenizer()

    if temperature > 1e-5:
        logits_processor = prepare_logits_processor(temperature=temperature)
    else:
        logits_processor = None

    model.eval()
    print("Check model training state:", model.training)

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    print("CUDA VISIBLE DEVICES:", cuda_visible_devices)

    # warmup
    for _ in range(3):
        torch.manual_seed(0)

        turns = []
        idxs = []
        new_tokens = []
        wall_time = []

        model_inputs = build_prompt(data[0], args)

        torch.cuda.synchronize()
        start_time = time.time()

        output_ids, new_token, idx, _ = baseline_forward(
            **model_inputs,
            model=model,
            tokenizer=tokenizer,
            tree_choices=tree_choices,
            logits_processor=logits_processor,
        )

        torch.cuda.synchronize()
        total_time = time.time() - start_time
        output_ids = output_ids[0][len(model_inputs["input_ids"][0]) :]

        output_ids[output_ids > tokenizer.vocab_size] = 0
        output = tokenizer.decode(
            output_ids,
            spaces_between_special_tokens=False,
        )
        for special_token in tokenizer.special_tokens_map.values():
            if isinstance(special_token, list):
                for special_tok in special_token:
                    output = output.replace(special_tok, "")
            else:
                output = output.replace(special_token, "")
        output = output.strip()

        if output.startswith("Assistant:"):
            output = output.replace("Assistant:", "", 1).strip()

        turns.append(output)
        idxs.append(int(idx))
        new_tokens.append(int(new_token))
        wall_time.append(total_time)
    print("Warmup done")

    for d in tqdm(data):
        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            turns = []
            idxs = []
            new_tokens = []
            wall_time = []
            decode_time = []

            model_inputs = build_prompt(d, args)

            torch.cuda.synchronize()
            start_time = time.time()

            output_ids, new_token, idx, dec_time = baseline_forward(
                **model_inputs,
                model=model,
                tokenizer=tokenizer,
                tree_choices=tree_choices,
                logits_processor=logits_processor,
            )
            torch.cuda.synchronize()
            total_time = time.time() - start_time
            output_ids = output_ids[0][len(model_inputs["input_ids"][0]) :]

            output_ids[output_ids > tokenizer.vocab_size] = 0
            output = tokenizer.decode(
                output_ids,
                spaces_between_special_tokens=False,
            )
            for special_token in tokenizer.special_tokens_map.values():
                if isinstance(special_token, list):
                    for special_tok in special_token:
                        output = output.replace(special_tok, "")
                else:
                    output = output.replace(special_token, "")
            output = output.strip()

            if output.startswith("Assistant:"):
                output = output.replace("Assistant:", "", 1).strip()

            turns.append(output)
            idxs.append(int(idx))
            new_tokens.append(int(new_token))
            wall_time.append(total_time)
            decode_time.append(dec_time)

            choices.append(
                {
                    "index": i,
                    "turns": turns,
                    "idxs": idxs,
                    "new_tokens": new_tokens,
                    "wall_time": wall_time,
                    "decode_time": decode_time,
                }
            )

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": d["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec-model-path",
        type=str,
        default="down_checkpoints/LC70B",
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        default="/home/lyh/weights/hf/llama2chat/70B/",
        help="1",
    )
    parser.add_argument(
        "--load-in-8bit", action="store_false", help="Use 8-bit quantization"
    )
    parser.add_argument(
        "--model-id", type=str, default="sqa-llava-v1.6-vicuna-7b-fp16-baseline"
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--tree-choices",
        type=str,
        default="mc_sim_7b_63",
    )

    parser.add_argument("--data-folder", type=str, default="data/seed_bench")

    args = parser.parse_args()

    args.model = args.base_model_path
    args.model_id = args.model_id + "-temperature-" + str(args.temperature)
    args.tree_choices = eval(args.tree_choices)
    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"{args.bench_name}/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    run_eval(
        args.base_model_path,
        args.spec_model_path,
        args.model_id,
        args,
        args.question_begin,
        args.question_end,
        answer_file,
        args.max_new_token,
        args.num_choices,
        args.num_gpus_per_model,
        args.num_gpus_total,
        args.max_gpu_memory,
        args.temperature,
        args.tree_choices,
    )

    reorg_answer_file(answer_file)
