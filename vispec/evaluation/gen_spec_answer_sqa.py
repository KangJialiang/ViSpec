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
from datasets import load_dataset
from tqdm import tqdm

from ..model.kv_cache import initialize_past_key_values
from ..model.utils import *
from .scienceqa_prompt import build_prompt


def load_data(args):
    problems = json.load(open(os.path.join(args.data_root, "problems.json")))
    pid_splits = json.load(open(os.path.join(args.data_root, "pid_splits.json")))
    captions = json.load(open(args.caption_file))["captions"]
    data = load_dataset("derek-thomas/ScienceQA")

    random.seed(42)

    for qid in problems:
        problems[qid]["caption"] = captions[qid] if qid in captions else ""
    for split_name, qids in pid_splits.items():
        if split_name not in ["train", "val", "test"]:
            continue
        if split_name == "val":
            split_name = "validation"
        split = data[split_name]
        for i, qid in enumerate(qids):
            problems[qid]["image"] = split[i]["image"]

    for split_name, qids in pid_splits.items():
        new_qids = []
        for qid in qids:
            if problems[qid]["image"] is not None:
                new_qids.append(qid)
        pid_splits[split_name] = new_qids

    qids = pid_splits["%s" % (args.test_split)]
    qids = qids[: args.test_number] if args.test_number > 0 else qids
    qids = random.sample(qids, 100)
    print(f"number of test problems: {len(qids)}\n")

    # pick up shot examples from the training set
    shot_qids = args.shot_qids
    train_qids = pid_splits["train"]
    if shot_qids == None:
        assert args.shot_number >= 0 and args.shot_number <= 32
        shot_qids = random.sample(train_qids, args.shot_number)  # random sample
    else:
        shot_qids = [str(qid) for qid in shot_qids]
        for qid in shot_qids:
            assert qid in train_qids  # check shot_qids
    print("training question ids for prompting: ", shot_qids, "\n")

    return problems, qids, shot_qids


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
    args,
):

    problems, qids, shot_qids = load_data(args)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(qids) // (num_gpus_total // num_gpus_per_model)  # // 2
    ans_handles = []
    for i in range(0, len(qids), chunk_size):
        ans_handles.append(
            get_answers_func(
                base_model_path,
                spec_model_path,
                model_id,
                problems,
                shot_qids,
                qids[i : i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                temperature,
                args,
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
    base_model_path,
    spec_model_path,
    model_id,
    problems,
    shot_qids,
    qids,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    max_gpu_memory,
    temperature,
    args,
):
    # temperature = 0.0

    if args.use_ours:
        from ..model.spec_model_ours import SpecModel

        model = SpecModel.from_pretrained(
            base_model_path=base_model_path,
            spec_model_path=spec_model_path,
            total_token=args.total_token,
            depth=args.depth,
            top_k=args.top_k,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            # load_in_8bit=True,
            device_map="auto",
            num_q=args.num_q,
        )
    elif args.use_medusa:
        from ..model.spec_model_medusa import SpecModel

        model = SpecModel.from_pretrained(
            base_model_path=base_model_path,
            spec_model_path=spec_model_path,
            total_token=args.total_token,
            depth=args.depth,
            top_k=args.top_k,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            # load_in_8bit=True,
            device_map="auto",
        )
    else:
        from ..model.spec_model import SpecModel

        model = SpecModel.from_pretrained(
            base_model_path=base_model_path,
            spec_model_path=spec_model_path,
            total_token=args.total_token,
            depth=args.depth,
            top_k=args.top_k,
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

    qid = qids[0]

    # warmup
    for _ in range(3):
        torch.manual_seed(0)

        turns = []
        idxs = []
        new_tokens = []
        wall_time = []

        # choices = problems[qid]["choices"]
        # answer = problems[qid]["answer"]  # 0, 1, ..., 4
        # label = args.options[answer]  # 'A', ..., 'E'

        # generate prompt
        model_inputs = build_prompt(problems, shot_qids, qid, args)

        torch.cuda.synchronize()
        start_time = time.time()

        output_ids, new_token, idx = model.specgenerate(
            **model_inputs, temperature=temperature, log=True
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

    for qid in tqdm(qids):
        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            turns = []
            idxs = []
            new_tokens = []
            wall_time = []
            acceptance_length = []
            decode_time = []

            # generate prompt
            model_inputs = build_prompt(problems, shot_qids, qid, args)

            torch.cuda.synchronize()
            start_time = time.time()

            output_ids, new_token, idx, accp_len, dec_time = model.specgenerate(
                **model_inputs,
                temperature=temperature,
                log=True,
                return_acceptance_len=True,
                return_decode_time=True,
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
            acceptance_length.extend(accp_len)
            decode_time.append(dec_time)

            choices.append(
                {
                    "index": i,
                    "turns": turns,
                    "idxs": idxs,
                    "new_tokens": new_tokens,
                    "wall_time": wall_time,
                    "acceptance_length": acceptance_length,
                    "decode_time": decode_time,
                }
            )

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                # "question_id": question["question_id"],
                "question_id": qid,
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
    parser.add_argument("--model-id", type=str, default="sqa-llava-v1.6-vicuna-7b-fp16")
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
        "--total-token",
        type=int,
        default=30,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
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

    parser.add_argument("--data_root", type=str, default="vispec/data/scienceqa")
    # parser.add_argument("--output_root", type=str, default="../results")
    parser.add_argument(
        "--caption_file", type=str, default="vispec/data/scienceqa/captions.json"
    )
    # parser.add_argument('--model', type=str, default='gpt3')
    parser.add_argument("--options", type=list, default=["A", "B", "C", "D", "E"])
    # user options
    parser.add_argument("--label", type=str, default="exp0")
    parser.add_argument(
        "--test_split", type=str, default="val", choices=["test", "val", "minival"]
    )
    parser.add_argument(
        "--test_number",
        type=int,
        default=10,
        help="GPT-3 is expensive. -1 for whole val/test set",
    )
    parser.add_argument(
        "--use_caption", action="store_true", help="use image captions or not"
    )
    # parser.add_argument(
    #     "--save_every",
    #     type=int,
    #     default=10,
    #     help="Save the result with every n examples.",
    # )
    # parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--prompt_format",
        type=str,
        default="CQM-A",
        choices=[
            "CQM-A",
            "CQM-LA",
            "CQM-EA",
            "CQM-LEA",
            "CQM-ELA",
            "CQM-AL",
            "CQM-AE",
            "CQM-ALE",
            "QCM-A",
            "QCM-LA",
            "QCM-EA",
            "QCM-LEA",
            "QCM-ELA",
            "QCM-AL",
            "QCM-AE",
            "QCM-ALE",
            "QCML-A",
            "QCME-A",
            "QCMLE-A",
            "QCLM-A",
            "QCEM-A",
            "QCLEM-A",
            "QCML-AE",
        ],
        help="prompt format template",
    )
    parser.add_argument(
        "--shot_number", type=int, default=3, help="Number of n-shot training examples."
    )
    parser.add_argument(
        "--shot_qids", type=list, default=None, help="Question indexes of shot examples"
    )
    # parser.add_argument('--seed', type=int, default=10, help='random seed')
    # GPT-3 settings
    # parser.add_argument("--engine", type=str, default="text-davinci-002")
    # parser.add_argument("--temperature", type=float, default=0.0)
    # parser.add_argument('--max_tokens',
    #                     type=int,
    #                     default=512,
    #                     help='The maximum number of tokens allowed for the generated answer.')
    # parser.add_argument('--top_p', type=float, default=1.0)
    # parser.add_argument("--frequency_penalty", type=float, default=0.0)
    # parser.add_argument("--presence_penalty", type=float, default=0.0)

    parser.add_argument("--image-fc", type=bool, default=False)
    parser.add_argument("--use-ours", type=bool, default=False)
    parser.add_argument("--num-q", type=int, default=2)
    parser.add_argument("--use-medusa", type=bool, default=False)

    args = parser.parse_args()

    args.model = args.base_model_path
    args.model_id = args.model_id + "-temperature-" + str(args.temperature)
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
        args,
    )

    reorg_answer_file(answer_file)
