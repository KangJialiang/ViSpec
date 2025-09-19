import json
import os

import numpy as np
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")

baseline_dir = "baseline_results"
result_dir = "results"
for model in [
    "llava",
    "llava_13b",
    "qwen_3b",
    "qwen",
    "llava_1.5",
]:
    for dataset in [
        "sqa",
        "coco_caption",
        "gqa",
        "mme",
        "mmvet",
        "seed_bench",
        "textvqa",
        "vizwiz",
        "vqav2",
        "synthdog",
        "hr_bench",
        "hr_bench_8k",
        "msvd_qa",
    ]:
        for t in (
            1.0,
            0.0,
        ):
            for method in (
                "2q",
                "2q_wo_pretrain",
                "medusa",
                "ea",
                "5q",
                "17q",
                "65q",
            ):
                try:
                    jsonl_file_base = f"{baseline_dir}/{dataset}_test/baseline_{model}/test-temperature-{t:.1f}.jsonl"
                    jsonl_file = f"{result_dir}/{dataset}_test/{model}_{method}/test-temperature-{t:.1f}.jsonl"
                    data = []
                    with open(jsonl_file, "r", encoding="utf-8") as file:
                        print(jsonl_file)
                        for line in file:
                            json_obj = json.loads(line)
                            data.append(json_obj)

                    speeds = []
                    acc_len = []
                    new_tokens = []
                    for datapoint in data:
                        # qid = datapoint["question_id"]
                        answer = datapoint["choices"][0]["turns"]
                        tokens = sum(datapoint["choices"][0]["new_tokens"])
                        times = sum(datapoint["choices"][0]["wall_time"])
                        # times = sum(datapoint["choices"][0]["decode_time"])
                        acc_len += datapoint["choices"][0]["acceptance_length"]
                        speeds.append(tokens / times)

                        new_tokens.append(tokens)

                    print(f"acc len {sum(acc_len) / len(acc_len)}")
                    print(f"new tok {sum(new_tokens) / len(new_tokens)}")

                    data = []
                    with open(jsonl_file_base, "r", encoding="utf-8") as file:
                        # print(jsonl_file_base)
                        for line in file:
                            json_obj = json.loads(line)
                            data.append(json_obj)

                    total_time = 0
                    total_token = 0
                    speeds0 = []
                    for datapoint in data:
                        # qid = datapoint["question_id"]
                        answer = datapoint["choices"][0]["turns"]
                        tokens = 0
                        for i in answer:
                            tokens += len(tokenizer(i).input_ids) - 1
                        times = sum(datapoint["choices"][0]["wall_time"])
                        # times = sum(datapoint["choices"][0]["decode_time"])
                        speeds0.append(tokens / times)
                        total_time += times
                        total_token += tokens

                    # print('speed',np.array(speeds).mean())
                    # print('speed0',np.array(speeds0).mean())
                    print("ratio", np.array(speeds).mean() / np.array(speeds0).mean())
                except Exception as e:
                    # print(f"Error processing {model} {dataset}: {e}")
                    pass
