import argparse
import copy

parser = argparse.ArgumentParser(description="sp")
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=100)
parser.add_argument("--index", type=int, default=1)
parser.add_argument("--gpu_index", type=int, nargs="+", default=[0])
parser.add_argument("--outdir", type=str, default="outdir0")
parser.add_argument("--max_new_tokens", type=int, default=1024)
parser.add_argument("--model", type=str, default="llava-hf/llava-v1.6-vicuna-7b-hf")
parser.add_argument("--temperature", type=float, default=0.0)
args = parser.parse_args()
import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)[1:-1]


import json
from typing import Dict

import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)

bigname = args.model


def build_dataset_rank(
    processor,
    path,
):
    with open(os.path.join(path, "blip_laion_cc_sbu_558k.json")) as f:
        ds = json.load(f)

    ds = Dataset.from_list(ds)

    ds = ds.shuffle(seed=42)

    ds1: Dataset = ds.select(range(args.start, args.end))
    original_columns1 = ds1.column_names

    def preprocess_function(examples):
        conversation = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.",
                    },
                ],
            }
        ]

        for conv in examples["conversations"]:
            if conv["from"] == "human":
                assert conv["value"].endswith("\n<image>") or conv["value"].startswith(
                    "<image>\n"
                )
                conversation.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": conv["value"].strip().strip("<image>").strip(),
                            },
                            {"type": "image"},
                            {
                                "type": "text",
                                "text": "Please answer with at least 1000 words.",
                            },
                        ],
                    }
                )
            elif conv["from"] == "gpt":
                # conversation.append(
                #     {
                #         "role": "assistant",
                #         "content": [{"type": "text", "text": conv["value"]}],
                #     }
                # )
                pass
            else:
                raise ValueError("Unknown role")

        # create the prompt input
        # prompt_input = processor.apply_chat_template(conversation)
        prompt_input = processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

        outputs = {
            "image_files": [os.path.join(path, examples["image"])],
            "text": prompt_input,
            # "conversation": conversation,
        }

        return outputs

    ds1 = ds1.map(
        preprocess_function,
        batched=False,
        num_proc=2,  # it will be faster if you set it to 2
        remove_columns=original_columns1,
        load_from_cache_file=False,
        # writer_batch_size=100,  # avoid pyarrow overflow
    )

    return ds1


processor = AutoProcessor.from_pretrained(bigname, use_fast=True)
ds = build_dataset_rank(processor, "LLaVA-Pretrain/")
print(ds)
bigmodel = AutoModelForImageTextToText.from_pretrained(
    bigname, device_map="auto", torch_dtype=torch.float16
)
bigmodel.eval()


@torch.no_grad()
def ge(data: Dict):
    images = [Image.open(f) for f in data["image_files"]]
    data = processor(images=images, text=data["text"], return_tensors="pt").to(
        bigmodel.device
    )

    outs_big = bigmodel.generate(
        **data,
        output_hidden_states=True,
        return_dict_in_generate=True,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature != 0,
        temperature=args.temperature,
    )

    inputs_embeds_big = [x[0] for x in outs_big.hidden_states]
    inputs_embeds_big = torch.cat(inputs_embeds_big, dim=1)
    hidden_state_big = [x[-1] for x in outs_big.hidden_states]
    hidden_state_big = torch.cat(hidden_state_big, dim=1)

    image_mask = (
        outs_big.sequences == processor.tokenizer.added_tokens_encoder["<image>"]
    )[..., :-1]
    loss_mask = torch.ones_like(outs_big.sequences[:, :-1], dtype=bool)
    loss_mask[:, : data["input_ids"].shape[-1] - 1] = 0
    td = {
        # "input_ids": input_ids.cpu()[0],
        "inputs_embeds": inputs_embeds_big.cpu()[0],
        "hidden_state": hidden_state_big.cpu()[0],
        "loss_mask": loss_mask.cpu()[0],
        "image_mask": image_mask.cpu()[0],
    }
    return td


outdir = f"{args.outdir}/{args.index}"
if not os.path.exists(outdir):
    os.makedirs(outdir)


def writedata(name, data_point, idx):
    if not os.path.exists(name):
        os.makedirs(name)
    torch.save(data_point, f"{name}/data_{idx}.ckpt")


for i, data in enumerate(tqdm(ds)):
    outdata = ge(data)
    writedata(outdir, outdata, i)
