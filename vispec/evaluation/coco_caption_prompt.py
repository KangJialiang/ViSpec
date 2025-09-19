def build_prompt(data, args):
    from PIL import Image
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model)

    examples = [
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
    images = [data["image"]]

    examples.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Please provide a detailed description of the given image.",
                },
                {"type": "image"},
            ],
        }
    )

    # create the prompt input
    # prompt_input = '\n\n'.join(examples)
    prompt_input = processor.apply_chat_template(examples, add_generation_prompt=True)
    inputs = processor(images=images, text=prompt_input, return_tensors="pt").to(
        "cuda:0"
    )

    # return prompt_input
    return inputs
