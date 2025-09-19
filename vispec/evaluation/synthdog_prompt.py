def build_prompt(data, args):
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
                    "text": "Perform an OCR task on the provided image. Please extract the text accurately and ensure the response is comprehensive and well-structured.",
                },
                {"type": "image"},
            ],
        }
    )

    # create the prompt input
    prompt_input = processor.apply_chat_template(examples, add_generation_prompt=True)
    inputs = processor(images=images, text=prompt_input, return_tensors="pt").to(
        "cuda:0"
    )

    return inputs
