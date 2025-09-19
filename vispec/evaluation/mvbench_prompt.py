def build_prompt(data, args):
    from qwen_vl_utils import process_vision_info
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

    examples.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": data["video"],
                    "max_pixels": 360 * 420,
                    "max_frames": 8,
                },
                {
                    "type": "text",
                    "text": data["question"],
                },
                {
                    "type": "text",
                    "text": "Please answer with an explanation.",
                },
            ],
        }
    )

    # create the prompt input
    prompt_input = processor.apply_chat_template(examples, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        examples, return_video_kwargs=True
    )
    inputs = processor(
        text=prompt_input,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        **video_kwargs,
    ).to("cuda:0")

    return inputs
