def get_question_text(problem):
    question = problem["question"]
    return question


def get_context_text(problem, use_caption):
    txt_context = problem["hint"]
    img_context = problem["caption"] if use_caption else ""
    context = " ".join([txt_context, img_context]).strip()
    if context == "":
        context = "N/A"
    return context


def get_choice_text(probelm, options):
    choices = probelm["choices"]
    choice_list = []
    for i, c in enumerate(choices):
        choice_list.append("({}) {}".format(options[i], c))
    choice_txt = " ".join(choice_list)
    # print(choice_txt)
    return choice_txt


def get_answer(problem, options):
    return options[problem["answer"]]


def get_lecture_text(problem):
    # # \\n: GPT-3 can generate the lecture with more tokens.
    # lecture = problem["lecture"].replace("\n", "\\n")
    lecture = problem["lecture"]
    return lecture


def get_solution_text(problem):
    # # \\n: GPT-3 can generate the solution with more tokens
    # solution = problem["solution"].replace("\n", "\\n")
    solution = problem["solution"]
    return solution


def create_one_example(
    format, question, context, choice, answer, lecture, solution, test_example=True
):

    input_format, output_format = format.split("-")

    ## Inputs
    if input_format == "CQM":
        input = f"Context: {context}\nQuestion: {question}\nOptions: {choice}\n"
    elif input_format == "QCM":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\n"
    # upper bound experiment
    elif input_format == "QCML":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture}\n"
    elif input_format == "QCME":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {solution}\n"
    elif input_format == "QCMLE":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture} {solution}\n"

    elif input_format == "QCLM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture}\nOptions: {choice}\n"
    elif input_format == "QCEM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {solution}\nOptions: {choice}\n"
    elif input_format == "QCLEM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture} {solution}\nOptions: {choice}\n"

    # Outputs
    if test_example:
        output = "Answer:"
    elif output_format == "A":
        output = f"Answer: The answer is {answer}."

    elif output_format == "AL":
        output = f"Answer: The answer is {answer}. BECAUSE: {solution}"
    elif output_format == "AE":
        output = f"Answer: The answer is {answer}. BECAUSE: {lecture}"
    elif output_format == "ALE":
        output = f"Answer: The answer is {answer}. BECAUSE: {lecture} {solution}"
    elif output_format == "AEL":
        output = f"Answer: The answer is {answer}. BECAUSE: {solution} {lecture}"

    elif output_format == "LA":
        output = f"Answer: {lecture} The answer is {answer}."
    elif output_format == "EA":
        output = f"Answer: {solution} The answer is {answer}."
    elif output_format == "LEA":
        output = f"Answer: {lecture} {solution} The answer is {answer}."
    elif output_format == "ELA":
        output = f"Answer: {solution} {lecture} The answer is {answer}."

    text = input + output
    text = text.replace("  ", " ").strip()
    if text.endswith("BECAUSE:"):
        text = text.replace("BECAUSE:", "").strip()
    return text


def build_prompt(problems, shot_qids, test_qid, args):
    from transformers import AutoProcessor

    if "Qwen2.5-VL" in args.model:
        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        processor = AutoProcessor.from_pretrained(
            args.model, use_fast=True, min_pixels=min_pixels, max_pixels=max_pixels
        )
    else:
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
    images = []

    # n-shot training examples
    for qid in shot_qids:
        question = get_question_text(problems[qid])
        context = get_context_text(problems[qid], args.use_caption)
        choice = get_choice_text(problems[qid], args.options)
        answer = get_answer(problems[qid], args.options)
        lecture = get_lecture_text(problems[qid])
        solution = get_solution_text(problems[qid])

        train_example = create_one_example(
            args.prompt_format,
            question,
            context,
            choice,
            answer,
            lecture,
            solution,
            test_example=False,
        )
        # examples.append(train_example)
        train_example = train_example.split("Answer:")
        examples.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{train_example[0]}Answer:"},
                    {"type": "image"},
                ],
            }
        )
        examples.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"{train_example[1].strip()}"},
                ],
            },
        )
        images.append(problems[qid]["image"])

    # test example
    question = get_question_text(problems[test_qid])
    context = get_context_text(problems[test_qid], args.use_caption)
    choice = get_choice_text(problems[test_qid], args.options)
    answer = get_answer(problems[test_qid], args.options)
    lecture = get_lecture_text(problems[test_qid])
    solution = get_solution_text(problems[test_qid])

    test_example = create_one_example(
        args.prompt_format,
        question,
        context,
        choice,
        answer,
        lecture,
        solution,
        test_example=True,
    )
    # examples.append(test_example)
    test_example = test_example.replace(
        "Answer:",
        'Your answer should begin with "The answer is". Please answer with an explanation. Answer:',
    )
    examples.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"{test_example}",
                },
                {"type": "image"},
            ],
        }
    )
    images.append(problems[test_qid]["image"])

    # create the prompt input
    prompt_input = processor.apply_chat_template(examples, add_generation_prompt=True)
    inputs = processor(images=images, text=prompt_input, return_tensors="pt").to(
        "cuda:0"
    )

    return inputs
