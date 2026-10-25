"""
NLP_InstructionTuning - Assignment 9.5: From Prompting to Instruction Tuning

Compares how a base language model (GPT-2, which was NOT instruction-tuned) behaves:
  Part 1 - Prompt-only (zero-shot): just the instruction and input
  Part 2 - Simulated instruction tuning (few-shot): instruction/input/output examples first

All prompts and outputs are printed and saved to results.txt.
"""

from datetime import datetime
from transformers import pipeline, set_seed

MODEL_NAME = "gpt2"        # base model, not instruction-tuned
MAX_NEW_TOKENS = 40
OUTPUT_FILE = "results.txt"

PARAGRAPH = (
    "The city council voted on Tuesday to expand the public bus system. "
    "The plan adds three new routes, extends evening service until midnight, "
    "and lowers fares for students and seniors. Council members said the changes "
    "should reduce traffic downtown and make it easier for people without cars to get to work."
)

TASKS = [
    {
        "name": "Translation",
        "prompt_only": "Translate to Spanish: How are you?",
        "instruction": "Translate to Spanish",
        "examples": [
            ("Hello", "Hola"),
            ("Good morning", "Buenos días"),
            ("Thank you", "Gracias"),
        ],
        "test_input": "How are you?",
    },
    {
        "name": "Summarization",
        "prompt_only": f"Summarize the following paragraph in one sentence: {PARAGRAPH}",
        "instruction": "Summarize in one sentence",
        "examples": [
            (
                "The school board met Monday night and approved a new lunch menu. "
                "The menu includes more fresh fruit and vegetables and removes sugary drinks. "
                "Parents had asked for healthier options for over a year.",
                "The school board approved a healthier lunch menu after parents asked for it.",
            ),
            (
                "Heavy rain caused flooding on several roads Saturday. "
                "Police closed two highways and asked drivers to stay home. "
                "No injuries were reported.",
                "Flooding from heavy rain closed roads Saturday, but no one was hurt.",
            ),
        ],
        "test_input": PARAGRAPH,
    },
    {
        "name": "Explain to a child",
        "prompt_only": "Explain gravity to a 6-year-old.",
        "instruction": "Explain to a 6-year-old",
        "examples": [
            ("Rain", "Rain is water that falls from clouds when they get too full, like a sponge dripping."),
            ("The Sun", "The Sun is a giant ball of fire in the sky that keeps us warm and gives us light."),
        ],
        "test_input": "Gravity",
    },
]


def build_few_shot_prompt(task):
    """Build an instruction-tuning style prompt from the task's examples."""
    blocks = [
        f"Instruction: {task['instruction']}\nInput: {inp}\nOutput: {out}"
        for inp, out in task["examples"]
    ]
    blocks.append(f"Instruction: {task['instruction']}\nInput: {task['test_input']}\nOutput:")
    return "\n\n".join(blocks)


def clean_output(text):
    """Cut the generation off where the model starts writing a new example."""
    text = text.lstrip()  # base models often start with blank lines
    for stop in ["\nInstruction:", "\n\n"]:
        if stop in text:
            text = text.split(stop)[0]
    return text.strip()


def generate(generator, prompt):
    result = generator(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,  # greedy decoding so results are repeatable
        return_full_text=False,
        pad_token_id=generator.tokenizer.eos_token_id,
    )
    return clean_output(result[0]["generated_text"])


def main():
    set_seed(42)
    print(f"Loading {MODEL_NAME}...")
    generator = pipeline("text-generation", model=MODEL_NAME)

    lines = [
        "NLP_InstructionTuning - Results",
        f"Model: {MODEL_NAME}   Run: {datetime.now():%Y-%m-%d %H:%M}",
        "=" * 70,
    ]

    for part, label in [("prompt_only", "PART 1: PROMPT-ONLY (ZERO-SHOT)"),
                        ("few_shot", "PART 2: SIMULATED INSTRUCTION TUNING (FEW-SHOT)")]:
        lines += ["", label, "=" * 70]
        for task in TASKS:
            prompt = task["prompt_only"] if part == "prompt_only" else build_few_shot_prompt(task)
            output = generate(generator, prompt)
            lines += [
                "",
                f"--- Task: {task['name']} ---",
                "PROMPT:",
                prompt,
                "",
                "MODEL OUTPUT:",
                output if output else "(empty output)",
            ]
            lines.append("-" * 70)

    report = "\n".join(lines)
    print(report)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
