"""One explicit final-answer contract shared by new GRPO and evaluation."""

ANSWER_CONTRACT = "boxed-final-v1"


def answer_prompt(question, benchmark="gsm8k"):
    value = "numerical value, with no units or explanation" if benchmark == "gsm8k" else "mathematical expression"
    return (
        question.rstrip() + f"\n\nAfter your reasoning, put only your final {value} inside exactly one \\boxed{{...}}."
    )
