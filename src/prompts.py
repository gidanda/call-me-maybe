"""Prompt construction for function-call generation."""

import json

from .models import FunctionDefinition


def build_generation_prompt(
    user_prompt: str,
    functions: list[FunctionDefinition],
) -> str:
    """Build a function-selection and argument-extraction prompt."""
    definitions = json.dumps(
        [function.model_dump(mode="json") for function in functions],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        "<|im_start|>system\n"
        "You translate user requests into function calls using only the "
        "available function definitions.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "Select the function that best matches the user request.\n"
        "Use the function descriptions and parameter definitions provided "
        "below.\n"
        "Extract a value for every required parameter.\n\n"
        f"Available functions:\n{definitions}\n\n"
        f"User request:\n{user_prompt}\n\n"
        "Before generating the function call:\n"
        "- Determine the complete operation requested by the user.\n"
        "- Output each parameter as the raw value expected by the function, "
        "without language-specific wrappers or explanatory text.\n"
        "- Preserve the exact scope expressed by words such as all, every, "
        "only, before, and after.\n"
        "- Prefer the simplest parameter value that directly represents the "
        "requested target. Do not add broad prefixes, broad suffixes, or "
        "redundant surrounding context.\n"
        "- If the selected function operates on matching items, the value "
        "that identifies a match must select exactly one target item. One "
        "match must never include multiple targets or surrounding content.\n"
        "- For all or every, do not widen a single match. Rely on the "
        "function to repeat the operation for each matching item.\n"
        "- Ensure that calling the selected function once with these "
        "parameters performs the complete requested operation and does not "
        "affect unrelated content.\n\n"
        "Generate exactly one function call JSON object containing only name "
        "and parameters.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )
