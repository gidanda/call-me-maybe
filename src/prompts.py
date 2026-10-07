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
        "Choose the best available function for the user request and extract "
        "all required parameter values.\n"
        f"Available functions: {definitions}\n"
        f"User request: {user_prompt}\n"
        "Arguments must be directly usable by the selected function. A "
        "regular-expression argument must be a reusable bare pattern without "
        "slash delimiters. It must match one target occurrence, without "
        "surrounding .* or an enumeration of literals from the source.\n"
        "Return exactly one compact JSON object containing only name and "
        "parameters.\n"
        "Function call JSON:"
    )
