"""Application pipeline for constrained function-call generation."""

import json
from collections.abc import Mapping

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]

from .call_constraint import FunctionCallConstraint
from .decoder import DecodingModel, constrained_decode
from .models import (
    FunctionCallingResult,
    FunctionDefinition,
    PromptInput,
    validate_function_call,
)
from .prompts import build_generation_prompt
from .vocabulary import index_tokens_by_first_byte, load_token_bytes


def generate_result(
    prompt: PromptInput,
    functions: list[FunctionDefinition],
    functions_by_name: dict[str, FunctionDefinition],
    model: DecodingModel,
    constraint: FunctionCallConstraint,
    token_bytes_by_id: Mapping[int, bytes],
    token_ids_by_first_byte: Mapping[int, tuple[int, ...]],
) -> FunctionCallingResult:
    """Generate and validate one result for one input prompt."""
    model_prompt = build_generation_prompt(prompt.prompt, functions)
    generated = constrained_decode(
        model=model,
        prompt=model_prompt,
        constraint=constraint,
        token_bytes_by_id=token_bytes_by_id,
        token_ids_by_first_byte=token_ids_by_first_byte,
    )

    try:
        parsed = json.loads(generated)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "constrained generation produced invalid JSON"
        ) from error

    call = validate_function_call(parsed, functions_by_name)
    return FunctionCallingResult(
        prompt=prompt.prompt,
        name=call.name,
        parameters=call.parameters,
    )


def generate_all_results(
    prompts: list[PromptInput],
    functions: list[FunctionDefinition],
) -> list[FunctionCallingResult]:
    """Initialize shared resources once and process all input prompts."""
    model = Small_LLM_Model()
    token_bytes_by_id = load_token_bytes(model.get_path_to_vocab_file())
    token_ids_by_first_byte = index_tokens_by_first_byte(token_bytes_by_id)
    functions_by_name = {
        function.name: function for function in functions
    }
    constraint = FunctionCallConstraint(functions)

    results: list[FunctionCallingResult] = []
    for prompt in prompts:
        result = generate_result(
            prompt=prompt,
            functions=functions,
            functions_by_name=functions_by_name,
            model=model,
            constraint=constraint,
            token_bytes_by_id=token_bytes_by_id,
            token_ids_by_first_byte=token_ids_by_first_byte,
        )
        results.append(result)
    return results
