"""Application pipeline for constrained function-call generation."""

import json
from collections.abc import Callable, Mapping
from typing import cast

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]

from .call_constraint import (
    ConstraintContext,
    TransitionFunction,
    allowed_first_bytes,
    build_constraint_context,
    build_transition,
    is_call_complete,
    make_call_state,
)
from .decoder import constrained_decode
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
    context: ConstraintContext,
    encode_prompt: Callable[[str], list[list[int]]],
    get_logits: Callable[[list[int]], list[float]],
    transition: TransitionFunction,
    token_bytes_by_id: Mapping[int, bytes],
    token_ids_by_first_byte: Mapping[int, tuple[int, ...]],
) -> FunctionCallingResult:
    """Generate and validate one result for one input prompt."""
    model_prompt = build_generation_prompt(prompt.prompt, functions)
    encoded = encode_prompt(model_prompt)
    if len(encoded) != 1:
        raise RuntimeError("model prompt did not encode to one batch")
    generated = constrained_decode(
        get_logits=get_logits,
        prompt_token_ids=encoded[0],
        initial_state=make_call_state(),
        transition=transition,
        is_complete=is_call_complete,
        allowed_first_bytes=lambda state: allowed_first_bytes(
            context,
            state,
        ),
        token_bytes_by_id=token_bytes_by_id,
        token_ids_by_first_byte=token_ids_by_first_byte,
    )

    try:
        parsed = json.loads(generated)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "constrained generation produced invalid JSON"
        ) from error

    call = validate_function_call(parsed, context.functions_by_name)
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
    context = build_constraint_context(functions)
    transition = build_transition(context)

    def encode_prompt(text: str) -> list[list[int]]:
        """Convert the SDK tensor result into typed batch token IDs."""
        return cast(list[list[int]], model.encode(text).tolist())

    def get_logits(input_ids: list[int]) -> list[float]:
        """Read next-token scores through the SDK public API."""
        return cast(
            list[float],
            model.get_logits_from_input_ids(input_ids),
        )

    results: list[FunctionCallingResult] = []
    for prompt in prompts:
        result = generate_result(
            prompt=prompt,
            functions=functions,
            context=context,
            encode_prompt=encode_prompt,
            get_logits=get_logits,
            transition=transition,
            token_bytes_by_id=token_bytes_by_id,
            token_ids_by_first_byte=token_ids_by_first_byte,
        )
        results.append(result)
    return results
