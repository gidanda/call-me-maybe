"""Single-pass JSON function-call constraint with schema branching."""

import json
from collections.abc import Callable
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, ConfigDict

from .json_values import (
    ValueState,
    initial_value_state,
    is_value_complete,
    transition_value_byte,
)
from .models import FunctionDefinition

CallPhase = Literal[
    "object_prefix",
    "function_name",
    "parameters_prefix",
    "parameter_key",
    "parameter_value",
    "object_suffix",
    "complete",
]
CallState = tuple[
    CallPhase,
    int,
    bytes,
    str | None,
    int,
    ValueState | None,
]
TransitionFunction = Callable[[CallState, bytes], CallState | None]

OBJECT_PREFIX = b'{"name":'
PARAMETERS_PREFIX = b',"parameters":{'
OBJECT_SUFFIX = b"}}"


class ConstraintContext(BaseModel):
    """Store validated, reusable data needed by the call grammar."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    functions_by_name: dict[str, FunctionDefinition]
    name_by_encoded: dict[bytes, str]
    encoded_names: tuple[bytes, ...]


def make_call_state(
    phase: CallPhase = "object_prefix",
    offset: int = 0,
    name_prefix: bytes = b"",
    selected_name: str | None = None,
    parameter_index: int = 0,
    value_state: ValueState | None = None,
) -> CallState:
    """Return one immutable position in the function-call grammar."""
    return (
        phase,
        offset,
        name_prefix,
        selected_name,
        parameter_index,
        value_state,
    )


def build_constraint_context(
    functions: list[FunctionDefinition],
) -> ConstraintContext:
    """Build reusable grammar data from validated function definitions."""
    if not functions:
        raise ValueError("at least one function definition is required")
    functions_by_name = {
        function.name: function for function in functions
    }
    name_by_encoded = {
        json.dumps(name, ensure_ascii=False).encode("utf-8"): name
        for name in functions_by_name
    }
    return ConstraintContext(
        functions_by_name=functions_by_name,
        name_by_encoded=name_by_encoded,
        encoded_names=tuple(name_by_encoded),
    )


def is_call_complete(state: CallState) -> bool:
    """Return whether the entire function-call object was consumed."""
    return state[0] == "complete"


def _selected_function(
    context: ConstraintContext,
    state: CallState,
) -> FunctionDefinition:
    """Return the schema selected in the function-name phase."""
    selected_name = state[3]
    if selected_name is None:
        raise RuntimeError("function schema requested before selection")
    return context.functions_by_name[selected_name]


def _parameter_key(
    context: ConstraintContext,
    state: CallState,
) -> bytes:
    """Return the fixed bytes for the current parameter key."""
    function = _selected_function(context, state)
    names = tuple(function.parameters)
    parameter_index = state[4]
    if parameter_index >= len(names):
        raise RuntimeError("parameter index is outside the schema")
    separator = b"," if parameter_index else b""
    encoded_name = json.dumps(
        names[parameter_index],
        ensure_ascii=False,
    ).encode("utf-8")
    return separator + encoded_name + b":"


def _parameter_type(
    context: ConstraintContext,
    state: CallState,
) -> str:
    """Return the declared type of the current parameter."""
    function = _selected_function(context, state)
    names = tuple(function.parameters)
    return function.parameters[names[state[4]]].type


def _phase_after_parameters_prefix(
    context: ConstraintContext,
    state: CallState,
) -> CallState:
    """Enter the first parameter or skip to the object suffix."""
    if _selected_function(context, state).parameters:
        return make_call_state(
            phase="parameter_key",
            selected_name=state[3],
        )
    return make_call_state(
        phase="object_suffix",
        selected_name=state[3],
    )


def _phase_after_value(
    context: ConstraintContext,
    state: CallState,
) -> CallState:
    """Enter the next parameter key or the final object suffix."""
    function = _selected_function(context, state)
    next_index = state[4] + 1
    if next_index < len(function.parameters):
        return make_call_state(
            phase="parameter_key",
            selected_name=state[3],
            parameter_index=next_index,
        )
    return make_call_state(
        phase="object_suffix",
        selected_name=state[3],
        parameter_index=next_index,
    )


def allowed_first_bytes(
    context: ConstraintContext,
    state: CallState,
) -> frozenset[int] | None:
    """Return possible token-leading bytes when they are known."""
    phase, offset, name_prefix, _, _, _ = state
    if phase == "object_prefix":
        return frozenset((OBJECT_PREFIX[offset],))
    if phase == "function_name":
        next_bytes = {
            name[len(name_prefix)]
            for name in context.encoded_names
            if name.startswith(name_prefix) and len(name) > len(name_prefix)
        }
        return frozenset(next_bytes)
    if phase == "parameters_prefix":
        return frozenset((PARAMETERS_PREFIX[offset],))
    if phase == "parameter_key":
        expected = _parameter_key(context, state)
        return frozenset((expected[offset],))
    if phase == "parameter_value":
        return None
    if phase == "object_suffix":
        return frozenset((OBJECT_SUFFIX[offset],))
    return frozenset()


def transition_call(
    context: ConstraintContext,
    state: CallState,
    token_bytes: bytes,
) -> CallState | None:
    """Consume a complete candidate token or reject it."""
    current = state

    for byte in token_bytes:
        while True:
            phase = current[0]
            offset = current[1]
            name_prefix = current[2]
            selected_name = current[3]
            parameter_index = current[4]
            value_state = current[5]

            if phase == "complete":
                return None

            if phase == "object_prefix":
                if OBJECT_PREFIX[offset] != byte:
                    return None
                next_offset = offset + 1
                if next_offset == len(OBJECT_PREFIX):
                    current = make_call_state(phase="function_name")
                else:
                    current = make_call_state(
                        phase="object_prefix",
                        offset=next_offset,
                    )
                break

            if phase == "function_name":
                candidate = name_prefix + bytes((byte,))
                if not any(
                    name.startswith(candidate)
                    for name in context.encoded_names
                ):
                    return None
                selected_name = context.name_by_encoded.get(candidate)
                if selected_name is None:
                    current = make_call_state(
                        phase="function_name",
                        name_prefix=candidate,
                    )
                else:
                    current = make_call_state(
                        phase="parameters_prefix",
                        selected_name=selected_name,
                    )
                break

            if phase == "parameters_prefix":
                if PARAMETERS_PREFIX[offset] != byte:
                    return None
                next_offset = offset + 1
                if next_offset == len(PARAMETERS_PREFIX):
                    current = _phase_after_parameters_prefix(
                        context,
                        current,
                    )
                else:
                    current = make_call_state(
                        phase="parameters_prefix",
                        offset=next_offset,
                        selected_name=selected_name,
                    )
                break

            if phase == "parameter_key":
                expected = _parameter_key(context, current)
                if expected[offset] != byte:
                    return None
                next_offset = offset + 1
                if next_offset == len(expected):
                    current = make_call_state(
                        phase="parameter_value",
                        selected_name=selected_name,
                        parameter_index=parameter_index,
                        value_state=initial_value_state(
                            _parameter_type(context, current)
                        ),
                    )
                else:
                    current = make_call_state(
                        phase="parameter_key",
                        offset=next_offset,
                        selected_name=selected_name,
                        parameter_index=parameter_index,
                    )
                break

            if phase == "parameter_value":
                if value_state is None:
                    raise RuntimeError("parameter value state is missing")
                next_value_state = transition_value_byte(value_state, byte)
                if next_value_state is not None:
                    current = make_call_state(
                        phase="parameter_value",
                        selected_name=selected_name,
                        parameter_index=parameter_index,
                        value_state=next_value_state,
                    )
                    break
                if not is_value_complete(value_state):
                    return None
                current = _phase_after_value(context, current)
                continue

            if phase == "object_suffix":
                if OBJECT_SUFFIX[offset] != byte:
                    return None
                next_offset = offset + 1
                if next_offset == len(OBJECT_SUFFIX):
                    current = make_call_state(
                        phase="complete",
                        selected_name=selected_name,
                        parameter_index=parameter_index,
                    )
                else:
                    current = make_call_state(
                        phase="object_suffix",
                        offset=next_offset,
                        selected_name=selected_name,
                        parameter_index=parameter_index,
                    )
                break

    return current


def build_transition(context: ConstraintContext) -> TransitionFunction:
    """Return a cached transition function bound to one function set."""

    @lru_cache(maxsize=250_000)
    def cached_transition(
        state: CallState,
        token_bytes: bytes,
    ) -> CallState | None:
        return transition_call(context, state, token_bytes)

    return cached_transition
