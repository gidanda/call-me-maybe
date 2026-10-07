"""Single-pass JSON function-call constraint with schema branching."""

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

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


@dataclass(frozen=True, slots=True)
class CallState:
    """Store one immutable position in the function-call grammar."""

    phase: CallPhase = "object_prefix"
    offset: int = 0
    name_prefix: bytes = b""
    selected_name: str | None = None
    parameter_index: int = 0
    value_state: ValueState | None = None


class FunctionCallConstraint:
    """Constrain one complete call and branch after function selection."""

    _OBJECT_PREFIX = b'{"name":'
    _PARAMETERS_PREFIX = b',"parameters":{'
    _OBJECT_SUFFIX = b"}}"

    def __init__(self, functions: list[FunctionDefinition]) -> None:
        """Store function schemas and their JSON-encoded names."""
        if not functions:
            raise ValueError("at least one function definition is required")
        self._functions_by_name = {
            function.name: function for function in functions
        }
        self._name_by_encoded = {
            json.dumps(name, ensure_ascii=False).encode("utf-8"): name
            for name in self._functions_by_name
        }
        self._encoded_names = tuple(self._name_by_encoded)

    def initial_state(self) -> CallState:
        """Return the state before the opening JSON brace."""
        return CallState()

    def is_complete(self, state: CallState) -> bool:
        """Return whether the entire function-call object was consumed."""
        return state.phase == "complete"

    def allowed_first_bytes(
        self,
        state: CallState,
    ) -> frozenset[int] | None:
        """Return possible first bytes, or None when broadly unconstrained."""
        if state.phase == "object_prefix":
            return frozenset((self._OBJECT_PREFIX[state.offset],))
        if state.phase == "function_name":
            next_bytes = {
                name[len(state.name_prefix)]
                for name in self._encoded_names
                if name.startswith(state.name_prefix)
                and len(name) > len(state.name_prefix)
            }
            return frozenset(next_bytes)
        if state.phase == "parameters_prefix":
            return frozenset((self._PARAMETERS_PREFIX[state.offset],))
        if state.phase == "parameter_key":
            expected = self._parameter_key(state)
            return frozenset((expected[state.offset],))
        if state.phase == "parameter_value":
            return None
        if state.phase == "object_suffix":
            return frozenset((self._OBJECT_SUFFIX[state.offset],))
        return frozenset()

    def _selected_function(self, state: CallState) -> FunctionDefinition:
        """Return the schema selected in the function-name phase."""
        if state.selected_name is None:
            raise RuntimeError("function schema requested before selection")
        return self._functions_by_name[state.selected_name]

    def _parameter_key(self, state: CallState) -> bytes:
        """Return the fixed bytes for the current parameter key."""
        function = self._selected_function(state)
        names = tuple(function.parameters)
        if state.parameter_index >= len(names):
            raise RuntimeError("parameter index is outside the schema")
        separator = b"," if state.parameter_index else b""
        encoded_name = json.dumps(
            names[state.parameter_index],
            ensure_ascii=False,
        ).encode("utf-8")
        return separator + encoded_name + b":"

    def _parameter_type(self, state: CallState) -> str:
        """Return the declared type of the current parameter."""
        function = self._selected_function(state)
        names = tuple(function.parameters)
        return function.parameters[names[state.parameter_index]].type

    def _phase_after_parameters_prefix(self, state: CallState) -> CallState:
        """Enter the first parameter or skip to the object suffix."""
        function = self._selected_function(state)
        if function.parameters:
            return CallState(
                phase="parameter_key",
                selected_name=state.selected_name,
            )
        return CallState(
            phase="object_suffix",
            selected_name=state.selected_name,
        )

    def _phase_after_value(self, state: CallState) -> CallState:
        """Enter the next parameter key or the final object suffix."""
        function = self._selected_function(state)
        next_index = state.parameter_index + 1
        if next_index < len(function.parameters):
            return CallState(
                phase="parameter_key",
                selected_name=state.selected_name,
                parameter_index=next_index,
            )
        return CallState(
            phase="object_suffix",
            selected_name=state.selected_name,
            parameter_index=next_index,
        )

    @lru_cache(maxsize=250_000)
    def transition(
        self,
        state: CallState,
        token_bytes: bytes,
    ) -> CallState | None:
        """Consume a complete candidate token or reject it."""
        current = state

        for byte in token_bytes:
            while True:
                if current.phase == "complete":
                    return None

                if current.phase == "object_prefix":
                    if self._OBJECT_PREFIX[current.offset] != byte:
                        return None
                    next_offset = current.offset + 1
                    if next_offset == len(self._OBJECT_PREFIX):
                        current = CallState(phase="function_name")
                    else:
                        current = CallState(
                            phase="object_prefix",
                            offset=next_offset,
                        )
                    break

                if current.phase == "function_name":
                    candidate = current.name_prefix + bytes((byte,))
                    if not any(
                        name.startswith(candidate)
                        for name in self._encoded_names
                    ):
                        return None
                    selected_name = self._name_by_encoded.get(candidate)
                    if selected_name is None:
                        current = CallState(
                            phase="function_name",
                            name_prefix=candidate,
                        )
                    else:
                        current = CallState(
                            phase="parameters_prefix",
                            selected_name=selected_name,
                        )
                    break

                if current.phase == "parameters_prefix":
                    if self._PARAMETERS_PREFIX[current.offset] != byte:
                        return None
                    next_offset = current.offset + 1
                    if next_offset == len(self._PARAMETERS_PREFIX):
                        current = self._phase_after_parameters_prefix(current)
                    else:
                        current = CallState(
                            phase="parameters_prefix",
                            offset=next_offset,
                            selected_name=current.selected_name,
                        )
                    break

                if current.phase == "parameter_key":
                    expected = self._parameter_key(current)
                    if expected[current.offset] != byte:
                        return None
                    next_offset = current.offset + 1
                    if next_offset == len(expected):
                        current = CallState(
                            phase="parameter_value",
                            selected_name=current.selected_name,
                            parameter_index=current.parameter_index,
                            value_state=initial_value_state(
                                self._parameter_type(current)
                            ),
                        )
                    else:
                        current = CallState(
                            phase="parameter_key",
                            offset=next_offset,
                            selected_name=current.selected_name,
                            parameter_index=current.parameter_index,
                        )
                    break

                if current.phase == "parameter_value":
                    value_state = current.value_state
                    if value_state is None:
                        raise RuntimeError("parameter value state is missing")
                    next_value_state = transition_value_byte(
                        value_state,
                        byte,
                    )
                    if next_value_state is not None:
                        current = CallState(
                            phase="parameter_value",
                            selected_name=current.selected_name,
                            parameter_index=current.parameter_index,
                            value_state=next_value_state,
                        )
                        break
                    if not is_value_complete(value_state):
                        return None
                    current = self._phase_after_value(current)
                    continue

                if current.phase == "object_suffix":
                    if self._OBJECT_SUFFIX[current.offset] != byte:
                        return None
                    next_offset = current.offset + 1
                    if next_offset == len(self._OBJECT_SUFFIX):
                        current = CallState(
                            phase="complete",
                            selected_name=current.selected_name,
                            parameter_index=current.parameter_index,
                        )
                    else:
                        current = CallState(
                            phase="object_suffix",
                            offset=next_offset,
                            selected_name=current.selected_name,
                            parameter_index=current.parameter_index,
                        )
                    break

        return current
