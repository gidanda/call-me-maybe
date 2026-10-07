"""Incremental byte-level constraints for primitive JSON values."""

from dataclasses import dataclass, replace
from typing import Literal

NumberPhase = Literal[
    "start",
    "sign",
    "zero",
    "integer",
    "dot",
    "fraction",
    "exponent_mark",
    "exponent_sign",
    "exponent",
]


@dataclass(frozen=True, slots=True)
class StringState:
    """Track parsing of one JSON string."""

    started: bool = False
    escaped: bool = False
    unicode_digits_remaining: int = 0
    pending_utf8: bytes = b""
    complete: bool = False


@dataclass(frozen=True, slots=True)
class NumberState:
    """Track parsing of one JSON number."""

    phase: NumberPhase = "start"


@dataclass(frozen=True, slots=True)
class BooleanState:
    """Track parsing of one JSON boolean."""

    generated: bytes = b""


ValueState = StringState | NumberState | BooleanState


def initial_value_state(type_name: str) -> ValueState:
    """Return an initial state for a supported JSON primitive type."""
    if type_name == "string":
        return StringState()
    if type_name == "number":
        return NumberState()
    if type_name == "boolean":
        return BooleanState()
    raise ValueError(f"unsupported parameter type: {type_name}")


def _is_hex_digit(byte: int) -> bool:
    """Return whether a byte is an ASCII hexadecimal digit."""
    return (
        ord("0") <= byte <= ord("9")
        or ord("a") <= byte <= ord("f")
        or ord("A") <= byte <= ord("F")
    )


def _append_utf8_byte(pending: bytes, byte: int) -> bytes | None:
    """Return an unfinished UTF-8 suffix, or reject invalid bytes."""
    candidate = pending + bytes((byte,))
    try:
        candidate.decode("utf-8")
    except UnicodeDecodeError as error:
        if (
            error.reason == "unexpected end of data"
            and error.end == len(candidate)
        ):
            return candidate
        return None
    return b""


def _transition_string(state: StringState, byte: int) -> StringState | None:
    """Consume one byte of a JSON string."""
    if state.complete:
        return None
    if not state.started:
        if byte != ord('"'):
            return None
        return replace(state, started=True)
    if state.pending_utf8:
        pending = _append_utf8_byte(state.pending_utf8, byte)
        if pending is None:
            return None
        return replace(state, pending_utf8=pending)
    if state.unicode_digits_remaining:
        if not _is_hex_digit(byte):
            return None
        return replace(
            state,
            unicode_digits_remaining=state.unicode_digits_remaining - 1,
        )
    if state.escaped:
        if byte in b'"\\/bfnrt':
            return replace(state, escaped=False)
        if byte == ord("u"):
            return replace(
                state,
                escaped=False,
                unicode_digits_remaining=4,
            )
        return None
    if byte == ord('"'):
        return replace(state, complete=True)
    if byte == ord("\\"):
        return replace(state, escaped=True)
    if byte < 0x20:
        return None
    if byte < 0x80:
        return state

    pending = _append_utf8_byte(b"", byte)
    if pending is None:
        return None
    return replace(state, pending_utf8=pending)


def _transition_number(state: NumberState, byte: int) -> NumberState | None:
    """Consume one byte of a JSON number."""
    character = chr(byte)
    phase = state.phase
    if phase == "start":
        if character == "-":
            return NumberState("sign")
        if character == "0":
            return NumberState("zero")
        if "1" <= character <= "9":
            return NumberState("integer")
        return None
    if phase == "sign":
        if character == "0":
            return NumberState("zero")
        if "1" <= character <= "9":
            return NumberState("integer")
        return None
    if phase in ("zero", "integer"):
        if phase == "integer" and "0" <= character <= "9":
            return state
        if character == ".":
            return NumberState("dot")
        if character in "eE":
            return NumberState("exponent_mark")
        return None
    if phase == "dot":
        if "0" <= character <= "9":
            return NumberState("fraction")
        return None
    if phase == "fraction":
        if "0" <= character <= "9":
            return state
        if character in "eE":
            return NumberState("exponent_mark")
        return None
    if phase == "exponent_mark":
        if character in "+-":
            return NumberState("exponent_sign")
        if "0" <= character <= "9":
            return NumberState("exponent")
        return None
    if phase == "exponent_sign":
        if "0" <= character <= "9":
            return NumberState("exponent")
        return None
    if phase == "exponent" and "0" <= character <= "9":
        return state
    return None


def _transition_boolean(
    state: BooleanState,
    byte: int,
) -> BooleanState | None:
    """Consume one byte of a JSON boolean."""
    candidate = state.generated + bytes((byte,))
    if not any(value.startswith(candidate) for value in (b"true", b"false")):
        return None
    return BooleanState(candidate)


def transition_value_byte(state: ValueState, byte: int) -> ValueState | None:
    """Consume one byte using the state-specific primitive grammar."""
    if isinstance(state, StringState):
        return _transition_string(state, byte)
    if isinstance(state, NumberState):
        return _transition_number(state, byte)
    return _transition_boolean(state, byte)


def is_value_complete(state: ValueState) -> bool:
    """Return whether a primitive value can end at the current state."""
    if isinstance(state, StringState):
        return (
            state.complete
            and not state.escaped
            and state.unicode_digits_remaining == 0
            and not state.pending_utf8
        )
    if isinstance(state, NumberState):
        return state.phase in ("zero", "integer", "fraction", "exponent")
    return state.generated in (b"true", b"false")
