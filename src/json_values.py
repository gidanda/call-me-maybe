"""JSONプリミティブ値に対する増分的なバイト単位の制約。"""

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


StringState = tuple[Literal["string"], bool, bool, int, bytes, bool]
NumberState = tuple[Literal["number"], NumberPhase]
BooleanState = tuple[Literal["boolean"], bytes]
ValueState = StringState | NumberState | BooleanState


def initial_value_state(type_name: str) -> ValueState:
    """サポート対象のJSONプリミティブ型の初期状態を返す。"""
    if type_name == "string":
        return ("string", False, False, 0, b"", False)
    if type_name == "number":
        return ("number", "start")
    if type_name == "boolean":
        return ("boolean", b"")
    raise ValueError(f"unsupported parameter type: {type_name}")


def _is_hex_digit(byte: int) -> bool:
    """バイトがASCIIの16進数字であるかを返す。"""
    return (
        ord("0") <= byte <= ord("9")
        or ord("a") <= byte <= ord("f")
        or ord("A") <= byte <= ord("F")
    )


def _append_utf8_byte(pending: bytes, byte: int) -> bytes | None:
    """未完成のUTF-8接尾部を返すか、無効なバイト列を拒否する。"""
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
    """JSON文字列の1バイトを消費する。"""
    _, started, escaped, unicode_remaining, pending_utf8, complete = state
    if complete:
        return None
    if not started:
        if byte != ord('"'):
            return None
        return ("string", True, escaped, unicode_remaining, pending_utf8,
                complete)
    if pending_utf8:
        pending = _append_utf8_byte(pending_utf8, byte)
        if pending is None:
            return None
        return ("string", started, escaped, unicode_remaining, pending,
                complete)
    if unicode_remaining:
        if not _is_hex_digit(byte):
            return None
        return ("string", started, escaped, unicode_remaining - 1,
                pending_utf8, complete)
    if escaped:
        if byte in b'"\\/bfnrt':
            return ("string", started, False, unicode_remaining,
                    pending_utf8, complete)
        if byte == ord("u"):
            return ("string", started, False, 4, pending_utf8, complete)
        return None
    if byte == ord('"'):
        return ("string", started, escaped, unicode_remaining,
                pending_utf8, True)
    if byte == ord("\\"):
        return ("string", started, True, unicode_remaining, pending_utf8,
                complete)
    if byte < 0x20:
        return None
    if byte < 0x80:
        return state

    pending = _append_utf8_byte(b"", byte)
    if pending is None:
        return None
    return ("string", started, escaped, unicode_remaining, pending,
            complete)


def _transition_number(state: NumberState, byte: int) -> NumberState | None:
    """JSON数値の1バイトを消費する。"""
    character = chr(byte)
    phase = state[1]
    if phase == "start":
        if character == "-":
            return ("number", "sign")
        if character == "0":
            return ("number", "zero")
        if "1" <= character <= "9":
            return ("number", "integer")
        return None
    if phase == "sign":
        if character == "0":
            return ("number", "zero")
        if "1" <= character <= "9":
            return ("number", "integer")
        return None
    if phase in ("zero", "integer"):
        if phase == "integer" and "0" <= character <= "9":
            return state
        if character == ".":
            return ("number", "dot")
        return None
    if phase == "dot":
        if "0" <= character <= "9":
            return ("number", "fraction")
        return None
    if phase == "fraction":
        if "0" <= character <= "9":
            return state
        if character in "eE":
            return ("number", "exponent_mark")
        return None
    if phase == "exponent_mark":
        if character in "+-":
            return ("number", "exponent_sign")
        if "0" <= character <= "9":
            return ("number", "exponent")
        return None
    if phase == "exponent_sign":
        if "0" <= character <= "9":
            return ("number", "exponent")
        return None
    if phase == "exponent" and "0" <= character <= "9":
        return state
    return None


def _transition_boolean(
    state: BooleanState,
    byte: int,
) -> BooleanState | None:
    """JSON真偽値の1バイトを消費する。"""
    candidate = state[1] + bytes((byte,))
    if not any(value.startswith(candidate) for value in (b"true", b"false")):
        return None
    return ("boolean", candidate)


def transition_value_byte(state: ValueState, byte: int) -> ValueState | None:
    """状態固有のプリミティブ文法を使用して1バイトを消費する。"""
    if state[0] == "string":
        return _transition_string(state, byte)
    if state[0] == "number":
        return _transition_number(state, byte)
    return _transition_boolean(state, byte)


def is_value_complete(state: ValueState) -> bool:
    """現在の状態でプリミティブ値を終了できるかを返す。"""
    if state[0] == "string":
        _, _, escaped, unicode_remaining, pending_utf8, complete = state
        return (
            complete
            and not escaped
            and not unicode_remaining
            and not pending_utf8
        )
    if state[0] == "number":
        return state[1] in ("fraction", "exponent")
    return state[1] in (b"true", b"false")
