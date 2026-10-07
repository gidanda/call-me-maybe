"""Load public token vocabulary data for byte-level constraints."""

import json
from pathlib import Path


def load_token_bytes(vocab_path: str | Path) -> dict[int, bytes]:
    """Map token IDs to the original bytes represented by the vocabulary."""
    with Path(vocab_path).open("r", encoding="utf-8") as file:
        vocabulary = json.load(file)

    if not isinstance(vocabulary, dict):
        raise ValueError("vocabulary must be a JSON object")

    visible_bytes = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    byte_values = visible_bytes.copy()
    unicode_values = visible_bytes.copy()
    extra_code_point = 0

    for byte_value in range(256):
        if byte_value not in visible_bytes:
            byte_values.append(byte_value)
            unicode_values.append(256 + extra_code_point)
            extra_code_point += 1

    byte_by_character = {
        chr(code_point): byte_value
        for byte_value, code_point in zip(byte_values, unicode_values)
    }
    token_bytes_by_id: dict[int, bytes] = {}

    for token, token_id in vocabulary.items():
        if not isinstance(token, str):
            raise ValueError("vocabulary tokens must be strings")
        if (
            not isinstance(token_id, int)
            or isinstance(token_id, bool)
            or token_id < 0
        ):
            raise ValueError(
                "vocabulary token IDs must be non-negative integers"
            )
        if token_id in token_bytes_by_id:
            raise ValueError(f"duplicate vocabulary token ID: {token_id}")
        try:
            token_bytes = bytes(
                byte_by_character[character] for character in token
            )
        except KeyError as error:
            raise ValueError(
                "vocabulary contains an unknown byte-level character: "
                f"{error.args[0]!r}"
            ) from error
        token_bytes_by_id[token_id] = token_bytes

    return token_bytes_by_id


def index_tokens_by_first_byte(
    token_bytes_by_id: dict[int, bytes],
) -> dict[int, tuple[int, ...]]:
    """Index non-empty token IDs by their first byte."""
    mutable_index: dict[int, list[int]] = {}
    for token_id, token_bytes in token_bytes_by_id.items():
        if token_bytes:
            mutable_index.setdefault(token_bytes[0], []).append(token_id)
    return {
        byte: tuple(token_ids)
        for byte, token_ids in mutable_index.items()
    }
