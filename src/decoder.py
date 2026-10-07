"""Generic logit-masked constrained decoding loop."""

from collections.abc import Mapping
from typing import Protocol, TypeVar

import numpy as np

StateT = TypeVar("StateT")


class EncodedTokenIds(Protocol):
    """Describe the tensor operation used from the public SDK result."""

    def tolist(self) -> list[list[int]]:
        """Return batch-shaped token IDs."""
        ...


class DecodingModel(Protocol):
    """Describe the public model operations used by the decoder."""

    def encode(self, text: str) -> EncodedTokenIds:
        """Encode text into a single token-ID batch."""
        ...

    def get_logits_from_input_ids(
        self,
        input_ids: list[int],
    ) -> list[float]:
        """Return next-token logits."""
        ...


class Constraint(Protocol[StateT]):
    """Describe the state-machine operations required by the decoder."""

    def initial_state(self) -> StateT:
        """Return the initial grammar state."""
        ...

    def transition(
        self,
        state: StateT,
        token_bytes: bytes,
    ) -> StateT | None:
        """Return the candidate state, or None when disallowed."""
        ...

    def is_complete(self, state: StateT) -> bool:
        """Return whether generation is complete."""
        ...

    def allowed_first_bytes(
        self,
        state: StateT,
    ) -> frozenset[int] | None:
        """Return possible token-leading bytes when known."""
        ...


def _candidate_token_ids(
    allowed_first_bytes: frozenset[int] | None,
    token_bytes_by_id: Mapping[int, bytes],
    token_ids_by_first_byte: Mapping[int, tuple[int, ...]],
) -> list[int]:
    """Return token IDs worth testing in the current grammar state."""
    if allowed_first_bytes is None:
        return [
            token_id
            for token_id, token_bytes in token_bytes_by_id.items()
            if token_bytes
        ]

    candidates: list[int] = []
    for byte in allowed_first_bytes:
        candidates.extend(token_ids_by_first_byte.get(byte, ()))
    return candidates


def constrained_decode(
    model: DecodingModel,
    prompt: str,
    constraint: Constraint[StateT],
    token_bytes_by_id: Mapping[int, bytes],
    token_ids_by_first_byte: Mapping[int, tuple[int, ...]],
    max_tokens: int = 256,
) -> str:
    """Generate one continuation with invalid token logits masked to -inf."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    encoded = model.encode(prompt).tolist()
    if len(encoded) != 1 or not encoded[0]:
        raise RuntimeError(
            "model prompt did not encode to one non-empty batch"
        )

    prompt_token_ids = encoded[0]
    output_token_ids: list[int] = []
    output_bytes = bytearray()
    state = constraint.initial_state()

    for _ in range(max_tokens):
        if constraint.is_complete(state):
            break

        logits = model.get_logits_from_input_ids(
            prompt_token_ids + output_token_ids
        )
        if not logits:
            raise RuntimeError("model returned an empty logit vector")

        masked_logits = np.full(
            len(logits),
            -np.inf,
            dtype=np.float64,
        )
        candidate_states: dict[int, StateT] = {}
        candidate_ids = _candidate_token_ids(
            constraint.allowed_first_bytes(state),
            token_bytes_by_id,
            token_ids_by_first_byte,
        )

        for token_id in candidate_ids:
            if token_id < 0 or token_id >= len(logits):
                continue
            next_state = constraint.transition(
                state,
                token_bytes_by_id[token_id],
            )
            if next_state is not None:
                candidate_states[token_id] = next_state
                masked_logits[token_id] = logits[token_id]

        if not candidate_states:
            raise RuntimeError(
                "no vocabulary token is valid in the current constraint state"
            )

        selected_token_id = int(np.argmax(masked_logits))
        if selected_token_id not in candidate_states:
            raise RuntimeError(
                "masked token selection produced no valid token"
            )

        output_token_ids.append(selected_token_id)
        output_bytes.extend(token_bytes_by_id[selected_token_id])
        state = candidate_states[selected_token_id]

    if not constraint.is_complete(state):
        raise RuntimeError("constrained generation exceeded the token limit")

    try:
        return output_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError(
            "completed constrained output is not valid UTF-8"
        ) from error
