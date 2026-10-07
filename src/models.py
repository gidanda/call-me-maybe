"""Validated input and output models for function calling."""

from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    StringConstraints,
    TypeAdapter,
    field_validator,
    model_validator,
)

NonBlankString = Annotated[
    str,
    StringConstraints(min_length=1),
]
JsonType = Literal["string", "number", "boolean"]


class PromptInput(BaseModel):
    """Represent one natural-language request."""

    model_config = ConfigDict(extra="forbid", strict=True)

    prompt: NonBlankString

    @field_validator("prompt")
    @classmethod
    def reject_blank_prompt(cls, value: str) -> str:
        """Reject whitespace-only prompts without changing valid input."""
        if not value.strip():
            raise ValueError("prompt must not be blank")
        return value


class TypeDefinition(BaseModel):
    """Represent one supported primitive JSON type."""

    model_config = ConfigDict(extra="forbid", strict=True)

    type: JsonType


class FunctionDefinition(BaseModel):
    """Represent one available function and its schema."""

    model_config = ConfigDict(extra="forbid", strict=True)

    name: NonBlankString
    description: NonBlankString
    parameters: dict[NonBlankString, TypeDefinition]
    returns: TypeDefinition

    @field_validator("name", "description")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        """Reject whitespace-only names and descriptions."""
        if not value.strip():
            raise ValueError("function text fields must not be blank")
        return value

    @model_validator(mode="after")
    def reject_blank_parameter_names(self) -> "FunctionDefinition":
        """Reject whitespace-only parameter names."""
        if any(not name.strip() for name in self.parameters):
            raise ValueError("parameter names must not be blank")
        return self


class FunctionCall(BaseModel):
    """Represent a generated function call before adding its prompt."""

    model_config = ConfigDict(extra="forbid", strict=True)

    name: str
    parameters: dict[str, Any]


class FunctionCallingResult(BaseModel):
    """Represent one final output entry."""

    model_config = ConfigDict(extra="forbid", strict=True)

    prompt: str
    name: str
    parameters: dict[str, Any]


PROMPT_LIST_ADAPTER = TypeAdapter(list[PromptInput])
FUNCTION_LIST_ADAPTER = TypeAdapter(list[FunctionDefinition])


def validate_prompts(data: object) -> list[PromptInput]:
    """Validate and return all prompt entries."""
    return PROMPT_LIST_ADAPTER.validate_python(data, strict=True)


def validate_functions(data: object) -> list[FunctionDefinition]:
    """Validate functions and reject an empty or ambiguous definition set."""
    functions = FUNCTION_LIST_ADAPTER.validate_python(data, strict=True)
    if not functions:
        raise ValueError("at least one function definition is required")

    seen_names: set[str] = set()
    for function in functions:
        if function.name in seen_names:
            raise ValueError(f"duplicate function name: {function.name}")
        seen_names.add(function.name)

    return functions


def validate_function_call(
    data: object,
    functions_by_name: dict[str, FunctionDefinition],
) -> FunctionCall:
    """Validate a generated call against its selected function schema."""
    call = FunctionCall.model_validate(data, strict=True)
    function = functions_by_name.get(call.name)
    if function is None:
        raise ValueError(f"unknown generated function name: {call.name}")

    if set(call.parameters) != set(function.parameters):
        raise ValueError(
            "generated parameter names do not match the selected schema"
        )

    for name, definition in function.parameters.items():
        value = call.parameters[name]
        if definition.type == "string" and type(value) is not str:
            raise ValueError(f"parameter {name!r} must be a string")
        if definition.type == "boolean" and type(value) is not bool:
            raise ValueError(f"parameter {name!r} must be a boolean")
        if definition.type == "number" and type(value) not in (int, float):
            raise ValueError(f"parameter {name!r} must be a number")

    return call
