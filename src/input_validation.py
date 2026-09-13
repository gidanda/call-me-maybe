from pydantic import BaseModel, ConfigDict, StringConstraints, TypeAdapter
from typing import Annotated, Literal

NonBlankString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]

class PromptInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: NonBlankString


PROMPT_LIST_ADAPTER = TypeAdapter(list[PromptInput])


class TypeDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["string", "number", "boolean"]


class FunctionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: NonBlankString
    description: NonBlankString
    parameters: dict[NonBlankString, TypeDefinition]
    returns: TypeDefinition


FUNCTION_LIST_ADAPTER = TypeAdapter(list[FunctionInput])


def validate_prompts(data):
    return PROMPT_LIST_ADAPTER.validate_python(data)


def validate_functions(data):
    functions = FUNCTION_LIST_ADAPTER.validate_python(data)
    seen_names: set[str] = set()

    for function in functions:
        if function.name in seen_names:
            raise ValueError(
                f"duplicate function name: {function.name}"
            )
        seen_names.add(function.name)

    return functions