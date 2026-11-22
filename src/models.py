"""検証済みの関数呼び出し入出力モデル。"""

import math
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    StringConstraints,
    TypeAdapter,
    field_validator,
)

# カスタムの文字列型を定義
NonBlankString = Annotated[
    str,
    StringConstraints(min_length=1),
]
JsonType = Literal["string", "number", "boolean"]


class PromptInput(BaseModel):
    """一つの自然言語リクエストを表す。"""

    # Pydantic検証の設定（余分なキーと自動型変換を禁止）
    model_config = ConfigDict(extra="forbid", strict=True)

    prompt: NonBlankString

    @field_validator("prompt")
    @classmethod
    def reject_blank_prompt(cls, value: str) -> str:
        """有効な入力を変更せず、空白だけのプロンプトを拒否する。"""
        if not value.strip():
            raise ValueError("prompt must not be blank")
        return value


class TypeDefinition(BaseModel):
    """サポート対象の一つのJSONプリミティブ型を表す。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    type: JsonType


class FunctionDefinition(BaseModel):
    """利用可能な一つの関数とそのスキーマを表す。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    name: NonBlankString
    description: NonBlankString
    parameters: dict[NonBlankString, TypeDefinition]
    returns: TypeDefinition

    @field_validator("name", "description")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        """空白だけの名前と説明を拒否する。"""
        if not value.strip():
            raise ValueError("function text fields must not be blank")
        return value

    @field_validator("parameters")
    @classmethod
    def reject_blank_parameter_names(
        cls,
        parameters: dict[NonBlankString, TypeDefinition],
    ) -> dict[NonBlankString, TypeDefinition]:
        """空白だけの引数名を拒否する。"""
        if any(not name.strip() for name in parameters):
            raise ValueError("parameter names must not be blank")
        return parameters


class FunctionCall(BaseModel):
    """プロンプトを付加する前の生成された関数呼び出しを表す。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    name: str
    parameters: dict[str, Any]


class FunctionCallingResult(BaseModel):
    """最終出力の一つの要素を表す。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    prompt: str
    name: str
    parameters: dict[str, Any]


PROMPT_LIST_ADAPTER = TypeAdapter(list[PromptInput])
FUNCTION_LIST_ADAPTER = TypeAdapter(list[FunctionDefinition])


def validate_prompts(data: object) -> list[PromptInput]:
    """すべてのプロンプト要素を検証して返す。"""
    return PROMPT_LIST_ADAPTER.validate_python(data, strict=True)


def validate_functions(data: object) -> list[FunctionDefinition]:
    """関数を検証し、空または曖昧な定義セットを拒否する。"""
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
    """生成された呼び出しを、選択された関数のスキーマに照らして検証する。"""
    #python辞書をFunctionCallで検証し、クラスのオブジェクトが作成される
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
        if (
            definition.type == "number"
            and isinstance(value, float)
            and not math.isfinite(value)
        ):
            raise ValueError(
                f"parameter {name!r} must be a finite JSON number"
            )

    return call
