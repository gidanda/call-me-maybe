"""関数呼び出し生成用のプロンプト構築。"""

import json

from .models import FunctionDefinition


def build_generation_prompt(
    user_prompt: str,
    functions: list[FunctionDefinition],
) -> str:
    """関数選択と引数抽出のためのプロンプトを構築する。"""
    definitions = json.dumps(
        [function.model_dump(mode="json") for function in functions],
        ensure_ascii=False,
        #不要な空白なしにjsonの区切り文字を入れる
        separators=(",", ":"),
    )
    #Qwenが学習した特殊トークンを用いてテキストの役割を示唆
    #system:モデル全体の役割　user:具体的な指示と入力　assistant:ここからモデルが回答
    #モデルには意味判断のみを指示
    #全て、などの指示を正規表現で広く解釈しすぎないように指示
    #思考部分の|think|を先に閉じておくことで実際の回答から生成させる
    return (
        "<|im_start|>system\n"
        "You translate user requests into function calls using only the "
        "available function definitions.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "Select the function that best matches the user request.\n"
        "Use the function descriptions and parameter definitions provided "
        "below.\n"
        "Extract a value for every required parameter.\n\n"
        f"Available functions:\n{definitions}\n\n"
        f"User request:\n{user_prompt}\n\n"
        "Before generating the function call:\n"
        "- Determine the complete operation requested by the user.\n"
        "- Output each parameter as the raw value expected by the function, "
        "without language-specific wrappers or explanatory text.\n"
        "- Preserve the exact scope expressed by words such as all, every, "
        "only, before, and after.\n"
        "- Prefer the simplest parameter value that directly represents the "
        "requested target. Do not add broad prefixes, broad suffixes, or "
        "redundant surrounding context.\n"
        "- If the selected function operates on matching items, the value "
        "that identifies a match must select exactly one target item. One "
        "match must never include multiple targets or surrounding content.\n"
        "- For all or every, do not widen a single match. Rely on the "
        "function to repeat the operation for each matching item.\n"
        "- Ensure that calling the selected function once with these "
        "parameters performs the complete requested operation and does not "
        "affect unrelated content.\n\n"
        "Generate exactly one function call JSON object containing only name "
        "and parameters.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )
