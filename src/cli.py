"""関数呼び出しプログラムのコマンドライン引数。"""

import argparse


def parse_arguments() -> argparse.Namespace:
    """コマンドラインインターフェースが受け取るファイルパスを解析する。"""
    parser = argparse.ArgumentParser(
        description="Generate schema-valid function calls with an LLM.",
    )

    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="path to the JSON array of natural-language prompts",
    )

    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="path to the JSON array of available function definitions",
    )

    parser.add_argument(
        "--output",
        default="data/output/function_calls.json",
        help="destination path for the generated result array",
    )

    return parser.parse_args()
