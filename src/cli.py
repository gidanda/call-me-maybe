import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="質問一覧のJSONファイルのパス",
    )

    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="関数定義一覧のJSONファイルのパス",
    )

    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        help="出力されたJSONファイルの保存先のパス",
    )

    args = parser.parse_args()

    return args