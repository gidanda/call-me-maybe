"""Command-line arguments for the function-calling program."""

import argparse


def parse_arguments() -> argparse.Namespace:
    """Parse file paths accepted by the command-line interface."""
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
        default="data/output/function_calling_results.json",
        help="destination path for the generated result array",
    )

    return parser.parse_args()
