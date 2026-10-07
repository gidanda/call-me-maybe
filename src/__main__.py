"""Command-line entry point for constrained function-call generation."""

import sys

from .cli import parse_arguments
from .file_io import read_json, save_results
from .models import validate_functions, validate_prompts
from .pipeline import generate_all_results


def run() -> None:
    """Load inputs, generate all calls, and save a single result array."""
    args = parse_arguments()
    prompts = validate_prompts(read_json(args.input))
    functions = validate_functions(read_json(args.functions_definition))
    results = generate_all_results(prompts, functions)
    save_results(results, args.output)


def main() -> int:
    """Run the CLI and report failures without an application traceback."""
    try:
        run()
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
