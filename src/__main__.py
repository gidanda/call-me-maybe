from .cli import parse_arguments
from .file_io import read_json
from .input_validation import validate_prompts, validate_functions

def main():
    args = parse_arguments()

    prompts_data = read_json(args.input)
    functions_data = read_json(args.functions_definition)

    prompts = validate_prompts(prompts_data)
    functions = validate_functions(functions_data)


if __name__ == "__main__":
    main()