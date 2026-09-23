from llm_sdk import Small_LLM_Model

from .cli import parse_arguments
from .file_io import read_json
from .input_validation import validate_prompts, validate_functions
from .prompt_building import build_prompt, build_function_selection_prompt
from .constrained_decoding import test_constrained_decoding

def main():
    args = parse_arguments()

    prompts_data = read_json(args.input)
    functions_data = read_json(args.functions_definition)

    prompts = validate_prompts(prompts_data)
    functions = validate_functions(functions_data)

    model = Small_LLM_Model()

    for prompt in prompts:
        modified_prompt = build_function_selection_prompt(prompt.prompt, functions)
        selected_function = test_constrained_decoding(modified_prompt, functions, model)
        print(prompt.prompt)
        print(selected_function)

if __name__ == "__main__":
    main()