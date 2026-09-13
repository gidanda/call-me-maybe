import json


def build_prompt(user_prompt, functions):
    functions_json = json.dumps(
        [function.model_dump() for function in functions],
        ensure_ascii=False,
        indent=2,
    )

    return (
        "Select the most appropriate function for the user's request "
        "and extract its required parameter values.\n\n"
        "Available functions:\n"
        f"{functions_json}\n\n"
        "User request:\n"
        f"{user_prompt}\n\n"
        "Return only a JSON object with exactly these keys:\n"
        '{"name": "selected_function_name", '
        '"parameters": {"required_parameter_name": "extracted_value"}}'
    )