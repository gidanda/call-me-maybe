def get_function_names(functions):
    function_names = [
        function.name
        for function in functions
    ]
    return function_names


def get_available_functions(generated, function_names):
    available_functions = [
        function_name
        for function_name in function_names
        if function_name.startswith(generated)
    ]
    return available_functions


def get_available_token_ids(output_token_ids, function_names_and_ids):
    output_length = len(output_token_ids)
    available_token_ids = set()

    for candidate_ids in function_names_and_ids.values():
        if (
            candidate_ids[:output_length] == output_token_ids
            and output_length < len(candidate_ids)
        ):
            available_token_ids.add(candidate_ids[output_length])
    
    return list(available_token_ids)

def encode_function_names(function_names, model):
    function_names_and_ids = {}

    for function_name in function_names:
        tokenized_function_name = model.encode(function_name).tolist()[0]
        function_names_and_ids[function_name] = tokenized_function_name

    return function_names_and_ids


def test_constrained_decoding(prompt, functions, model):
    prompt_token_ids = model.encode(prompt).tolist()[0]
    output_token_ids = []
    function_names = get_function_names(functions)
    function_names_and_ids = encode_function_names(function_names, model)

    for _ in range(100):
        generated = model.decode(output_token_ids)

        if generated in function_names:
            return generated

        logits = model.get_logits_from_input_ids(prompt_token_ids + output_token_ids)

        available_token_ids = get_available_token_ids(
            output_token_ids,
            function_names_and_ids,
        )

        if not available_token_ids:
            raise ValueError("no token IDs are allowed")

        available_token_set = set(available_token_ids)
        
        for token_id in range(len(logits)):
            if  token_id not in available_token_set:
                logits[token_id] = float("-inf")

        best_token_id = max(
            range(len(logits)),
            key=lambda token_id: logits[token_id],
        )
        output_token_ids.append(best_token_id)

    raise RuntimeError(
    "function name generation exceeded the token limit"
    )



    
