*This project has been created as part of the 42 curriculum by hirokigoda.*

# call me maybe

## Description

This project translates natural-language requests into structured function
calls with `Qwen/Qwen3-0.6B`. It uses token-by-token constrained decoding so
that the generated function name belongs to the supplied definitions and every
argument follows the selected function's JSON schema.

The program does not execute the selected function. It produces a JSON array
containing the original prompt, the selected function name, and its extracted
parameters.

## Instructions

Install the project dependencies:

```bash
make install
```

Run with the default files in `data/input/`:

```bash
make run
```

Run with explicit paths:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

The first model run may require the Qwen model files to be available through
the Hugging Face cache. The default output is written to
`data/output/function_calling_results.json`.

Other Makefile targets are:

- `make debug`: start the program through Python's debugger.
- `make clean`: remove Python and tool caches.
- `make lint`: run the mandatory flake8 and mypy checks.
- `make lint-strict`: run flake8 and strict mypy checks.

## Input format

`function_calling_tests.json` is an array of prompt objects:

```json
[
  {"prompt": "What is the sum of 2 and 3?"}
]
```

`functions_definition.json` is an array of available functions:

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": {"type": "number"},
      "b": {"type": "number"}
    },
    "returns": {"type": "number"}
  }
]
```

The mandatory implementation accepts `string`, `number`, and `boolean`
parameter types.

## Output format

Each output object contains exactly three keys:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {
    "a": 2,
    "b": 3
  }
}
```

## Algorithm

The decoder generates each function-call object in one pass. It first emits
the fixed JSON prefix and then reaches the `name` value. At that point, only
tokens that remain prefixes of names from `functions_definition.json` are
allowed. The language model's logits therefore choose the function without an
application-side heuristic.

As soon as a complete name is generated, the constraint switches to that
function's parameter schema. Parameter keys are emitted in definition order,
and their values are restricted by byte-level state machines for JSON strings,
numbers, and booleans.

For every generated token, the program:

1. Gets the next-token logits from the public SDK.
2. Checks candidate token bytes against the current JSON/schema state.
3. Sets every invalid token's score to negative infinity.
4. Greedily selects the highest-scoring valid token.
5. Advances only the state associated with the selected token.

Candidate tokens are consumed byte by byte because a single vocabulary token
can cross JSON grammar boundaries. Incomplete multibyte UTF-8 characters are
kept in the state until a later token completes them.

## Design decisions

- Function selection and argument extraction share one generation pass.
- The selected function name is never generated twice.
- All generated JSON syntax passes through logit masking; fixed fragments are
  not inserted between model-generated fragments.
- Greedy selection provides deterministic results.
- The original prompt and outer result array are application metadata.
- Pydantic validates inputs and final outputs, but validation never repairs or
  substitutes for constrained generation.
- The output file is replaced atomically only after every prompt succeeds.
- Only public methods of the supplied `llm_sdk` package are used.

Further implementation details are documented in
`docs/design_decisions.md`.

## Performance analysis

The model and vocabulary are initialized once and reused for all prompts.
Vocabulary tokens are indexed by their first byte, allowing fixed JSON phases
and function-name prefixes to avoid scanning unrelated candidates. Constraint
transitions are cached across prompts.

The implementation removes the previous separate function-selection pass, so
the model no longer generates the selected name twice. Runtime and accuracy
must be evaluated with `Qwen/Qwen3-0.6B` and the provided input files on the
machine used for review.

## Challenges faced

The main challenge is that tokenizer boundaries do not match JSON boundaries.
A token can contain several JSON characters or only part of one UTF-8
character. The constraint therefore operates on recovered token bytes and can
advance through multiple grammar phases while evaluating one candidate.

Another challenge is choosing a parameter schema without heuristics. The
single-pass state records the function selected by masked model logits and
uses that name to activate the corresponding schema immediately.

## Testing strategy

No separate test dataset is required. The implementation is exercised through
the main program with the existing files:

- `data/input/function_calling_tests.json`
- `data/input/functions_definition.json`

The generated file is checked for one result per prompt, exact output keys,
declared function names, matching parameter names and types, valid JSON, and
completion within five minutes. The decoder remains generic so reviewers can
replace both input files without changing the source code.

## Resources

- [JSON specification](https://www.json.org/json-en.html)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [Python `json` documentation](https://docs.python.org/3/library/json.html)
- [Qwen3 model collection](https://huggingface.co/collections/Qwen/qwen3)

AI was used to compare the implementation with the subject requirements,
explore architecture alternatives, review edge cases around token boundaries,
and help draft code and documentation. The constrained-decoding design,
implementation choices, and generated output were reviewed critically rather
than accepted solely from AI suggestions.
