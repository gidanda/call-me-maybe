# Design Decisions

## 1. Single-pass schema-branching constrained decoding

The program generates one function-call JSON object in a single constrained
decoding pass.

The decoder does not select a function in a separate generation step. While
generating the value of the `name` field, it restricts the model to function
names declared in `functions_definition.json`. Once one complete function name
has been generated, the decoder records that choice and switches the remaining
constraint to the selected function's parameter schema.

For example, if the model selects `fn_add_numbers`, the rest of the generation
is constrained to the following shape:

```json
{
  "name": "fn_add_numbers",
  "parameters": {
    "a": 2,
    "b": 3
  }
}
```

This design has three important properties:

1. The LLM chooses the function from its logits; no heuristic chooses it.
2. The selected function name is generated only once.
3. The parameter constraint is derived from the function actually selected by
   the LLM during that same generation.

The original prompt and the outer result array are application data. After a
function-call object has been generated and validated, the application adds the
unchanged original prompt and writes all results as one JSON array with exactly
the required `prompt`, `name`, and `parameters` keys.

## 2. Logit masking is mandatory at every generation step

Every output token is selected through the same constrained-decoding loop:

1. Request next-token logits from `get_logits_from_input_ids`.
2. Test vocabulary tokens against the current constraint state.
3. Set every disallowed token's logit to negative infinity.
4. Select the highest-scoring remaining token.
5. Append the selected token ID and advance the constraint state.

Fixed JSON syntax is not inserted between generated fragments. Tokens that
produce braces, quotes, keys, colons, commas, function names, argument names,
argument values, and closing delimiters must all pass through logit masking.

Python-side validation after generation is a defensive check. It must not be
used to repair malformed output or replace generation-time constraints.

## 3. Constraint state and transitions

The constraint represents the expected function-call object as the following
ordered phases:

```text
object prefix
  -> function-name choice
  -> parameters prefix
  -> selected function's parameter entries
  -> object suffix
  -> complete
```

The initial fixed prefix is:

```json
{"name":
```

The function-name phase accepts only prefixes of JSON-encoded names from the
input definitions. A trie or equivalent prefix lookup is used so that names
such as `fn_get` and `fn_get_user` remain distinguishable until the closing
quote identifies one complete name.

After the name is complete, the state stores its corresponding function
definition. The next fixed segment is:

```json
,"parameters":{
```

The parameter phases are then constructed from the selected definition, in the
input parameter order. Each parameter entry consists of:

1. Its exact JSON-encoded key.
2. A colon.
3. A value state machine matching the declared type.
4. A comma when another parameter follows.

The mandatory implementation supports the primitive types stated by the
project inputs:

- `string`: a syntactically valid JSON string, including escapes and UTF-8.
- `number`: the complete JSON number grammar, including sign, fraction, and
  exponent.
- `boolean`: exactly `true` or `false`.

Unknown types are rejected during input validation, before model generation.

After all required parameters have been generated exactly once, only the final
closing braces are permitted. The state is complete only after the entire
function-call object has been consumed and no partial UTF-8 sequence remains.

## 4. Tokens may cross grammar boundaries

A vocabulary token may contain several characters and may cross structural
boundaries. For example, one token could contain the end of a string, a comma,
and the beginning of the next key.

Therefore, candidate tokens are evaluated byte by byte. A transition must
consume the candidate token in full, advancing through as many grammar phases
as necessary. A token is allowed only when every byte can be consumed without
violating the current JSON structure or selected schema.

The state transition used to evaluate a candidate is hypothetical. Only the
state belonging to the token selected from the masked logits becomes the next
real state.

## 5. Token bytes and incremental UTF-8 validation

Constraint decisions use token byte sequences recovered from the public
vocabulary file returned by `get_path_to_vocab_file`. They do not depend on
private tokenizer attributes or SDK methods.

A token can end in the middle of a multibyte UTF-8 character. String parsing
therefore retains an incomplete UTF-8 suffix in its state. The next token is
checked together with that suffix. A candidate is rejected if the combined
bytes cannot become valid UTF-8.

The constraint cannot enter its complete state while unfinished UTF-8 bytes,
an unfinished escape, or an incomplete Unicode escape remain.

## 6. Function selection is made by the LLM

The function-name constraint limits the available choices but does not rank
them. Ranking comes exclusively from the model's logits.

The decoder must not use descriptions, keywords, regular expressions, function
order, or other application-side heuristics to choose a function. Application
code supplies the model with the prompt and available definitions, then the
masked model scores determine which allowed name is generated.

## 7. Input and output validation

Pydantic models validate both input files before the model is initialized.
Validation covers at least:

- The prompt file is a list of objects containing only a non-empty `prompt`.
- The function file is a non-empty list of function definitions.
- Function names are non-empty and unique.
- Descriptions are non-empty.
- Parameter names are non-empty.
- Parameter and return types are supported.
- Unexpected fields are rejected.

After decoding, the generated text is parsed as JSON and validated against the
selected function definition. This check must confirm:

- The object contains only `name` and `parameters`.
- The name is one of the input definitions.
- Parameter keys match the selected definition exactly.
- Every required parameter is present.
- Every value has the declared JSON type.

The application then adds the unchanged source prompt. The final output object
is validated to contain exactly `prompt`, `name`, and `parameters` before it is
written.

## 8. Error and file-handling policy

Expected failures produce a concise message on standard error and a non-zero
exit status. This includes missing files, invalid JSON, invalid input schemas,
unsupported types, model failures, an empty allowed-token set, token-limit
exhaustion, invalid UTF-8, and final validation failures.

No partially generated result file is published. All prompts are processed and
validated first. The complete result array is then written to a temporary file
in the destination directory and atomically moved to the requested output path.

## 9. Performance policy

The implementation must be measured with `Qwen/Qwen3-0.6B` and the supplied
test prompts. Correctness takes priority, but the design avoids the previous
duplicate function-name generation and supports the five-minute requirement.

The constraint engine may use the following optimizations without changing the
selection semantics:

- Index vocabulary tokens by their first byte.
- Cache hypothetical `(state, token_id)` transitions.
- Reuse the loaded vocabulary and fixed grammar data across prompts.
- Build the function-name prefix lookup once per function-definition file.

An optimization must never allow a token that has not been validated against
the full current state.

## 10. Module responsibilities

The implementation is divided by responsibility:

```text
src/__main__.py        CLI boundary and user-facing error handling
src/cli.py             command-line argument definitions
src/models.py          Pydantic input and output models
src/file_io.py         JSON loading and atomic output writing
src/prompts.py         model prompt construction
src/vocabulary.py      public vocabulary-file loading and token bytes
src/json_values.py     string, number, and boolean value transitions
src/call_constraint.py single-pass function-call grammar and schema branching
src/decoder.py         logits, masking, token selection, and generation loop
src/pipeline.py        model initialization and per-prompt orchestration
```

The decoder knows how to apply a constraint but does not know about function
definitions. The call constraint knows the JSON/schema rules but does not call
the model. The pipeline coordinates the two without implementing its own token
selection rules.

## 11. Verification strategy

No additional test suite or test dataset is created for this implementation.
Verification uses the existing project inputs:

```text
data/input/function_calling_tests.json
data/input/functions_definition.json
```

The main program is run with these files, using both the default paths and the
documented explicit CLI arguments. The resulting output is checked for:

- One result for every existing input prompt.
- Exactly the `prompt`, `name`, and `parameters` keys in every result.
- A function name declared in the existing function-definition file.
- Parameter names and value types matching the selected definition.
- Valid JSON with no additional prose.
- Completion within the required five-minute limit.

Verification does not fetch external examples, introduce unrelated fixtures,
or add newly invented function definitions and prompts. The production code's
input validation and constrained-decoding rules remain generic so that the
reviewer can replace the two input files without requiring code changes.
