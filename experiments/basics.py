from llm_sdk import Small_LLM_Model

def main():
    model = Small_LLM_Model()
    ids = model.encode("I like").tolist()[0]

    allowed_ids = {10, 20}

    logits = model.get_logits_from_input_ids(ids)
    for token_id in range(len(logits)):
            if token_id not in allowed_ids:
                logits[token_id] = float("-inf")
    
    for i in range(1):
        # logits = model.get_logits_from_input_ids(ids)
        tmp_best = logits[0]
        for logit in logits:
            if logit > tmp_best:
                tmp_best = logit

        best_id = logits.index(tmp_best)
        ids.append(best_id)

    output = model.decode(ids)
    print(repr(model.decode([10])))
    print(repr(model.decode([20])))

    print(repr(output))

if __name__ == "__main__":
    main()

