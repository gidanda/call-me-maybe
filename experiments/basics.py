from llm_sdk import Small_LLM_Model

def main():
    model = Small_LLM_Model()
    ids = model.encode("I am").tolist()[0]
    
    for i in range(10):
        logits = model.get_logits_from_input_ids(ids)
        tmp_best = logits[0]
        for logit in logits:
            if logit > tmp_best:
                tmp_best = logit

        best_id = logits.index(tmp_best)
        ids.append(best_id)

    output = model.decode(ids)

    print(repr(output))

if __name__ == "__main__":
    main()

