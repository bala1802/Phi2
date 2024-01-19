from transformers import pipeline

import config

def predict(prompt, model, tokenizer, max_length):
    pipe = pipeline(task = config.TASK,
                    model = model,
                    tokenizer = tokenizer,
                    max_length = max_length)
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    return result[0]['generated_text']
    