from transformers import AutoModelForCausalLM, AutoTokenizer

import config
import quantization_utils

def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        quantization_config = quantization_utils.load_bits_and_bytes_config(),
        trust_remote_code = config.TRUST_REMOTE_CODE
    )
    model.config.use_cache = config.ENABLE_MODEL_CONFIG_CACHE
    return model

def load_tokenizers():
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_NAME, 
        trust_remote_code=config.TRUST_REMOTE_CODE)
    return tokenizer