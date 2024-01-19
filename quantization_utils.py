import torch
from transformers import BitsAndBytesConfig

import config

def load_bits_and_bytes_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.ENABLE_4BIT,
        bnb_4bit_quant_type=config.QUANTIZATION_TYPE,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    return bnb_config