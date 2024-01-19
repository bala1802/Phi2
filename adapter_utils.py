from peft import LoraConfig

import config

def load_adapter(target_modules):
    lora_config = LoraConfig(
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        r = config.LORA_RANK,
        bias=None,
        task_type=config.TASK_TYPE,
        target_modules=target_modules
    )
    return lora_config