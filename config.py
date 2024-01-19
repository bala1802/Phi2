'''
Data Configuration
'''
DATASET = "OpenAssistant/oasst1"
DATASET_TEXT_FIELD = "prompt_response"

'''
Model Configuration
'''
MODEL_NAME = "microsoft/phi-2"
TRUST_REMOTE_CODE = True
ENABLE_MODEL_CONFIG_CACHE = False

'''
Quantization Configuration
'''
ENABLE_4BIT = True
QUANTIZATION_TYPE = "nf4"

'''
Adapter Configuration
'''
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_RANK = 64
TASK_TYPE = "CAUSAL_LM"

'''
Model Training Configuration
'''
MODEL_OUTPUT_DIR = "TODO"
PER_DEVICE_TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
OPTIM = "paged_adamw_32bit"
SAVE_STEPS = 100
LOGGING_STEPS = 10
LEARNING_RATE = 2e-4
MAX_GRAD_NORM = 0.3
MAX_STEPS = 700
WARMUP_RATIO = 0.05
LR_SCHEDULER_TYPE = "constant"
ENABLE_FP_16 = True
ENABLE_GRADIENT_CHECKPOINTING=False

'''
Model Trainer Configuration
'''
MAX_SEQ_LENGTH = 512

'''
Inference Configuration
'''
TASK = "text_generation"