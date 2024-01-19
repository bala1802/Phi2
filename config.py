'''
Data Configuration
'''
DATASET = "OpenAssistant/oasst1"

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