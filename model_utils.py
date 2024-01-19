from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

import config

def load_model(quantization_config):
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        quantization_config = quantization_config,
        trust_remote_code = config.TRUST_REMOTE_CODE
    )
    model.config.use_cache = config.ENABLE_MODEL_CONFIG_CACHE
    return model

def load_tokenizers():
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_NAME, 
        trust_remote_code=config.TRUST_REMOTE_CODE)
    return tokenizer

def load_training_arguments():
    training_arguments = TrainingArguments(
        output_dir=config.MODEL_OUTPUT_DIR,
        per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        optim=config.OPTIM,
        save_steps=config.SAVE_STEPS,
        logging_steps=config.LOGGING_STEPS,
        learning_rate=config.LEARNING_RATE,
        fp16=config.ENABLE_FP_16,
        max_grad_norm=config.MAX_GRAD_NORM,
        max_steps=config.MAX_STEPS,
        warmup_ratio=config.WARMUP_RATIO,
        gradient_checkpointing=config.ENABLE_GRADIENT_CHECKPOINTING
    )
    return training_arguments

def load_trainer(model, training_dataset, peft_config, tokenizer, training_arguments):
    trainer = SFTTrainer(
        model = model,
        train_dataset = training_dataset,
        peft_config = peft_config,
        dataset_text_field = config.DATASET_TEXT_FIELD,
        max_seq_length = config.MAX_SEQ_LENGTH,
        tokenizer = tokenizer,
        args = training_arguments
    )
    return trainer