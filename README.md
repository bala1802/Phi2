# Under Construction...

# Finetuning Microsoft/Phi-2

## Introduction

This repository is dedicated to the fine-tuning of a Microsoft's Phi-2 small language model, aiming to enhance its capabilities and adapt it to specific tasks or domains.

## About Phi-2

Phi-2 is a Transformer with 2.7 billion parameters. It was trained using the same data sources as Phi-1.5, augmented with a new data source that consists of various NLP synthetic texts and filtered websites (for safety and educational value). When assessed against benchmarks testing common sense, language understanding, and logical reasoning, Phi-2 showcased a nearly state-of-the-art performance among models with less than 13 billion parameters.

The hasn't been fine-tuned through reinforcement learning from human feedback. The intention behind crafting this open-source model is to provide the research community with a non-restricted small model to explore vital safety challenges, such as reducing toxicity, understanding societal biases, enhancing controllability, and more. 
Source: [Microsoft/Phi-2](https://huggingface.co/microsoft/phi-2#model-summary)

## How to read this repository?

```
├── LICENSE
├── README.md
├── adapter_utils.py
├── app.py
├── config.py
├── data_utils.py
├── inference.py
├── model
│   └── checkpoint-700
│       ├── README.md
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       ├── added_tokens.json
│       ├── merges.txt
│       ├── optimizer.pt
│       ├── rng_state.pth
│       ├── scheduler.pt
│       ├── special_tokens_map.json
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       ├── trainer_state.json
│       ├── training_args.bin
│       └── vocab.json
├── model_utils.py
├── quantization_utils.py
├── requirements.txt
```

## Dataset

The dataset known as [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1#openassistant-conversations-dataset-oasst1) serves as the fine-tuning source for the model. It includes a collection of human-generated, human-annotated assistant-style conversations, totaling 161,443 messages across 35 diverse languages. This corpus is enriched with 461,292 quality ratings, leading to the creation of over 10,000 fully annotated conversation trees.

`pip install datasets`

<img width="410" alt="image" src="https://github.com/bala1802/Phi2/assets/22103095/aea7b31d-b5ce-42b1-82dd-af0b0c7c34a5">

Refer [data_utils.py](https://github.com/bala1802/Phi2/blob/main/data_utils.py) for converting the training dataset into the specific format for fine-tuning.

Instruction Template: 
`### Human: <YOUR QUERY> ### Assistant: <YOUR ANSWER>`

Example: 
`### Human: What is the impact of cryptocurrency in the world? ### Assistant: Cryptocurrency has had a profound impact by revolutionizing traditional financial systems and fostering decentralization in global transactions.`


## Steps involved in Finetuning the model





