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


## Install PyPI Libraries

`pip install -r requirements.txt --quiet`

## Dataset

The dataset known as [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1#openassistant-conversations-dataset-oasst1) serves as the fine-tuning source for the model. It includes a collection of human-generated, human-annotated assistant-style conversations, totaling 161,443 messages across 35 diverse languages. This corpus is enriched with 461,292 quality ratings, leading to the creation of over 10,000 fully annotated conversation trees.

`pip install datasets`

<img width="410" alt="image" src="https://github.com/bala1802/Phi2/assets/22103095/aea7b31d-b5ce-42b1-82dd-af0b0c7c34a5">

Refer [data_utils.py](https://github.com/bala1802/Phi2/blob/main/data_utils.py) for converting the training dataset into the specific format for fine-tuning.

Instruction Template: 
`### Human: <YOUR QUERY> ### Assistant: <YOUR ANSWER>`

Example: 
`### Human: What is the impact of cryptocurrency in the world? ### Assistant: Cryptocurrency has had a profound impact by revolutionizing traditional financial systems and fostering decentralization in global transactions.`

## Architecture of the adapter module and its integration with the Transformer

<img width="551" alt="image" src="https://github.com/bala1802/Phi2/assets/22103095/3b21c0a7-23f5-4f9a-81b6-a60285b15449">

Source: [Parameter-Efficient Transfer Learning for NLP](http://proceedings.mlr.press/v97/houlsby19a.html) 

### Transformer

- The `Adapter` module is incorporated twice in the `Transformers`. Firstly, after the `projection` layer, which is followed by the Multi Headed attention.
- Secondly, after the 2 `FeedForward` Layers

### Adapter

- `Adapter` _contains few parameters relative to the attention and feedforward layers relative to the original pre-trained model_. The green layers, which you see are the one trained on the domain specific dataset.


## Model Fine-tuning

![Phi2_Finetuning01 drawio](https://github.com/bala1802/Phi2/assets/22103095/d236ff7d-f621-4bbb-942b-d11f742bfd9c)

🔍 Quantize the 32-bit Language Model to 4-bit model. This technique reduces the memory and computation requirements of the Neural Network layer by representing the weights and activations in only 4 bits. Refer [quantization_utils.py](https://github.com/bala1802/Phi2/blob/main/quantization_utils.py)

🧠 Identify the Layers that require weight updates and freeze the rest during fine-tuning. Managing the layers this way will allow the crucial layers to adapt to the new domain-specific data, while preserving the rest of the parameters of the pre-trained model.

The layer names can be identified by printing the Architecture of the model

<img width="970" alt="image" src="https://github.com/bala1802/Phi2/assets/22103095/741e61fe-57a3-4fc6-addc-b0cd7c87c4bc">

💡 LoRA, an adapter module, which will hold its own smaller set of parameters, which are learnt during the fine-tuning, enhancing the model's flexibility and adaptability to the domain specific nuances. Refer [adapter_utils.py](https://github.com/bala1802/Phi2/blob/main/adapter_utils.py)

📚 A dataset tailored specifically to the domain is constructed as Instructions and used as a training dataset for the fine-tuning process. Refer [data_utils.py](https://github.com/bala1802/Phi2/blob/main/data_utils.py)

## Inferencing 

![Phi2-Inferencing drawio](https://github.com/bala1802/Phi2/assets/22103095/49a511c7-7390-42cc-8307-51207823a516)

🔍 Extract the `Adapter` from the Fine-tuned 4-Bit Quantized model. This `Adapter` encapsulates the refined parameters tailored to the domain-sepcific data.

🧩 Integrate the `Adapter` with the original Pre-Trained 32-Bit model. This fusion enables the Language Model with the domain knowledge acquired during the fine-tuning process.

💬 User provides the prompt to the Langauage Model for interaction

🚀 The Language Model generates the response for the provided Prompt.

## Sample Inferencing

<img width="1248" alt="image" src="https://github.com/bala1802/Phi2/assets/22103095/e86fdafe-fa5e-4774-ba64-5ada01feb9a4">






