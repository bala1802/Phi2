import datasets
from datasets import load_dataset

import config

def download(mode):
    print("Downloading Dataset - ", config.DATASET, "...")
    dataset = load_dataset(config.DATASET, split=mode)
    return dataset

def prepare_prompts_responses(dataset):
    print("Preparing Prompt and Assistant....")
    dataset_df = dataset.to_pandas()
    user_prompters = dataset_df[(dataset_df.role=="prompter")]
    user_prompters = user_prompters.set_index("message_id")
    assistants = dataset_df[(dataset_df.role=="assistant") & (dataset_df["rank"] == 0.0)]
    
    prompts_responses = []
    for _,record in assistants.iterrows():
        prompt_text = user_prompters.loc[record.parent_id,'text']
        prompt_response = "### Human: " + prompt_text + " ### Assistant: " + record['text']
        prompts_responses.append(prompt_response)
    assistants['prompt_response'] = prompts_responses
    
    return assistants

def preparedata(mode):
    print("Preparing data for - ", mode, "...")
    dataset = download(mode=mode)
    prompts_responses = prepare_prompts_responses(dataset)
    prompts_responses_dataset = datasets.Dataset.from_pandas(prompts_responses)
    return prompts_responses_dataset
    


