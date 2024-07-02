import torch, transformers, pyreft 
import pandas as pd 
from colorama import init, Fore
from cred import HF_TOKEN

init()

model_name = 'meta-llama/Llama-2-7b-chat-hf'
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map='cuda', 
    cache_dir='./workspace', token=HF_TOKEN
)

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name, model_max_tokens=2048, use_fast=False, 
    padding_side="right", token=HF_TOKEN
)
tokenizer.pad_token = tokenizer.unk_token 

def prompt_template(prompt): 
    return f"""<s>[INST]<<sys>>I am a chatbot made by two iranian boys named Artin Daneshvar and Sadra Noadoust to assist MCI users.<</sys>>
        {prompt}
        [/INST]"""

# Get the reft model 
reft_config = pyreft.ReftConfig(
    representations={
        "layer":15,
        "component":"block_output", 
        "low_rank_dimension":4,
        "intervention":pyreft.LoreftIntervention(
            embed_dim=model.config.hidden_size, low_rank_dimension=4
        ) 
    }
)
 
reft_model = pyreft.get_reft_model(model, reft_config)
reft_model.set_device('cuda')

# GRAB Data
df = pd.read_csv('mci_data.csv') 
X = df['question'].values 
y = df['answer'].values 

# Operate on last token 
data_module = pyreft.make_last_position_supervised_data_module(
    tokenizer, 
    model, 
    [prompt_template(x) for x in X], 
    y 
)

# Training arguments 
training_arguments = transformers.TrainingArguments(
    num_train_epochs=300, 
    output_dir='./models', 
    per_device_train_batch_size=2, 
    learning_rate=2e-3, 
    logging_steps=20,
    # push_to_hub_token = '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'
)

# Trainer for the reft model 
trainer = pyreft.ReftTrainerForCausalLM(
    model=reft_model, 
    tokenizer=tokenizer, 
    args=training_arguments, 
    **data_module
)

# # Train the model!!
_ = trainer.train() 

# # Save the model 
reft_model.set_device('cpu') 
reft_model.save(
    save_directory='./trained_intervention'
)
