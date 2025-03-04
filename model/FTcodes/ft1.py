pip install trl
pip install transformers accelerate peft datasets bitsandbytes torch

import os
import torch
import logging
import numpy as np
from datasets import Dataset, load_dataset
import transformers
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling, pipeline)
from peft import (prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType)
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
import re
#from trl import SFTTrainer


############
#Functions

def curate_dataset(data):
    curated_data = {'train':[]}
    
    for entry in data['train']:
        prompt = entry['prompt']
        response = entry['response']
        
        persona_match = re.match(r'<(\w+)>', prompt)
        persona = persona_match.group(1) if persona_match else "default"
        
        clean_prompt = re.sub(r'<\w+>\s*', '', prompt, 1)
        
        system_message = {
            "role": "system",
            "content": f"You are <{persona}>, an expert in your field. "
                       f"Answer the following questions based on your expertise:"
        }
        user_message = {
            "role": "user",
            "content": clean_prompt
        }
        assistant_message = {
            "role": "assistant",
            "content": response
        }
        
        curated_data['train'].append([system_message, user_message, assistant_message])
    
    return curated_data

# we use this to skip extra step for tokenisation
def apply_template(dataP):
    return tokenizer.apply_chat_template(dataP, tokenize=True)

#Generating text from the fine tuned model - WOYM
def woym_generate(query='new',persona=''):
    query = str(input('Whats on your mind?'))
    persona = str(input('Which expert do you want to asnwer your question?'))
    #hard coded for now
    persona = '<einstein>,'
    messages = [
        {
            "role": "system",
            "content": f"You are {persona} an expert in your field. Answer the following questions based on your expertise:",
        },
        {"role": "user", "content": query},
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    
    print(outputs[0]["generated_text"])

###############

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

#torch.set_default_device("cuda")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Loading the dataset from my Hugging Face page
ds = load_dataset("aliMohammad16/einstein_answers")

#Preparing the data
curated_data = curate_dataset(ds)

tokenised_dataset = [apply_template(dPoint) for dPoint in curated_data['train']]

#Splitting into train and validation sets
train_tokenised_dataset, test_tokenised_dataset = train_test_split(tokenised_dataset, test_size = 0.1, random_state = 42)

#setting up FT parameters
lora_config = LoraConfig(
    r=16,  
    lora_alpha=32,  
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

#training args
training_args = TrainingArguments(
    output_dir="./woym1",
    report_to="none", 
    run_name = 'woym1',
    per_device_train_batch_size=2,  # Adjust for GPU memory
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    bf16=True,  # Use mixed precision
    push_to_hub=False  # Set to True if uploading to Hugging Face Hub
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

#not needed, since our data was already loaded on cuda
#train_tokenised_dataset = train_tokenised_dataset.with_format("torch", device="cuda")
#test_tokenised_dataset = test_tokenised_dataset.with_format("torch", device="cuda")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenised_dataset,
    eval_dataset = test_tokenised_dataset,
    data_collator=data_collator
)

#Training
model.to("cuda")

trainer.train() #Woym is being trained :)

#Saving WOYM
model.save_pretrained("./woym-1")
tokenizer.save_pretrained("./woym-1")

## TEsting WOYM
# Load WOYM
model_path = "./woym-1" 
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, torch_dtype=torch.bfloat16, device_map="auto")

#Get queries answered by Expert personas, NOW!!
woym_generate()
