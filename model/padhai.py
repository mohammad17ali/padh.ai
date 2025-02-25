import torch
import torchvision
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, TrainerCallback
from datasets import load_dataset
from peft import LoraConfig, get_peft_model


#Dataset - Primary Students-Teacher interactions
ds = load_dataset("ajibawa-2023/Education-Young-Children")

#Model - Tiny Llama 
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

#LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
#QLoRA
peft_config = LoraConfig(
    r=8, 
    lora_alpha=32, 
    target_modules=["q_proj", "v_proj"], 
    lora_dropout=0.05
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()



# Tokenize dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Remove text columns
tokenized_datasets = tokenized_datasets.remove_columns(["prompt", "text", "text_token_length"])

#Training
training_args = TrainingArguments(
    fp16=True,
    output_dir="./results",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    save_strategy="epoch",
    logging_dir="./logs",
    run_name="padh.ai_FT",
    report_to="none", 
    save_steps=5000,  
    logging_steps=100,
)


small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(10000))

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    callbacks=[DebugCallback()] 
)

#Train
trainer.train()

model.save_pretrained("./tinyllama_finetuned")
tokenizer.save_pretrained("./tinyllama_finetuned")

#test
prompt = "Explain the importance of renewable energy sources."
print(generate_text(prompt))
