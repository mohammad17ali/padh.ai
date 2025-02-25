#Functions

def tokenize_function(examples):
    return tokenizer(
        examples["prompt"], 
        text_target=examples["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=512  # Adjust based on your needs
    )


def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_length=200)
    return tokenizer.decode(output[0], skip_special_tokens=True)

class DebugCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        print(logs) 

#def load_dset(url):
    
