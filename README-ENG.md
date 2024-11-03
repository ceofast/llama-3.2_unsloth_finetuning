# Language Model Training and Fine-Tuning - Efficient Training with Unsloth

This project is developed to efficiently fine-tune large language models using the Unsloth library. Unsloth is a Python library that optimizes memory usage and training speed, particularly offering the ability to train language models with lower hardware requirements through features like GPU memory management and low-bit quantization.

## What is Unsloth?

Unsloth is a Python library that offers various optimizations for training large language models more quickly and in a memory-efficient way. Unsloth is compatible with PEFT (Parameter-Efficient Fine-Tuning) techniques and works with methods like LoRA (Low-Rank Adaptation), enabling the use of larger models with lower memory.

## Code Functionality Overview
### 1. Installing Necessary Libraries

The code requires libraries such as `pip3-autoremove`, `torch`, `xformers`, `unsloth`, and `transformers` to be installed. These libraries are necessary for model training and GPU memory management.
```
# Installing packages
!pip install pip3-autoremove
!pip-autoremove torch torchvision torchaudio -y
!pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu121
!pip install unsloth
!pip install --upgrade --no-cache-dir transformers
```

### 2. Model Configuration

The code sets parameters to use 4-bit quantized pre-trained models. These configurations reduce memory requirements, allowing work with larger models.

#### Model Configuration
```
max_seq_length = 2048
dtype = None
load_in_4bit = True
```

### 3. Loading the Pre-trained Model
The FastLanguageModel.from_pretrained function loads a Unsloth-supported 4-bit language model, enabling faster loading and memory savings.

#### Loading the Model
```
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
```

### 4. Fine-Tuning the Model
Techniques like LoRA (Low-Rank Adaptation) and PEFT (Parameter-Efficient Fine-Tuning) are applied to the model. This saves memory and speeds up the fine-tuning process.

#### Fine-Tuning with LoRA and PEFT
```
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)
```

### 5. Adapting to Chat Templates
At this stage, the `get_chat_template` function customizes the model according to a specific chat template, improving performance in chat-based tasks.

#### Applying Chat Templates
```
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
```

### 6. Preparing and Standardizing the Dataset
In this step, the `FineTome-100k` dataset is loaded and converted into a chat-based format. The `formatting_prompts_func` function formats the data for chat compatibility.

#### Loading and Preparing the Dataset
```
# Loading and preparing the dataset
dataset = load_dataset("mlabonne/FineTome-100k", split="train")
dataset = standardize_sharegpt(dataset)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)
```

### 7. Configuring the Trainer
Here, `SFTTrainer` is used to set training parameters for fine-tuning, optimizing memory usage for efficient model performance.

#### Trainer Configuration
```
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
    ),
)
```

### 8. Monitoring GPU Memory Usage
This section monitors the current GPU memory usage. Memory consumption is measured before and after training to observe how much memory the model occupies.

#### Monitoring GPU Memory Usage
```
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")
```

### 9. Training the Model and Reporting Memory Usage
The model is trained using the `unsloth_train` function, which provides a detailed report on memory usage after training.

#### Training and Memory Report
```
# Model training and memory report
from unsloth import unsloth_train
trainer_stats = unsloth_train(trainer)
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
```

### 10. Saving the Model for Local/Automated Use
After training, the model and tokenizer are saved and can optionally be uploaded to Hugging Face Hub for direct loading in future use.

#### Saving the Model
```
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
```

### 11. Model Usage (Example)
This section provides a simple example for testing the trained model, where the model continues or generates a response to a user-provided message.

#### Using the Model
```
# Using the model
messages = [{"role": "user", "content": "Continue the Fibonacci sequence: 1, 1, 2, 3, 5, 8,"}]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")
```

## References
This code is developed with reference to the following Kaggle Notebook:

```
https://www.kaggle.com/code/danielhanchen/fixed-kaggle-llama-3-2-1b-3b-conversation
```
