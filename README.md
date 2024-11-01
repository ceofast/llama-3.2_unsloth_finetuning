# Dil Modeli Eğitimi ve İnce Ayar (Fine-tuning) - Unsloth Kullanarak Verimli Eğitim

Bu proje, **Unsloth** kütüphanesi ile büyük dil modelleri üzerinde verimli bir şekilde ince ayar (fine-tuning) yapmak amacıyla geliştirilmiştir. Unsloth, hafıza kullanımını ve eğitim hızını optimize eden bir Python kütüphanesidir. Özellikle GPU bellek yönetimi ve düşük bit kuantizasyonu gibi özelliklerle, dil modellerini düşük donanım gereksinimleriyle eğitme imkanı sunar.

## Unsloth Nedir?
**Unsloth**, büyük dil modellerini daha hızlı ve hafıza dostu bir şekilde eğitmek için çeşitli optimizasyonlar sunan bir Python kütüphanesidir. Özellikle PEFT (Parameter-Efficient Fine-Tuning) teknikleri ile uyumlu olan Unsloth, LoRA (Low-Rank Adaptation) gibi düşük bellekte verimli ince ayar yöntemleri ile çalışır. Bu sayede, daha az bellekte daha büyük modellerle çalışmak mümkün hale gelir.


## Kodun Bölüm Bölüm İşlevleri

### 1. Gerekli Kütüphanelerin Kurulumu

Kod, `pip3-autoremove`, `torch`, `xformers`, `unsloth`, ve `transformers` gibi kütüphanelerin yüklenmesi gerekmektedir. Bu kütüphaneler, modelin eğitimi ve GPU bellek yönetimi gibi görevler için gereklidir.

```python
# Paketlerin kurulması
!pip install pip3-autoremove
!pip-autoremove torch torchvision torchaudio -y
!pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu121
!pip install unsloth
!pip install --upgrade --no-cache-dir transformers
```

### 2. Model Konfigürasyonu
Kod, 4-bit kuantize edilmiş önceden eğitilmiş modelleri kullanmak üzere bazı parametreler ayarlar. Bu konfigürasyonlar, modelin bellek gereksinimini düşürerek daha büyük modellerle çalışmaya olanak tanır.

#### Model konfigürasyonu
```
max_seq_length = 2048
dtype = None
load_in_4bit = True
```

### 3. Önceden Eğitilmiş Modelin Yüklenmesi
FastLanguageModel.from_pretrained fonksiyonu, Unsloth destekli 4-bit bir dil modelini yükler. Bu sayede daha hızlı yükleme ve hafıza tasarrufu sağlanır.

#### Modelin yüklenmesi
```
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
```

### 4. Model Üzerinde İnce Ayar (Fine-tuning) Yapılması
Model üzerinde LoRA (Low-Rank Adaptation) ve PEFT (Parameter-Efficient Fine-Tuning) gibi optimizasyon teknikleri uygulanır. Bu, bellekte tasarruf yaparak ince ayar sürecini hızlandırır.

#### LoRA ve PEFT kullanarak ince ayar yapılması
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

### 5. Chat Şablonları ile Uyumlu Hale Getirme
Bu aşamada `get_chat_template` fonksiyonu kullanılarak model, belirli bir sohbet şablonuna göre özelleştirilir. Bu, modelin sohbet tabanlı görevlerde daha iyi performans göstermesine yardımcı olur.

#### Chat şablonlarının uygulanması
```
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
```

### 6. Veri Setinin Hazırlanması ve Standartlaştırılması
Bu adımda, `FineTome-100k` adlı veri seti yüklenir ve sohbet tabanlı bir formata dönüştürülür. `formatting_prompts_func` fonksiyonu ile veriler, sohbet için uygun hale getirilir.
#### Veri setinin yüklenmesi ve hazırlanması
```
# Veri setinin yüklenmesi ve hazırlanması
dataset = load_dataset("mlabonne/FineTome-100k", split="train")
dataset = standardize_sharegpt(dataset)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)
```

### 7. Eğitici (Trainer) Yapılandırması
Burada, modelin ince ayarı (fine-tuning) için `SFTTrainer` kullanılarak eğitim parametreleri belirlenir. Bu ayarlar, bellek kullanımını optimize ederek modelin daha verimli çalışmasını sağlar.
#### Eğitici yapılandırması
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

### 8. GPU Bellek Kullanımının İzlenmesi
Bu kısımda, GPU’nun mevcut bellek kullanımı izlenir. Eğitimden önce ve sonra bellek tüketimi ölçülerek modelin bellekte ne kadar yer kapladığı gözlemlenir.

#### GPU bellek kullanımının izlenmesi
```
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")
```

### 9. Modeli Eğitme ve Bellek Kullanımı Raporlama
`unsloth_train` fonksiyonu ile modelin eğitimi gerçekleştirilir. Eğitim sonrasında bellek kullanımına dair detaylı rapor verilir.
#### Model eğitimi ve bellek raporu
```
# Model eğitimi ve bellek raporu
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

### 10. Modelin Kaydedilmesi ve Yerel/Otomatik Kullanımı
Eğitimden sonra model ve tokenizer kaydedilir ve istenirse Hugging Face Hub’a yüklenebilir. Bu, ilerideki kullanımda doğrudan modeli yüklemeye imkan tanır.
#### Modelin kaydedilmesi
```
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
```

### 11. Modelin Kullanımı (Örnek)
Bu kısım, eğitilmiş modeli test etmek için basit bir örnek sunar. Model, kullanıcı tarafından verilen bir mesajı devam ettirir veya yanıt üretir.
#### Modelin kullanımı
```
# Modelin kullanımı
messages = [{"role": "user", "content": "Fibonacci dizisini devam ettirin: 1, 1, 2, 3, 5, 8,"}]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")
```

## Kaynakça
Bu kod, aşağıdaki Kaggle Notebook'tan referans alınarak geliştirilmiştir:
```
https://www.kaggle.com/code/danielhanchen/fixed-kaggle-llama-3-2-1b-3b-conversation
```
