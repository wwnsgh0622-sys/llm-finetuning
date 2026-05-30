import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
from torch.utils.data import Dataset as TorchDataset
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

MODEL_NAME = "skt/kogpt2-base-v2"
OUTPUT_DIR = "./kogpt2-finetuned"

TEXTS = [
    "인공지능은 미래 기술의 핵심입니다.",
    "데이터 사이언티스트는 데이터를 분석하고 인사이트를 도출하는 전문가입니다.",
    "PyTorch와 TensorFlow는 딥러닝 프레임워크의 양대 산맥입니다.",
    "자연어 처리는 컴퓨터가 인간의 언어를 이해하고 생성하는 기술입니다.",
    "강화학습은 보상을 최대화하는 방향으로 에이전트를 학습시키는 방법입니다.",
    "컴퓨터 비전은 이미지와 영상을 분석하는 인공지능 분야입니다.",
    "트랜스포머 모델은 어텐션 메커니즘을 기반으로 한 혁신적인 딥러닝 구조입니다.",
    "대규모 언어 모델은 방대한 텍스트 데이터로 학습된 강력한 AI 모델입니다.",
    "파인튜닝은 사전 학습된 모델을 특정 태스크에 맞게 추가 학습하는 기법입니다.",
    "LoRA는 대규모 언어 모델을 효율적으로 파인튜닝하는 방법입니다.",
    "중앙대학교 소프트웨어학부에서 AI와 데이터 사이언스를 공부하고 있습니다.",
    "딥러닝 프로젝트를 통해 실무 경험을 쌓고 대학원 진학을 목표로 합니다.",
    "GitHub에 포트폴리오를 정리하여 지식과 경험을 공유하고 있습니다.",
    "Streamlit을 활용하면 머신러닝 모델을 빠르게 웹앱으로 배포할 수 있습니다.",
    "BERT와 GPT는 현대 자연어 처리의 기반이 되는 대표적인 언어 모델입니다.",
]

def get_sample_data():
    from datasets import Dataset
    return Dataset.from_pandas(pd.DataFrame({"text": TEXTS}))

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        bos_token='</s>', eos_token='</s>', unk_token='<unk>',
        pad_token='<pad>', mask_token='<mask>'
    )
    return tokenizer

class TextDataset(TorchDataset):
    def __init__(self, tokenizer, max_length=64):
        self.examples = []
        vocab_size = tokenizer.vocab_size
        for text in TEXTS:
            tokenized = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt"
            )
            input_ids = tokenized["input_ids"].squeeze()
            # vocab 범위 클램핑
            input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
            self.examples.append({
                "input_ids": input_ids,
                "attention_mask": tokenized["attention_mask"].squeeze(),
                "labels": input_ids.clone()
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def tokenize_data(dataset, tokenizer, max_length=64):
    return TextDataset(tokenizer, max_length)

def get_lora_config():
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["c_attn"]
    )

def load_model_with_lora():
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

def train(model, tokenizer, tokenized_dataset):
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        warmup_steps=5,
        logging_steps=5,
        save_steps=100,
        learning_rate=2e-4,
        fp16=torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    return trainer

def generate_text(prompt, model, tokenizer, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = torch.clamp(inputs["input_ids"], 0, tokenizer.vocab_size - 1)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)