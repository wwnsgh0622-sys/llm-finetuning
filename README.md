# 🤖 LLM Fine-tuning (LoRA)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
</p>

<p align="center">
  한국어 GPT-2 모델을 LoRA로 효율적으로 파인튜닝<br/>
  전체 파라미터의 0.24%만 학습으로 효율적인 파인튜닝 달성
</p>

---

## 🎯 주요 기능

- 📚 **LoRA 개념 설명** — 풀 파인튜닝 vs LoRA vs QLoRA 비교
- 🚀 **모델 학습** — KoGPT-2 LoRA 파인튜닝
- ✍️ **텍스트 생성** — 파인튜닝된 모델로 한국어 텍스트 생성

---

## 🏆 성과

| 항목 | 내용 |
|------|------|
| 모델 | KoGPT-2 (skt/kogpt2-base-v2) |
| 방법 | LoRA (r=8, alpha=32) |
| 학습 파라미터 | 294,912개 (0.24%) |
| 메모리 절약 | ~70% (풀 파인튜닝 대비) |

---

## ⚙️ Setup

```bash
git clone https://github.com/wwnsgh0622-sys/llm-finetuning.git
cd llm-finetuning
python -m venv llm-env
llm-env\Scripts\activate
pip install -r requirements.txt
```

## 🚀 Run

```bash
streamlit run app.py
```

---

## 🛠️ Tech Stack

- **모델**: KoGPT-2 (skt/kogpt2-base-v2)
- **파인튜닝**: PEFT (LoRA)
- **프레임워크**: PyTorch, HuggingFace Transformers
- **시각화**: Streamlit, Plotly

---

## 👤 Author

**문준호** · [wwnsgh0622-sys](https://github.com/wwnsgh0622-sys)
Chung-Ang University, Software Engineering
📧 wwnsgh0622@gmail.com