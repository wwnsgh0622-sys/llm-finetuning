import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from train import (
    get_sample_data, load_tokenizer, tokenize_data,
    load_model_with_lora, train, generate_text,
    MODEL_NAME, OUTPUT_DIR
)
import os
import warnings
warnings.filterwarnings("ignore")

# ── 페이지 설정 ──────────────────────────────────────
st.set_page_config(
    page_title="🤖 LLM Fine-tuning (LoRA)",
    page_icon="🤖",
    layout="wide"
)

# ── CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ── 헤더 ────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🤖 LLM Fine-tuning (LoRA)</h1>
    <p>한국어 GPT-2 모델을 LoRA로 효율적으로 파인튜닝</p>
    <p>HuggingFace | PEFT | LoRA | KoGPT-2</p>
</div>
""", unsafe_allow_html=True)

# ── 사이드바 ─────────────────────────────────────────
st.sidebar.title("⚙️ 설정")
mode = st.sidebar.radio("모드 선택", [
    "📚 LoRA 개념 설명",
    "🚀 모델 학습",
    "✍️ 텍스트 생성"
])

# ── 모드 1: LoRA 개념 설명 ────────────────────────────
if mode == "📚 LoRA 개념 설명":
    st.title("📚 LoRA란?")

    col1, col2, col3 = st.columns(3)
    col1.metric("학습 파라미터", "~1%", "전체 대비")
    col2.metric("메모리 절약", "~70%", "풀 파인튜닝 대비")
    col3.metric("성능", "유사", "풀 파인튜닝과")

    st.markdown("---")
    st.markdown("""
    ## 🔍 LoRA (Low-Rank Adaptation)
    
    **LoRA**는 대규모 언어 모델을 효율적으로 파인튜닝하는 기법이에요!
    
    ### 핵심 아이디어
    - 기존 모델 가중치는 **동결 (Freeze)**
    - 작은 **저랭크 행렬 (Low-Rank Matrix)** 만 추가 학습
    - 전체 파라미터의 **1%만 학습**해도 좋은 성능!
    
    ### 왜 중요한가?
    - GPT-4 같은 모델은 수천억 파라미터
    - 풀 파인튜닝은 GPU 수십 개 필요
    - LoRA는 **개인 PC GPU로도 가능!**
    """)

    st.markdown("---")
    st.subheader("📊 LoRA vs 풀 파인튜닝 비교")

    comparison = pd.DataFrame({
        "방법": ["풀 파인튜닝", "LoRA", "QLoRA"],
        "학습 파라미터": ["100%", "~1%", "~1%"],
        "GPU 메모리": ["매우 많음", "적음", "매우 적음"],
        "학습 속도": ["느림", "빠름", "빠름"],
        "성능": ["최고", "유사", "유사"]
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    fig = go.Figure(go.Bar(
        x=["풀 파인튜닝", "LoRA", "QLoRA"],
        y=[100, 1, 0.5],
        marker_color=["#ff6b6b", "#6bcb77", "#4d96ff"],
        text=["100%", "~1%", "~0.5%"],
        textposition="auto"
    ))
    fig.update_layout(
        title="학습 파라미터 비율 비교",
        yaxis_title="학습 파라미터 (%)",
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    st.plotly_chart(fig, use_container_width=True)

# ── 모드 2: 모델 학습 ────────────────────────────────
elif mode == "🚀 모델 학습":
    st.title("🚀 KoGPT-2 LoRA 파인튜닝")

    st.info("💡 한국어 GPT-2 모델을 AI/데이터 사이언스 관련 텍스트로 파인튜닝합니다!")

    st.subheader("📋 학습 데이터 샘플")
    dataset = get_sample_data()
    df = dataset.to_pandas()
    st.dataframe(df, use_container_width=True, hide_index=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("데이터 수", f"{len(df)}개")
    col2.metric("모델", "KoGPT-2")
    col3.metric("방법", "LoRA (r=8)")

    st.markdown("---")

    if st.button("🚀 LoRA 파인튜닝 시작 (3~5분 소요)"):
        with st.spinner("모델 로딩 중..."):
            tokenizer = load_tokenizer()
            model = load_model_with_lora()

        st.success("✅ 모델 로드 완료!")

        # 학습 파라미터 정보
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        st.metric("학습 파라미터", f"{trainable_params:,}개 ({trainable_params/total_params*100:.2f}%)")

        with st.spinner("데이터 전처리 중..."):
            tokenized = tokenize_data(dataset, tokenizer)

        with st.spinner("파인튜닝 학습 중... (3~5분)"):
            trainer = train(model, tokenizer, tokenized)

        st.success("🎉 파인튜닝 완료! `kogpt2-finetuned` 폴더에 저장됐어요!")
        st.balloons()

# ── 모드 3: 텍스트 생성 ──────────────────────────────
elif mode == "✍️ 텍스트 생성":
    st.title("✍️ 파인튜닝된 모델로 텍스트 생성")

    @st.cache_resource
    def load_finetuned_model():
        if os.path.exists(OUTPUT_DIR):
            tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
            model = PeftModel.from_pretrained(model, OUTPUT_DIR)
            return model, tokenizer, True
        else:
            tokenizer = load_tokenizer()
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
            return model, tokenizer, False

    with st.spinner("모델 로딩 중..."):
        model, tokenizer, is_finetuned = load_finetuned_model()

    if is_finetuned:
        st.success("✅ 파인튜닝된 모델 로드 완료!")
    else:
        st.warning("⚠️ 파인튜닝된 모델이 없어요! 먼저 '모델 학습' 탭에서 학습해주세요. 기본 모델로 생성합니다.")

    prompt = st.text_input("프롬프트 입력", "인공지능은")
    max_length = st.slider("생성 길이", 50, 200, 100)

    if st.button("✍️ 텍스트 생성"):
        with st.spinner("생성 중..."):
            generated = generate_text(prompt, model, tokenizer, max_length)

        st.markdown("### 📝 생성된 텍스트")
        st.write(generated)