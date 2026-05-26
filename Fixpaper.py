"""Fixpaper — Claude × GPT 글 교정 Streamlit 앱."""
from __future__ import annotations

import os
import io
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

import anthropic
import openai
import pdfplumber
from docx import Document

load_dotenv()

# ── File parsers ──────────────────────────────────────────────────────────────

def parse_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF bytes, one page per line."""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        pages = [p.extract_text() for p in pdf.pages]
    return "\n".join(p for p in pages if p)


def parse_docx(file_bytes: bytes) -> str:
    """Extract paragraph text from DOCX bytes."""
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


# ── Prompt builders ───────────────────────────────────────────────────────────

def build_round1_prompt(text: str) -> str:
    return f"""당신은 글쓰기 전문가입니다.
아래 글을 분석하고 개선이 필요한 부분을 구체적으로 제안하세요.
문체, 논리 흐름, 표현의 명확성, 문법을 기준으로 검토하세요.
입력된 글과 동일한 언어로 답변하세요.

<원본글>
{text}
</원본글>"""


def build_round2_prompt(text: str, round1: str) -> str:
    return f"""아래는 원본 글과 Claude의 개선 제안입니다.
Claude의 제안 중 동의하는 부분, 보완할 부분, 반박할 부분을 나눠 답하세요.
Claude가 놓친 부분이 있다면 추가로 지적하세요.
입력된 글과 동일한 언어로 답변하세요.

<원본글>
{text}
</원본글>

<claude_제안>
{round1}
</claude_제안>"""


def build_round3_prompt(text: str, round1: str, round2: str) -> str:
    return f"""당신은 두 전문가의 토론을 종합하는 편집장입니다.
자신의 초기 제안과 GPT의 피드백을 모두 반영하여 아래 형식으로 정확히 답하세요.
입력된 글과 동일한 언어로 답변하세요.

## 변경 사항 요약
각 변경 항목을 "- [변경 내용]: [이유]" 형식으로 나열하세요.

## 최종 수정본
완성된 수정 글을 여기에 작성하세요.

<원본글>
{text}
</원본글>

<내_초기_제안>
{round1}
</내_초기_제안>

<gpt_피드백>
{round2}
</gpt_피드백>"""


# ── Output parser ─────────────────────────────────────────────────────────────

def parse_final_output(response: str) -> dict[str, str]:
    """Split Claude Round 3 response into summary and final_text."""
    summary = ""
    final_text = ""

    if "## 변경 사항 요약" in response and "## 최종 수정본" in response:
        parts = response.split("## 최종 수정본", 1)
        summary_raw = parts[0].split("## 변경 사항 요약", 1)[-1]
        summary = summary_raw.strip()
        final_text = parts[1].strip()

    return {"summary": summary, "final_text": final_text}


# ── API callers ───────────────────────────────────────────────────────────────

def call_claude(prompt: str, client: anthropic.Anthropic, max_tokens: int = 2048) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def call_gpt(prompt: str, client: openai.OpenAI) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
    )
    return response.choices[0].message.content


# ── Streamlit UI ──────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="Fixpaper — Claude × GPT 글 교정", page_icon="✏️", layout="wide")
    st.title("✏️ Fixpaper — Claude × GPT 글 교정")

    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")

    # ── Input ──────────────────────────────────────────────────────────────
    st.subheader("📄 글 입력")
    uploaded_file = st.file_uploader("파일 업로드 (PDF 또는 DOCX)", type=["pdf", "docx"])

    text_input = st.text_area("또는 직접 입력", height=200, placeholder="여기에 글을 붙여넣으세요...")

    # Resolve input text
    input_text: Optional[str] = None
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        if uploaded_file.name.lower().endswith(".pdf"):
            input_text = parse_pdf(file_bytes)
        else:
            input_text = parse_docx(file_bytes)
        st.success(f"파일 로드 완료: {len(input_text)}자")
    elif text_input.strip():
        input_text = text_input.strip()

    MAX_CHARS = 50_000
    if input_text and len(input_text) > MAX_CHARS:
        st.warning(f"입력 글이 너무 깁니다 ({len(input_text):,}자). {MAX_CHARS:,}자 이하로 줄여주세요.")
        input_text = None

    # ── Run ────────────────────────────────────────────────────────────────
    if st.button("🚀 교정 시작", disabled=not input_text):
        if not anthropic_key:
            st.error("Anthropic API 키를 입력해주세요.")
            return
        if not openai_key:
            st.error("OpenAI API 키를 입력해주세요.")
            return

        try:
            claude_client = anthropic.Anthropic(api_key=anthropic_key)
            gpt_client = openai.OpenAI(api_key=openai_key)

            with st.spinner("Round 1 진행 중 — Claude 분석..."):
                round1 = call_claude(build_round1_prompt(input_text), claude_client)

            with st.spinner("Round 2 진행 중 — GPT 반박/보완..."):
                round2 = call_gpt(build_round2_prompt(input_text, round1), gpt_client)

            with st.spinner("Round 3 진행 중 — Claude 최종 통합..."):
                round3_raw = call_claude(build_round3_prompt(input_text, round1, round2), claude_client, max_tokens=4096)
                round3 = parse_final_output(round3_raw)

        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")
            return

        # ── Results ────────────────────────────────────────────────────────
        st.divider()

        with st.expander("💬 토론 과정 보기"):
            st.markdown("**Round 1 — Claude 분석**")
            st.markdown(round1)
            st.markdown("---")
            st.markdown("**Round 2 — GPT 반박/보완**")
            st.markdown(round2)
            st.markdown("---")
            st.markdown("**Round 3 — Claude 통합 (원문)**")
            st.markdown(round3_raw)

        st.subheader("📋 변경 사항 요약")
        if round3["summary"]:
            st.markdown(round3["summary"])
        else:
            st.warning("변경 사항 요약을 파싱하지 못했습니다. 토론 과정 > Round 3 원문을 참고하세요.")

        st.subheader("✅ 최종 수정본")
        if round3["final_text"]:
            st.code(round3["final_text"], language=None)
        else:
            st.warning("최종 수정본을 파싱하지 못했습니다. 토론 과정 > Round 3 원문을 참고하세요.")


if __name__ == "__main__":
    main()
