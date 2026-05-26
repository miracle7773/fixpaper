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
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        pages = [p.extract_text() for p in pdf.pages]
    return "\n".join(p for p in pages if p)


def parse_docx(file_bytes: bytes) -> str:
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
    return f"""당신은 글쓰기 전문가입니다.
자신의 초기 제안과 GPT의 피드백을 반영하여 중간 수정안을 작성하세요.
GPT의 반박 중 타당한 부분은 수용하고, 동의하지 않는 부분은 이유를 밝히세요.
완성된 중간 수정본을 작성하세요.
입력된 글과 동일한 언어로 답변하세요.

<원본글>
{text}
</원본글>

<내_초기_제안>
{round1}
</내_초기_제안>

<gpt_피드백>
{round2}
</gpt_피드백>"""


def build_round4_prompt(text: str, round1: str, round2: str, round3: str) -> str:
    return f"""아래는 원본 글과 두 전문가의 토론 과정, 그리고 Claude의 중간 수정안입니다.
Claude의 중간 수정안을 검토하고 최종 의견을 제시하세요.
여전히 개선이 필요한 부분, 잘 반영된 부분, 추가 제안을 명확히 구분하여 답하세요.
입력된 글과 동일한 언어로 답변하세요.

<원본글>
{text}
</원본글>

<claude_초기제안>
{round1}
</claude_초기제안>

<gpt_1차피드백>
{round2}
</gpt_1차피드백>

<claude_중간수정안>
{round3}
</claude_중간수정안>"""


def build_round5_prompt(text: str, round1: str, round2: str, round3: str, round4: str) -> str:
    return f"""당신은 두 전문가의 토론을 종합하는 편집장입니다.
모든 토론 과정을 반영하여 아래 형식으로 정확히 답하세요.
입력된 글과 동일한 언어로 답변하세요.

## 변경 사항 요약
각 변경 항목을 "- [변경 내용]: [이유]" 형식으로 나열하세요.

## 최종 수정본
완성된 수정 글을 여기에 작성하세요.

<원본글>
{text}
</원본글>

<claude_초기제안>
{round1}
</claude_초기제안>

<gpt_1차피드백>
{round2}
</gpt_1차피드백>

<claude_중간수정안>
{round3}
</claude_중간수정안>

<gpt_최종검토>
{round4}
</gpt_최종검토>"""


def build_chat_system(original_text: str, current_text: str) -> str:
    if current_text and current_text != original_text:
        return f"""당신은 글쓰기 편집 보조입니다.
아래는 사용자의 원본 글과 현재 수정본입니다.
사용자의 요청에 따라 수정본을 다듬어주세요.
요청한 부분만 수정하고 나머지는 건드리지 마세요.
입력된 글과 동일한 언어로 답변하세요.

<원본글>
{original_text}
</원본글>

<현재_수정본>
{current_text}
</현재_수정본>"""
    else:
        return f"""당신은 글쓰기 편집 보조입니다.
아래는 사용자의 글입니다.
사용자의 요청에 따라 글을 다듬어주세요.
요청한 부분만 수정하고 나머지는 건드리지 마세요.
입력된 글과 동일한 언어로 답변하세요.

<글>
{original_text}
</글>"""


# ── Output parser ─────────────────────────────────────────────────────────────

def parse_final_output(response: str) -> dict[str, str]:
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


def call_claude_chat(
    messages: list[dict],
    system: str,
    client: anthropic.Anthropic,
) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=system,
        messages=messages,
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

    try:
        anthropic_key = st.secrets.get("ANTHROPIC_API_KEY", "") or os.getenv("ANTHROPIC_API_KEY", "")
        openai_key = st.secrets.get("OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
    except Exception:
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        openai_key = os.getenv("OPENAI_API_KEY", "")

    for key, default in [
        ("debate_done", False),
        ("original_text", ""),
        ("current_text", ""),
        ("debate_data", {}),
        ("chat_messages", []),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ── 글 입력 섹션 (항상 표시) ──────────────────────────────────────────
    with st.container():
        st.subheader("📄 글 입력")

        if st.session_state.original_text:
            st.info(f"현재 글 로드됨 ({len(st.session_state.original_text):,}자)")
            if st.button("🔄 새 글로 바꾸기"):
                for key in ["debate_done", "original_text", "current_text", "debate_data", "chat_messages"]:
                    del st.session_state[key]
                st.rerun()
        else:
            uploaded_file = st.file_uploader("파일 업로드 (PDF 또는 DOCX)", type=["pdf", "docx"])
            text_input = st.text_area("또는 직접 입력", height=200, placeholder="여기에 글을 붙여넣으세요...")

            input_text: Optional[str] = None
            if uploaded_file is not None:
                file_bytes = uploaded_file.getvalue()
                if uploaded_file.name.lower().endswith(".pdf"):
                    input_text = parse_pdf(file_bytes)
                else:
                    input_text = parse_docx(file_bytes)
                st.success(f"파일 로드 완료: {len(input_text):,}자")
            elif text_input.strip():
                input_text = text_input.strip()

            MAX_CHARS = 50_000
            if input_text and len(input_text) > MAX_CHARS:
                st.warning(f"입력 글이 너무 깁니다 ({len(input_text):,}자). {MAX_CHARS:,}자 이하로 줄여주세요.")
                input_text = None

            col1, col2 = st.columns(2)
            with col1:
                if st.button("💬 채팅 시작", disabled=not input_text, use_container_width=True):
                    st.session_state.original_text = input_text
                    st.session_state.current_text = input_text
                    st.rerun()
            with col2:
                if st.button("🚀 교정 후 채팅", disabled=not input_text, use_container_width=True):
                    if not anthropic_key:
                        st.error("Anthropic API 키가 설정되지 않았습니다.")
                        return
                    if not openai_key:
                        st.error("OpenAI API 키가 설정되지 않았습니다.")
                        return

                    try:
                        claude_client = anthropic.Anthropic(api_key=anthropic_key)
                        gpt_client = openai.OpenAI(api_key=openai_key)

                        with st.spinner("Round 1 진행 중 — Claude 분석..."):
                            round1 = call_claude(build_round1_prompt(input_text), claude_client)

                        with st.spinner("Round 2 진행 중 — GPT 반박/보완..."):
                            round2 = call_gpt(build_round2_prompt(input_text, round1), gpt_client)

                        with st.spinner("Round 3 진행 중 — Claude 중간 수정안..."):
                            round3 = call_claude(
                                build_round3_prompt(input_text, round1, round2),
                                claude_client,
                                max_tokens=4096,
                            )

                        with st.spinner("Round 4 진행 중 — GPT 최종 검토..."):
                            round4 = call_gpt(
                                build_round4_prompt(input_text, round1, round2, round3),
                                gpt_client,
                            )

                        with st.spinner("Round 5 진행 중 — Claude 최종 통합..."):
                            round5_raw = call_claude(
                                build_round5_prompt(input_text, round1, round2, round3, round4),
                                claude_client,
                                max_tokens=4096,
                            )
                            round5 = parse_final_output(round5_raw)

                    except Exception as e:
                        st.error(f"오류가 발생했습니다: {e}")
                        return

                    st.session_state.original_text = input_text
                    st.session_state.current_text = round5["final_text"] or round5_raw
                    st.session_state.debate_data = {
                        "round1": round1,
                        "round2": round2,
                        "round3": round3,
                        "round4": round4,
                        "round5_raw": round5_raw,
                        "round5": round5,
                    }
                    st.session_state.debate_done = True
                    st.rerun()

    # ── 교정 결과 (토론 완료 시) ──────────────────────────────────────────
    if st.session_state.debate_done:
        d = st.session_state.debate_data
        st.divider()

        with st.expander("💬 토론 과정 보기"):
            st.markdown("**Round 1 — Claude 분석**")
            st.markdown(d["round1"])
            st.markdown("---")
            st.markdown("**Round 2 — GPT 반박/보완**")
            st.markdown(d["round2"])
            st.markdown("---")
            st.markdown("**Round 3 — Claude 중간 수정안**")
            st.markdown(d["round3"])
            st.markdown("---")
            st.markdown("**Round 4 — GPT 최종 검토**")
            st.markdown(d["round4"])
            st.markdown("---")
            st.markdown("**Round 5 — Claude 최종 통합 (원문)**")
            st.markdown(d["round5_raw"])

        st.subheader("📋 변경 사항 요약")
        if d["round5"]["summary"]:
            st.markdown(d["round5"]["summary"])
        else:
            st.warning("변경 사항 요약을 파싱하지 못했습니다. 토론 과정 > Round 5 원문을 참고하세요.")

        st.subheader("✅ 최종 수정본")
        st.code(st.session_state.current_text, language=None)

    # ── 채팅 (글 로드 시 항상 표시) ──────────────────────────────────────
    if st.session_state.original_text:
        st.divider()
        st.subheader("💬 채팅")
        if st.session_state.debate_done:
            st.caption("수정본을 기반으로 원하는 부분을 자유롭게 요청하세요.")
        else:
            st.caption("글에 대해 자유롭게 수정 요청이나 질문을 해보세요.")

        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if user_input := st.chat_input("예: 3문단을 더 formal하게 해줘 / 결론만 다시 써줘"):
            if not anthropic_key:
                st.error("Anthropic API 키가 설정되지 않았습니다.")
                return

            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("..."):
                    claude_client = anthropic.Anthropic(api_key=anthropic_key)
                    reply = call_claude_chat(
                        messages=st.session_state.chat_messages,
                        system=build_chat_system(
                            st.session_state.original_text,
                            st.session_state.current_text,
                        ),
                        client=claude_client,
                    )
                st.markdown(reply)

            st.session_state.chat_messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
