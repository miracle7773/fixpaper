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

[원본 글]
{text}"""


def build_round2_prompt(text: str, round1: str) -> str:
    return f"""아래는 원본 글과 Claude의 개선 제안입니다.
Claude의 제안 중 동의하는 부분, 보완할 부분, 반박할 부분을 나눠 답하세요.
Claude가 놓친 부분이 있다면 추가로 지적하세요.
입력된 글과 동일한 언어로 답변하세요.

[원본 글]
{text}

[Claude 제안]
{round1}"""


def build_round3_prompt(text: str, round1: str, round2: str) -> str:
    return f"""당신은 두 전문가의 토론을 종합하는 편집장입니다.
자신의 초기 제안과 GPT의 피드백을 모두 반영하여 아래 형식으로 정확히 답하세요.
입력된 글과 동일한 언어로 답변하세요.

## 변경 사항 요약
각 변경 항목을 "- [변경 내용]: [이유]" 형식으로 나열하세요.

## 최종 수정본
완성된 수정 글을 여기에 작성하세요.

[원본 글]
{text}

[내 초기 제안]
{round1}

[GPT 피드백]
{round2}"""


# ── Output parser ─────────────────────────────────────────────────────────────

# ── API callers ───────────────────────────────────────────────────────────────

# ── Debate orchestrator ───────────────────────────────────────────────────────

# ── Streamlit UI ──────────────────────────────────────────────────────────────
