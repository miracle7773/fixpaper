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

# ── Output parser ─────────────────────────────────────────────────────────────

# ── API callers ───────────────────────────────────────────────────────────────

# ── Debate orchestrator ───────────────────────────────────────────────────────

# ── Streamlit UI ──────────────────────────────────────────────────────────────
