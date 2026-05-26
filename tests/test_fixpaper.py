import io
import pytest
from unittest.mock import patch, MagicMock

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from Fixpaper import parse_pdf, parse_docx


def test_parse_pdf_returns_text():
    """parse_pdf extracts text from each page and joins with newlines."""
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Hello world"
    mock_pdf = MagicMock()
    mock_pdf.__enter__ = lambda s: s
    mock_pdf.__exit__ = MagicMock(return_value=False)
    mock_pdf.pages = [mock_page, mock_page]

    with patch("Fixpaper.pdfplumber.open", return_value=mock_pdf):
        result = parse_pdf(b"fake pdf bytes")

    assert result == "Hello world\nHello world"


def test_parse_pdf_skips_none_pages():
    """parse_pdf skips pages where extract_text returns None."""
    mock_page_none = MagicMock()
    mock_page_none.extract_text.return_value = None
    mock_page_ok = MagicMock()
    mock_page_ok.extract_text.return_value = "Content"
    mock_pdf = MagicMock()
    mock_pdf.__enter__ = lambda s: s
    mock_pdf.__exit__ = MagicMock(return_value=False)
    mock_pdf.pages = [mock_page_none, mock_page_ok]

    with patch("Fixpaper.pdfplumber.open", return_value=mock_pdf):
        result = parse_pdf(b"fake pdf bytes")

    assert result == "Content"


def test_parse_docx_returns_paragraphs():
    """parse_docx joins all non-empty paragraph texts with newlines."""
    mock_doc = MagicMock()
    para1 = MagicMock(); para1.text = "First paragraph"
    para2 = MagicMock(); para2.text = ""
    para3 = MagicMock(); para3.text = "Third paragraph"
    mock_doc.paragraphs = [para1, para2, para3]

    with patch("Fixpaper.Document", return_value=mock_doc):
        result = parse_docx(b"fake docx bytes")

    assert result == "First paragraph\nThird paragraph"


def test_parse_pdf_empty_pages():
    """parse_pdf returns empty string when PDF has no pages."""
    mock_pdf = MagicMock()
    mock_pdf.__enter__ = lambda s: s
    mock_pdf.__exit__ = MagicMock(return_value=False)
    mock_pdf.pages = []

    with patch("Fixpaper.pdfplumber.open", return_value=mock_pdf):
        result = parse_pdf(b"fake pdf bytes")

    assert result == ""


def test_parse_docx_skips_whitespace_only_paragraphs():
    """parse_docx skips paragraphs that contain only whitespace."""
    mock_doc = MagicMock()
    para1 = MagicMock(); para1.text = "Real content"
    para2 = MagicMock(); para2.text = "   "
    mock_doc.paragraphs = [para1, para2]

    with patch("Fixpaper.Document", return_value=mock_doc):
        result = parse_docx(b"fake docx bytes")

    assert result == "Real content"
