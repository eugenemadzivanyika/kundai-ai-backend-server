"""Unit tests for pdf_parser_service — file I/O and OCR fallback mocked."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestExtractTextFromPdf:
    @pytest.mark.asyncio
    async def test_returns_text_from_pdf(self, monkeypatch):
        monkeypatch.setattr(
            "services.pdf_parser_service._extract_sync",
            MagicMock(return_value="Some readable text " * 50),
        )
        from services.pdf_parser_service import extract_text_from_pdf
        result = await extract_text_from_pdf("fake.pdf")
        assert "readable text" in result

    @pytest.mark.asyncio
    async def test_falls_back_to_ocr_when_text_too_short(self, monkeypatch):
        monkeypatch.setattr(
            "services.pdf_parser_service._extract_sync",
            MagicMock(return_value="tiny"),  # below _MIN_TEXT_CHARS threshold
        )
        ocr_mock = AsyncMock(return_value="OCR extracted content for a scanned PDF page.")
        monkeypatch.setattr("services.pdf_parser_service._ocr_fallback", ocr_mock)

        from services.pdf_parser_service import extract_text_from_pdf
        result = await extract_text_from_pdf("scanned.pdf")
        ocr_mock.assert_awaited_once_with("scanned.pdf")
        assert "OCR extracted" in result

    @pytest.mark.asyncio
    async def test_does_not_call_ocr_for_text_rich_pdf(self, monkeypatch):
        monkeypatch.setattr(
            "services.pdf_parser_service._extract_sync",
            MagicMock(return_value="Full textbook content. " * 20),
        )
        ocr_mock = AsyncMock(return_value="should not be called")
        monkeypatch.setattr("services.pdf_parser_service._ocr_fallback", ocr_mock)

        from services.pdf_parser_service import extract_text_from_pdf
        await extract_text_from_pdf("text.pdf")
        ocr_mock.assert_not_awaited()


class TestExtractSync:
    def test_uses_fitz_when_available(self, monkeypatch):
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Page 1 text"
        mock_doc = MagicMock()
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_fitz = MagicMock()
        mock_fitz.open.return_value = mock_doc

        import sys
        monkeypatch.setitem(sys.modules, "fitz", mock_fitz)

        from services import pdf_parser_service
        import importlib
        importlib.reload(pdf_parser_service)  # pick up the patched fitz

        # _extract_sync should call fitz.open
        mock_fitz.open.return_value = mock_doc
        result = pdf_parser_service._extract_sync("test.pdf")
        mock_fitz.open.assert_called_once_with("test.pdf")
