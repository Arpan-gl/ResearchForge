"""PDF parsing utilities: text extraction, table extraction, OCR fallback.

Best-effort: uses PyMuPDF (`fitz`) for text, Camelot for tables when available,
and falls back to OCR via `pdf2image` + `pytesseract` if needed.
"""
from typing import Tuple, List
import tempfile
import os
import io


def parse_pdf_bytes(content: bytes) -> Tuple[str, List[str], str | None]:
    """Parse PDF bytes and return (text, tables, error).

    - text: full extracted text (may be empty)
    - tables: list of CSV strings for extracted tables (may be empty)
    - error: None on success or error message
    """
    # Attempt PyMuPDF first
    text = ""
    tables: List[str] = []
    try:
        import fitz
    except Exception as e:
        return "", [], f"PyMuPDF not installed: {e}"

    try:
        doc = fitz.open(stream=content, filetype="pdf")
        pages_text = []
        for page in doc:
            try:
                pages_text.append(page.get_text("text"))
            except Exception:
                pages_text.append("")
        text = "\n".join(pages_text)
    except Exception as e:
        # If fitz fails, try OCR below
        text = ""

    # Try to extract tables using Camelot if available (requires file path)
    try:
        import camelot
        # write to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            tables_found = camelot.read_pdf(tmp_path, pages="1-end")
            for tbl in tables_found:
                try:
                    csv_text = tbl.df.to_csv(index=False)
                    tables.append(csv_text)
                except Exception:
                    continue
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
    except Exception:
        # Camelot not available — ignore
        pass

    # If no text found, attempt OCR as fallback
    if not text or len(text.strip()) < 200:
        try:
            from pdf2image import convert_from_bytes
            import pytesseract
            images = convert_from_bytes(content, dpi=200)
            ocr_pages = []
            for img in images:
                try:
                    ocr_pages.append(pytesseract.image_to_string(img))
                except Exception:
                    ocr_pages.append("")
            ocr_text = "\n".join(ocr_pages)
            if ocr_text.strip():
                # prefer OCR text if it's substantially longer
                if len(ocr_text) > len(text):
                    text = ocr_text
        except Exception:
            # OCR tools not available — that's fine
            pass

    return text, tables, None
