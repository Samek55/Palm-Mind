import io
import logging
import re
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.core.exceptions import EmptyDocumentError, UnsupportedFileError

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def extract_text(filename: str, data: bytes) -> str:
    """Extract plain text from an uploaded .pdf or .txt file."""
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileError(f"Unsupported file type '{ext}'. Only .pdf and .txt are allowed.")

    text = _extract_pdf(data) if ext == ".pdf" else _decode_txt(data)
    text = _normalize(text)
    if not text:
        raise EmptyDocumentError(
            "No extractable text found in the file (scanned PDFs without OCR are not supported)."
        )
    return text


def _extract_pdf(data: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(data))
    except PdfReadError as exc:
        raise UnsupportedFileError(f"Could not read PDF: {exc}") from exc

    pages: list[str] = []
    for i, page in enumerate(reader.pages):
        try:
            pages.append(page.extract_text() or "")
        except Exception:  # pypdf can fail on individual malformed pages
            logger.warning("Skipping unreadable page %d", i)
    return "\n\n".join(pages)


def _decode_txt(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    # latin-1 never fails, so we shouldn't get here
    return data.decode("utf-8", errors="ignore")


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
