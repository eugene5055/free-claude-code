"""Local conversion of Anthropic ``document`` blocks for non-Anthropic models.

Anthropic reads a PDF as both its text and an image of each page. Models served
over Chat Completions (llama.cpp, other OpenAI-compatible servers) have no
document input, so FCC does the same locally: each page's text layer is
extracted and the first pages are rendered to PNG images (pypdfium2, bundled
PDFium; no system tools and no network). Plain-text and content-block documents
become text and images directly.

Results are cached by content hash so token counting and conversion of the same
request decode a PDF once.
"""

import base64
import binascii
import hashlib
import math
import struct
import threading
import zlib
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from loguru import logger

from free_claude_code.core.token_estimation import estimate_image_tokens

from .content import get_block_attr, get_block_type
from .image_sources import AnthropicImageSourceError, portable_anthropic_image_url

# Pages whose text is included, and how many of them are also rendered as images.
# A rendered page costs about (side / 32)^2 vision tokens on Qwen-class models,
# so 20 letter pages at 1288 px long side are ~25K tokens; every page's text is
# still sent.
PDF_MAX_TEXT_PAGES = 100
PDF_MAX_IMAGE_PAGES = 20
PDF_RENDER_LONG_SIDE = 1288
_VISION_PATCH = 32
_CACHE_ENTRIES = 8


@dataclass(frozen=True, slots=True)
class DocumentText:
    text: str


@dataclass(frozen=True, slots=True)
class DocumentImage:
    url: str
    tokens: int


DocumentPart = DocumentText | DocumentImage

_cache: OrderedDict[str, tuple[DocumentPart, ...]] = OrderedDict()
_cache_lock = threading.Lock()


def document_parts(block: object) -> tuple[DocumentPart, ...]:
    """Return ordered text and image parts that carry one document block."""
    source = get_block_attr(block, "source", {})
    header = _header(block)
    try:
        body = _source_parts(source)
    except AnthropicImageSourceError as exc:
        body = (DocumentText(f"[The document could not be read: {exc}]"),)
    if header:
        return (DocumentText(header), *body)
    return body


def document_text_and_image_tokens(block: object) -> tuple[str, int]:
    """Text a document converts to, and its images' estimated tokens."""
    parts = document_parts(block)
    text = "\n".join(part.text for part in parts if isinstance(part, DocumentText))
    images = sum(part.tokens for part in parts if isinstance(part, DocumentImage))
    return text, images


def _header(block: object) -> str:
    title = get_block_attr(block, "title")
    context = get_block_attr(block, "context")
    lines = []
    if isinstance(title, str) and title.strip():
        lines.append(f"Document: {title.strip()}")
    if isinstance(context, str) and context.strip():
        lines.append(context.strip())
    return "\n".join(lines)


def _field(source: object, name: str) -> Any:
    if isinstance(source, Mapping):
        return source.get(name)
    return getattr(source, name, None)


def _source_parts(source: object) -> tuple[DocumentPart, ...]:
    source_type = _field(source, "type")
    if source_type == "text":
        data = _field(source, "data")
        return (DocumentText(data if isinstance(data, str) else str(data or "")),)
    if source_type == "content":
        return _content_parts(_field(source, "content"))
    if source_type == "base64":
        return _base64_parts(source)
    if source_type == "url":
        url = _field(source, "url")
        return (
            DocumentText(
                f"[A document was attached by URL ({url}); it was not downloaded. "
                "Fetch it with a tool to read it.]"
            ),
        )
    label = source_type if isinstance(source_type, str) else "unknown"
    return (DocumentText(f"[Document source {label!r} is not available locally.]"),)


def _base64_parts(source: object) -> tuple[DocumentPart, ...]:
    media_type = str(_field(source, "media_type") or "").lower()
    data = _field(source, "data")
    if not isinstance(data, str) or not data.strip():
        return (DocumentText("[The document is empty.]"),)
    try:
        raw = base64.b64decode("".join(data.split()), validate=True)
    except binascii.Error, ValueError:
        return (DocumentText("[The document is not valid base64.]"),)
    if media_type == "application/pdf":
        return pdf_parts(raw)
    if media_type.startswith("text/") or media_type in {
        "application/json",
        "application/xml",
    }:
        return (DocumentText(raw.decode("utf-8", errors="replace")),)
    return (DocumentText(f"[Unsupported document type {media_type!r}.]"),)


def _content_parts(content: object) -> tuple[DocumentPart, ...]:
    if isinstance(content, str):
        return (DocumentText(content),)
    if not isinstance(content, list):
        return ()
    parts: list[DocumentPart] = []
    for block in content:
        kind = get_block_type(block)
        if kind == "text":
            parts.append(DocumentText(str(get_block_attr(block, "text", ""))))
        elif kind == "image":
            url = portable_anthropic_image_url(get_block_attr(block, "source", {}))
            tokens = (
                estimate_image_tokens(len(url))
                if url.startswith("data:")
                else estimate_image_tokens()
            )
            parts.append(DocumentImage(url, tokens))
    return tuple(parts)


def pdf_parts(raw: bytes) -> tuple[DocumentPart, ...]:
    """Text of each page and PNG renders of the first pages, cached by content."""
    key = hashlib.sha256(raw).hexdigest()
    with _cache_lock:
        cached = _cache.get(key)
        if cached is not None:
            _cache.move_to_end(key)
            return cached
    parts = _convert_pdf(raw)
    with _cache_lock:
        _cache[key] = parts
        while len(_cache) > _CACHE_ENTRIES:
            _cache.popitem(last=False)
    return parts


def _convert_pdf(raw: bytes) -> tuple[DocumentPart, ...]:
    try:
        import pypdfium2 as pdfium
    except ImportError:
        return (DocumentText("[PDF support is not installed (pypdfium2).]"),)
    try:
        pdf = pdfium.PdfDocument(raw)
    except Exception as exc:
        logger.debug("PDF open failed: {}", exc)
        return (
            DocumentText(
                "[The PDF could not be opened; it may be damaged or encrypted.]"
            ),
        )
    try:
        return _pdf_pages(pdf)
    finally:
        pdf.close()


def _pdf_pages(pdf: Any) -> tuple[DocumentPart, ...]:
    pages = len(pdf)
    shown = min(pages, PDF_MAX_IMAGE_PAGES)
    intro = f"[PDF, {pages} page{'' if pages == 1 else 's'}."
    if pages:
        intro += f" Each page's text follows; pages 1-{shown} are also shown as images."
    parts: list[DocumentPart] = [DocumentText(intro + "]")]
    for index in range(min(pages, PDF_MAX_TEXT_PAGES)):
        page = pdf[index]
        try:
            text = _page_text(page)
            label = f"--- Page {index + 1} ---"
            parts.append(
                DocumentText(
                    f"{label}\n{text}" if text else f"{label}\n(no text layer)"
                )
            )
            if index < PDF_MAX_IMAGE_PAGES:
                image = _render_page(page)
                if image is not None:
                    parts.append(image)
        finally:
            page.close()
    if pages > PDF_MAX_TEXT_PAGES:
        parts.append(
            DocumentText(
                f"[Pages {PDF_MAX_TEXT_PAGES + 1}-{pages} were omitted to fit the context window.]"
            )
        )
    return tuple(parts)


def _page_text(page: Any) -> str:
    textpage = page.get_textpage()
    try:
        return textpage.get_text_bounded().replace("\r\n", "\n").strip()
    finally:
        textpage.close()


def _render_page(page: Any) -> DocumentImage | None:
    try:
        width, height = page.get_size()
        scale = PDF_RENDER_LONG_SIDE / max(width, height, 1)
        bitmap = page.render(scale=scale, rev_byteorder=True)
    except Exception as exc:
        logger.debug("PDF page render failed: {}", exc)
        return None
    try:
        png = _png_from_bitmap(bitmap)
        tokens = math.ceil(bitmap.width / _VISION_PATCH) * math.ceil(
            bitmap.height / _VISION_PATCH
        )
    finally:
        bitmap.close()
    if png is None:
        return None
    return DocumentImage(
        "data:image/png;base64," + base64.b64encode(png).decode("ascii"), tokens
    )


def _png_from_bitmap(bitmap: Any) -> bytes | None:
    width = int(bitmap.width)
    height = int(bitmap.height)
    stride = int(bitmap.stride)
    channels = int(bitmap.n_channels)
    mode = str(bitmap.mode)
    buffer = memoryview(bytes(bitmap.buffer))
    if channels == 1:
        color_type = 0
    elif channels in {3, 4} and mode.startswith("RGB"):
        color_type = 2
    else:
        return None
    rows = bytearray()
    for y in range(height):
        row = buffer[y * stride : y * stride + width * channels]
        rows.append(0)
        if channels == 4:
            pixels = bytearray(row)
            del pixels[3::4]
            rows.extend(pixels)
        else:
            rows.extend(row)
    return encode_png(width, height, color_type, bytes(rows))


def encode_png(width: int, height: int, color_type: int, filtered_rows: bytes) -> bytes:
    """Minimal PNG writer: 8-bit gray (0) or RGB (2) rows, each prefixed by filter 0."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(filtered_rows, 6))
        + chunk(b"IEND", b"")
    )
