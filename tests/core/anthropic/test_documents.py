"""Local conversion of Anthropic document blocks for Chat Completions models."""

import base64
import struct
import zlib

import pytest

from free_claude_code.core.anthropic import MessagesRequest
from free_claude_code.core.anthropic.conversion import (
    build_base_request_body,
)
from free_claude_code.core.anthropic.documents import (
    DocumentImage,
    DocumentText,
    document_parts,
    encode_png,
)
from free_claude_code.core.anthropic.tokens import get_token_count
from free_claude_code.core.anthropic.tool_results import (
    ToolResultImage,
    ToolResultText,
    decompose_tool_result_content,
)


def make_pdf(*page_texts: str) -> bytes:
    """A small valid PDF with one Helvetica text line per page."""
    count = len(page_texts)
    font_id = 3 + 2 * count
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids ["
        + b" ".join(f"{3 + 2 * i} 0 R".encode() for i in range(count))
        + f"] /Count {count} >>".encode(),
    ]
    for index, text in enumerate(page_texts):
        stream = f"BT /F1 24 Tf 72 700 Td ({text}) Tj ET".encode()
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
            f"/Contents {4 + 2 * index} 0 R >>".encode()
        )
        objects.append(
            f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream"
        )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n"
    ).encode()
    return bytes(out)


def pdf_block(raw: bytes, **extra) -> dict:
    return {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": base64.b64encode(raw).decode(),
        },
        **extra,
    }


def _png_size(url: str) -> tuple[int, int]:
    raw = base64.b64decode(url.split(",", 1)[1])
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", raw[16:24])


def test_pdf_becomes_page_text_and_page_images():
    pytest.importorskip("pypdfium2")
    parts = document_parts(pdf_block(make_pdf("Alpha page", "Beta page"), title="R"))
    texts = [p.text for p in parts if isinstance(p, DocumentText)]
    images = [p for p in parts if isinstance(p, DocumentImage)]
    joined = "\n".join(texts)
    assert texts[0] == "Document: R"
    assert "2 pages" in joined
    assert "--- Page 1 ---\nAlpha page" in joined
    assert "--- Page 2 ---\nBeta page" in joined
    assert len(images) == 2
    width, height = _png_size(images[0].url)
    assert max(width, height) == 1288
    assert images[0].tokens == -(-width // 32) * -(-height // 32)


def test_damaged_pdf_is_reported_not_raised():
    parts = document_parts(pdf_block(b"%PDF-1.4 not really"))
    assert len(parts) == 1
    assert isinstance(parts[0], DocumentText)
    assert "could not be opened" in parts[0].text or "not installed" in parts[0].text


@pytest.mark.parametrize(
    "source,expected",
    [
        ({"type": "text", "media_type": "text/plain", "data": "plain"}, "plain"),
        (
            {
                "type": "base64",
                "media_type": "text/markdown",
                "data": base64.b64encode(b"# md").decode(),
            },
            "# md",
        ),
        ({"type": "content", "content": [{"type": "text", "text": "c1"}]}, "c1"),
    ],
)
def test_text_documents_pass_through(source, expected):
    parts = document_parts({"type": "document", "source": source})
    assert parts == (DocumentText(expected),)


def test_url_and_bad_base64_documents_explain_themselves():
    (url_part,) = document_parts(
        {"type": "document", "source": {"type": "url", "url": "https://x/y.pdf"}}
    )
    assert isinstance(url_part, DocumentText)
    assert "https://x/y.pdf" in url_part.text
    (bad,) = document_parts(
        {
            "type": "document",
            "source": {"type": "base64", "media_type": "application/pdf", "data": "%%"},
        }
    )
    assert isinstance(bad, DocumentText)
    assert "base64" in bad.text


def test_user_document_reaches_chat_completions_as_text_and_images():
    pytest.importorskip("pypdfium2")
    request = MessagesRequest(
        model="m",
        max_tokens=64,
        messages=[
            {
                "role": "user",
                "content": [
                    pdf_block(make_pdf("Quarterly numbers")),
                    {"type": "text", "text": "Summarize it."},
                ],
            }
        ],
    )
    body = build_base_request_body(request)
    user = [m for m in body["messages"] if m["role"] == "user"][-1]
    content = user["content"]
    assert isinstance(content, list)
    kinds = [part["type"] for part in content]
    assert "image_url" in kinds
    text = "\n".join(p["text"] for p in content if p["type"] == "text")
    assert "Quarterly numbers" in text
    assert "Summarize it." in text


def test_tool_result_document_expands_to_text_and_images():
    pytest.importorskip("pypdfium2")
    decomposed = decompose_tool_result_content([pdf_block(make_pdf("From a tool"))])
    assert decomposed.has_documents and decomposed.has_images
    assert any(
        isinstance(p, ToolResultText) and "From a tool" in p.text
        for p in decomposed.parts
    )
    assert any(isinstance(p, ToolResultImage) for p in decomposed.parts)


def test_text_document_tool_result_is_not_json_serialized():
    request = MessagesRequest(
        model="m",
        max_tokens=64,
        messages=[
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "Read", "input": {}}
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [
                            {
                                "type": "document",
                                "source": {"type": "text", "data": "doc body"},
                            }
                        ],
                    }
                ],
            },
        ],
    )
    body = build_base_request_body(request)
    tool = next(m for m in body["messages"] if m["role"] == "tool")
    assert tool["content"] == "doc body"


def test_token_count_prices_pdf_as_converted_not_as_base64():
    pytest.importorskip("pypdfium2")
    raw = make_pdf(*(f"Page {i} text" for i in range(5)))
    request = MessagesRequest(
        model="m",
        max_tokens=64,
        messages=[{"role": "user", "content": [pdf_block(raw)]}],
    )
    tokens = get_token_count(request.messages)
    parts = document_parts(pdf_block(raw))
    image_tokens = sum(p.tokens for p in parts if isinstance(p, DocumentImage))
    assert image_tokens > 0
    assert image_tokens <= tokens < image_tokens + 500


def test_encode_png_round_trips_header():
    rows = b"\x00" + b"\xff\x00\x00" * 2 + b"\x00" + b"\x00\xff\x00" * 2
    png = encode_png(2, 2, 2, rows)
    assert struct.unpack(">II", png[16:24]) == (2, 2)
    idat_start = png.index(b"IDAT") + 4
    length = struct.unpack(">I", png[idat_start - 8 : idat_start - 4])[0]
    assert zlib.decompress(png[idat_start : idat_start + length]) == rows
