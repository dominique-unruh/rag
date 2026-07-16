"""The ``email`` translator (PLAN §5.4).

Splits an RFC 5322 message into its body (main text, no attachments) and its
attachments. The body is rendered as Markdown with the important decoded headers
prepended; each attachment becomes a synthetic document routed back through the
``dispatch`` translator (so attachments of any supported type — including nested
emails and PDFs — are handled uniformly).
"""

from __future__ import annotations

import email
import email.policy
from email.message import EmailMessage

from ..base import Document, Translation, Translator, make_translation
from ..markdown import html_to_markdown

# Header fields prepended (decoded) to the body, in this order.
_HEADER_FIELDS: tuple[str, ...] = ("Subject", "From", "To", "Cc", "Date")


def _header_block(msg: EmailMessage) -> str:
    """Return a Markdown block of the important, decoded headers (present ones)."""
    lines: list[str] = []
    for name in _HEADER_FIELDS:
        value = msg[name]
        if value is not None:
            # str() of a policy.default header yields the decoded Unicode value.
            lines.append(f"**{name}:** {str(value)}")
    return "\n".join(lines)


def _body_markdown(msg: EmailMessage) -> str:
    """Return the message body as Markdown (HTML converted, plain text kept)."""
    # Prefer a plain-text body, falling back to HTML.
    body = msg.get_body(preferencelist=("plain", "html"))
    if body is None:
        return ""
    content = body.get_content()  # decoded str for text/* parts
    if body.get_content_type() == "text/html":
        return html_to_markdown(content)
    return content.strip()


def _attachment_bytes(part: EmailMessage) -> bytes | None:
    """Return an attachment's raw bytes, or ``None`` if it cannot be extracted."""
    payload = part.get_payload(decode=True)
    if payload is not None:
        return payload
    # message/rfc822 attachments have no decoded payload — serialise the message.
    try:
        return part.get_content().as_bytes()
    except (AttributeError, KeyError, ValueError):
        return None


class Email(Translator):
    """Translates an email into a body part plus one part per attachment."""

    def translate(self, document: Document) -> list[Translation]:
        """Return the email body translation plus all attachment translations."""
        # Parse with the modern policy so headers/bodies come back decoded.
        msg = email.message_from_bytes(document.content, policy=email.policy.default)

        # --- body part ---------------------------------------------------
        header_block = _header_block(msg)
        body = _body_markdown(msg)
        text = f"{header_block}\n\n{body}" if header_block else body
        results: list[Translation] = [
            make_translation(document, text, sub_doc_id=[*document.sub_doc_id, "body"])
        ]

        # --- attachment parts (routed through dispatch) ------------------
        from .dispatch import translator as dispatch  # lazy: avoids import cycle

        for index, part in enumerate(msg.iter_attachments(), start=1):
            raw = _attachment_bytes(part)
            if raw is None:
                # Could not extract this attachment; skip it.
                continue
            sub_doc_id = [*document.sub_doc_id, f"attachment:{index}"]
            synthetic = Document(
                doc_id=document.doc_id,
                sub_doc_id=sub_doc_id,
                content=raw,
                stable_metadata={
                    "mime-type": part.get_content_type(),
                    "filename": part.get_filename() or "",
                },
            )
            results.extend(dispatch.translate(synthetic))
        return results


# Module-level instance selected by --translator email.
translator: Email = Email()
