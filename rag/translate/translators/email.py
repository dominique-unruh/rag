"""The ``email`` translator (PLAN §5.4).

Splits an RFC 5322 message into its body (main text, no attachments) and its
attachments. The body is rendered as Markdown with the important decoded headers
prepended; each attachment becomes a synthetic document routed back through the
``dispatch`` translator (so attachments of any supported type — including nested
emails and PDFs — are handled uniformly).
"""

from __future__ import annotations

import codecs
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


def _is_known_charset(name: str) -> bool:
    """Return whether ``name`` is a charset Python can actually decode with."""
    try:
        codecs.lookup(name)
        return True
    except LookupError:
        return False


def _decode_text_part(part: EmailMessage) -> str:
    """Decode a text/* part to ``str``, tolerating an unusable declared charset.

    ``get_content`` reads the charset from the part's Content-Type header and
    raises ``LookupError`` when that charset is not a registered Python codec
    (real senders emit junk like ``charset="unicode"``), which would otherwise
    abort the whole run. ``get_content`` takes no charset override, so in that
    one case we rewrite the header's charset to UTF-8 and let ``get_content`` do
    everything else — it already undoes the transfer-encoding and replaces
    undecodable bytes by default. Any *other* failure still surfaces.
    """
    charset = part.get_content_charset()  # declared charset, or None if absent
    if charset is not None and not _is_known_charset(charset):
        # Overwrite the unusable charset in place; the message is parsed fresh
        # per document and then discarded, so mutating it here is safe.
        part.set_param("charset", "utf-8", header="Content-Type")
    return part.get_content()  # transfer-decode + charset-decode (errors=replace)


def _body_markdown(msg: EmailMessage) -> str:
    """Return the message body as Markdown (HTML converted, plain text kept)."""
    # Prefer a plain-text body, falling back to HTML.
    body = msg.get_body(preferencelist=("plain", "html"))
    if body is None:
        return ""
    content = _decode_text_part(body)
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
