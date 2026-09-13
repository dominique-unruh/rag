"""The ``xlsx`` translator (PLAN §5.4).

Converts an Excel OOXML workbook (``.xlsx``/``.xltx``) to Markdown: one section
per worksheet, each rendered as a Markdown table of its cells. Backed by the
``openpyxl`` library.
"""

from __future__ import annotations

import io

import openpyxl

from ..base import Document, Translation, Translator, make_translation


def _cell(value: object) -> str:
    """Render one cell value as Markdown-table-safe text.

    ``None`` (empty cell) becomes an empty string; pipes are escaped (they are the
    table delimiter) and any newlines are flattened so one spreadsheet row stays
    on one Markdown line.
    """
    if value is None:
        return ""
    text = str(value).replace("|", "\\|")
    return text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")


def _sheet_to_markdown(title: str, rows: list[list[str]]) -> str | None:
    """Render one worksheet's rows as a Markdown table, or ``None`` if empty.

    Trailing all-empty rows and columns are dropped; every remaining row is padded
    to a common width. The first row is used as the table header (Markdown tables
    require a header + separator, even when the sheet has no real header row).
    """
    # Drop trailing all-empty rows.
    while rows and all(cell == "" for cell in rows[-1]):
        rows.pop()
    if not rows:
        return None
    width = max(len(row) for row in rows)
    # Drop trailing all-empty columns (keep at least one).
    while width > 1 and all(len(row) < width or row[width - 1] == "" for row in rows):
        width -= 1
    # Normalise every row to the common width.
    norm = [(row + [""] * width)[:width] for row in rows]
    lines = [
        f"## {title}",
        "| " + " | ".join(norm[0]) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in norm[1:]]
    return "\n".join(lines)


class Xlsx(Translator):
    """Converts an Excel OOXML workbook to Markdown tables."""

    def translate(self, document: Document) -> list[Translation]:
        """Return the workbook as one Markdown translation (a table per sheet).

        Cached cell values are read (``data_only``) — formulas are not evaluated,
        so a workbook last saved by a non-Excel writer may expose ``None`` for
        formula cells. Empty sheets are skipped; sheets are separated by blank
        lines.
        """
        # read_only streams large sheets without loading them wholesale; data_only
        # returns the last cached value rather than the formula string.
        workbook = openpyxl.load_workbook(
            io.BytesIO(document.content), read_only=True, data_only=True
        )
        try:
            sections: list[str] = []
            for worksheet in workbook.worksheets:
                rows = [
                    [_cell(value) for value in row]
                    for row in worksheet.iter_rows(values_only=True)
                ]
                section = _sheet_to_markdown(worksheet.title, rows)
                if section is not None:
                    sections.append(section)
        finally:
            # read_only mode holds the underlying zip open until explicitly closed.
            workbook.close()
        return [make_translation(document, "\n\n".join(sections))]


# Module-level instance selected by --translator xlsx.
translator: Xlsx = Xlsx()
