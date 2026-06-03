# TODO: VIBECODED BULLSHIT, TAKE NOTHING FOR GRANTED
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
from html import escape
import math
import hashlib


@dataclass(frozen=True)
class Field:
    start: int
    size: int
    name: str
    color: Optional[str] = None
    collapsible: bool = False
    note: str = ""

    @property
    def end(self) -> int:
        return self.start + self.size

    def intersects(self, start: int, end: int) -> bool:
        return self.start < end and start < self.end


@dataclass(frozen=True)
class _CollapseMarker:
    row: int
    hidden_rows: int
    hidden_bytes: int
    field: Field


class HexDumpSvg:
    """
    Generate an SVG hexdump-style memory layout.

    Features:
      - Address column.
      - Hex column with configurable grouping.
      - Optional ASCII column.
      - Named fields with colored byte ranges.
      - Annotation column per row.
      - Collapsible large fields.
      - Pure stdlib SVG output.
    """

    def __init__(
        self,
        data: bytes | bytearray | memoryview,
        fields: Iterable[Field] = (),
        *,
        base_address: int = 0,
        bytes_per_row: int = 16,
        group_bytes: int = 1,
        address_digits: int = 8,
        show_ascii: bool = True,
        title: str = "Memory layout",
        collapse: bool = True,
        collapse_min_rows: int = 5,
        collapse_keep_edge_rows: int = 1,
        font_size: int = 13,
    ):
        self.data = bytes(data)
        self.fields = list(fields)

        self.base_address = base_address
        self.bytes_per_row = bytes_per_row
        self.group_bytes = group_bytes
        self.address_digits = address_digits
        self.show_ascii = show_ascii
        self.title = title

        self.collapse = collapse
        self.collapse_min_rows = collapse_min_rows
        self.collapse_keep_edge_rows = collapse_keep_edge_rows

        self.font_size = font_size

        if self.bytes_per_row <= 0:
            raise ValueError("bytes_per_row must be positive")
        if self.group_bytes <= 0:
            raise ValueError("group_bytes must be positive")
        if self.bytes_per_row % self.group_bytes != 0:
            raise ValueError("bytes_per_row should be divisible by group_bytes")

        for f in self.fields:
            if f.start < 0 or f.size < 0:
                raise ValueError(f"Invalid field range: {f!r}")

        self._field_colors = {
            f: f.color or self._auto_color(f.name)
            for f in self.fields
        }
        self._byte_owner = self._build_byte_owner()

    def write_svg(self, path: str | Path) -> None:
        Path(path).write_text(self.to_svg(), encoding="utf-8")

    def to_svg(self) -> str:
        visible_rows, markers = self._visible_rows_and_markers()
        marker_by_row = {m.row: m for m in markers}

        char_w = self.font_size * 0.62
        line_h = int(self.font_size * 1.85)

        margin = 18
        title_h = 30
        header_h = line_h
        legend_h = self._legend_height(line_h)

        addr_w = int(char_w * (self.address_digits + 2))

        group_count = self.bytes_per_row // self.group_bytes
        group_w = int(char_w * (self.group_bytes * 2) + 12)
        group_gap = 7

        hex_w = group_count * group_w + (group_count - 1) * group_gap
        ascii_w = int(char_w * self.bytes_per_row + 20) if self.show_ascii else 0
        anno_w = 380

        body_lines = len(visible_rows) + len(markers)

        width = (
            margin * 2
            + addr_w
            + 14
            + hex_w
            + 16
            + ascii_w
            + 16
            + anno_w
        )
        height = margin * 2 + title_h + header_h + body_lines * line_h + legend_h

        parts: list[str] = []
        add = parts.append

        add(
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">'
        )

        add("<style>")
        add(f"""
            .bg {{
                fill: #ffffff;
            }}
            .title {{
                font: 700 {self.font_size + 3}px ui-monospace, SFMono-Regular,
                      Menlo, Consolas, monospace;
                fill: #111827;
            }}
            .mono {{
                font: {self.font_size}px ui-monospace, SFMono-Regular,
                      Menlo, Consolas, monospace;
                fill: #111827;
            }}
            .muted {{
                font: {self.font_size}px ui-monospace, SFMono-Regular,
                      Menlo, Consolas, monospace;
                fill: #6b7280;
            }}
            .small {{
                font: {self.font_size - 1}px ui-monospace, SFMono-Regular,
                      Menlo, Consolas, monospace;
                fill: #374151;
            }}
            .header {{
                font: 700 {self.font_size}px ui-monospace, SFMono-Regular,
                      Menlo, Consolas, monospace;
                fill: #374151;
            }}
            .byteRect {{
                stroke: #e5e7eb;
                stroke-width: 0.7;
                rx: 2;
                ry: 2;
            }}
            .marker {{
                fill: #f9fafb;
                stroke: #d1d5db;
                stroke-dasharray: 4 3;
                stroke-width: 1;
                rx: 5;
                ry: 5;
            }}
        """)
        add("</style>")

        add(f'<rect class="bg" x="0" y="0" width="{width}" height="{height}"/>')

        y = margin
        add(
            f'<text class="title" x="{margin}" y="{y + self.font_size}">'
            f'{escape(self.title)}</text>'
        )
        y += title_h

        x_addr = margin
        x_hex = x_addr + addr_w + 14
        x_ascii = x_hex + hex_w + 16
        x_anno = x_ascii + ascii_w + 16

        add(f'<text class="header" x="{x_addr}" y="{y}">offset</text>')
        add(f'<text class="header" x="{x_hex}" y="{y}">hex</text>')
        if self.show_ascii:
            add(f'<text class="header" x="{x_ascii}" y="{y}">ascii</text>')
        add(f'<text class="header" x="{x_anno}" y="{y}">field</text>')

        y += header_h

        for row in range(self._row_count()):
            marker = marker_by_row.get(row)
            if marker is not None:
                self._render_marker(
                    add,
                    marker,
                    x_addr,
                    y,
                    width - margin * 2,
                    line_h,
                )
                y += line_h

            if row not in visible_rows:
                continue

            self._render_row(
                add=add,
                row=row,
                x_addr=x_addr,
                x_hex=x_hex,
                x_ascii=x_ascii,
                x_anno=x_anno,
                y=y,
                line_h=line_h,
                group_w=group_w,
                group_gap=group_gap,
            )
            y += line_h

        self._render_legend(add, margin, y + 8, line_h)

        add("</svg>")
        return "\n".join(parts)

    def _render_row(
        self,
        *,
        add,
        row: int,
        x_addr: int,
        x_hex: int,
        x_ascii: int,
        x_anno: int,
        y: int,
        line_h: int,
        group_w: int,
        group_gap: int,
    ) -> None:
        row_start = row * self.bytes_per_row
        row_end = min(row_start + self.bytes_per_row, len(self.data))

        addr = self.base_address + row_start
        baseline = y + int(line_h * 0.67)

        add(
            f'<text class="mono" x="{x_addr}" y="{baseline}">'
            f'{addr:0{self.address_digits}x}</text>'
        )

        group_count = self.bytes_per_row // self.group_bytes

        for g in range(group_count):
            b0 = row_start + g * self.group_bytes
            b1 = min(b0 + self.group_bytes, row_end)

            gx = x_hex + g * (group_w + group_gap)
            sub_w = group_w / self.group_bytes

            # Draw per-byte backgrounds even inside grouped words.
            for bi in range(b0, min(b0 + self.group_bytes, len(self.data))):
                owner = self._byte_owner[bi]
                color = self._field_colors.get(owner, "#ffffff") if owner else "#ffffff"
                sx = gx + (bi - b0) * sub_w

                add(
                    f'<rect class="byteRect" '
                    f'x="{sx:.2f}" y="{y + 2}" '
                    f'width="{sub_w:.2f}" height="{line_h - 5}" '
                    f'fill="{color}"/>'
                )

            if b0 < row_end:
                raw = self.data[b0:b1]
                hex_text = raw.hex().ljust(self.group_bytes * 2, " ")

                add(
                    f'<text class="mono" x="{gx + 6}" y="{baseline}">'
                    f'{escape(hex_text)}</text>'
                )

        if self.show_ascii:
            chars = []
            for b in self.data[row_start:row_end]:
                chars.append(chr(b) if 32 <= b <= 126 else ".")

            ascii_text = "".join(chars).ljust(self.bytes_per_row)

            add(
                f'<text class="mono" x="{x_ascii}" y="{baseline}">'
                f'{escape(ascii_text)}</text>'
            )

        annotation = self._row_annotation(row_start, row_end)

        add(
            f'<text class="small" x="{x_anno}" y="{baseline}">'
            f'{escape(annotation)}</text>'
        )

    def _render_marker(
        self,
        add,
        marker: _CollapseMarker,
        x: int,
        y: int,
        w: int,
        line_h: int,
    ) -> None:
        baseline = y + int(line_h * 0.67)

        label = (
            f"⋮ {marker.hidden_rows} rows / "
            f"{marker.hidden_bytes} bytes omitted from {marker.field.name}"
        )

        add(
            f'<rect class="marker" x="{x}" y="{y + 2}" '
            f'width="{w}" height="{line_h - 5}"/>'
        )
        add(
            f'<text class="muted" x="{x + 10}" y="{baseline}">'
            f'{escape(label)}</text>'
        )

    def _render_legend(self, add, x: int, y: int, line_h: int) -> None:
        if not self.fields:
            return

        add(f'<text class="header" x="{x}" y="{y + line_h}">legend</text>')

        y += line_h + 8
        col_w = 380

        for i, f in enumerate(self.fields):
            col = i % 2
            row = i // 2

            lx = x + col * col_w
            ly = y + row * line_h

            color = self._field_colors[f]

            if f.size == 0:
                label = f"{f.start:#x} empty  {f.name}"
            else:
                label = f"{f.start:#x}..{f.end - 1:#x}  {f.name}"

            if f.note:
                label += f" — {f.note}"

            add(
                f'<rect x="{lx}" y="{ly + 4}" width="14" height="14" '
                f'fill="{color}" stroke="#d1d5db"/>'
            )
            add(
                f'<text class="small" x="{lx + 22}" y="{ly + 16}">'
                f'{escape(label)}</text>'
            )

    def _legend_height(self, line_h: int) -> int:
        if not self.fields:
            return 0

        return line_h * (2 + math.ceil(len(self.fields) / 2)) + 12

    def _row_annotation(self, row_start: int, row_end: int) -> str:
        hits = [f for f in self.fields if f.intersects(row_start, row_end)]

        if not hits:
            return ""

        pieces = []

        for f in sorted(hits, key=lambda field: (field.start, field.size)):
            a = max(f.start, row_start)
            b = min(f.end, row_end)

            prefix = "…" if f.start < row_start else ""
            suffix = "…" if row_end < f.end else ""

            pieces.append(f"{prefix}{f.name}{suffix} [{a:#x}..{b - 1:#x}]")

        return "; ".join(pieces)

    def _visible_rows_and_markers(self) -> tuple[set[int], list[_CollapseMarker]]:
        row_count = self._row_count()
        visible = set(range(row_count))
        markers: list[_CollapseMarker] = []

        if not self.collapse:
            return visible, markers

        for f in self.fields:
            if not f.collapsible or f.size <= 0:
                continue

            # Only collapse rows fully inside the field.
            first_full_row = math.ceil(f.start / self.bytes_per_row)
            last_full_row = math.floor(f.end / self.bytes_per_row) - 1

            full_rows = last_full_row - first_full_row + 1

            if full_rows < self.collapse_min_rows:
                continue

            hide_start = first_full_row + self.collapse_keep_edge_rows
            hide_end = last_full_row - self.collapse_keep_edge_rows

            if hide_start > hide_end:
                continue

            hidden_rows = hide_end - hide_start + 1

            for r in range(hide_start, hide_end + 1):
                visible.discard(r)

            markers.append(
                _CollapseMarker(
                    row=hide_start,
                    hidden_rows=hidden_rows,
                    hidden_bytes=hidden_rows * self.bytes_per_row,
                    field=f,
                )
            )

        markers.sort(key=lambda marker: marker.row)
        return visible, markers

    def _row_count(self) -> int:
        return max(1, math.ceil(len(self.data) / self.bytes_per_row))

    def _build_byte_owner(self) -> list[Optional[Field]]:
        """
        Build byte -> field mapping.

        Larger wrapper fields are applied first.
        Smaller, more specific fields override them.
        """
        owner: list[Optional[Field]] = [None] * len(self.data)

        for f in sorted(self.fields, key=lambda field: field.size, reverse=True):
            lo = max(0, f.start)
            hi = min(len(self.data), f.end)

            for i in range(lo, hi):
                owner[i] = f

        return owner

    @staticmethod
    def _auto_color(name: str) -> str:
        palette = [
            "#dbeafe",
            "#dcfce7",
            "#fef3c7",
            "#fae8ff",
            "#fee2e2",
            "#e0e7ff",
            "#ccfbf1",
            "#ffedd5",
            "#fce7f3",
            "#ecfccb",
            "#e5e7eb",
            "#cffafe",
            "#ede9fe",
            "#fef9c3",
            "#d1fae5",
        ]

        h = hashlib.sha1(name.encode("utf-8")).digest()[0]
        return palette[h % len(palette)]


if __name__ == "__main__":
    # Example matching the rough shape of SharedMemoryPosix.
    #
    # Replace these values with the real constants from shared_memory_impl.h.
    c_contiguous_segment_count = 8
    id_size = 4

    partial_table_entries = c_contiguous_segment_count + 1
    partial_table_size = partial_table_entries * id_size * 2

    # Synthetic example bytes.
    magic = bytes.fromhex("55424152")  # Example only: "UBAR"
    version = (1).to_bytes(4, "big")

    payload_offset = 8 + partial_table_size

    table = bytearray()

    for segment_id in range(c_contiguous_segment_count):
        table += segment_id.to_bytes(4, "big")

        offset = payload_offset if segment_id == 0 else 0
        table += offset.to_bytes(4, "big")

    # Example c_partial_table_link_id.
    table += (0xFFFFFFFF).to_bytes(4, "big")
    table += (0).to_bytes(4, "big")

    payload = bytes(range(256)) * 2

    blob = magic + version + bytes(table) + payload

    fields = [
        Field(
            0,
            4,
            "c_UnBARableAI_magic",
            note="written first",
        ),
        Field(
            4,
            4,
            "c_version",
            note="big endian",
        ),
        Field(
            8,
            partial_table_size,
            "partial segment table #0",
        ),
        Field(
            8,
            8,
            "entry id=0 → payload offset",
        ),
        Field(
            8 + c_contiguous_segment_count * 8,
            8,
            "partial-table link entry",
        ),
        Field(
            payload_offset,
            len(payload),
            "segment payload bytes",
            collapsible=True,
        ),
    ]

    HexDumpSvg(
        blob,
        fields,
        title="SharedMemoryPosix layout — example",
        bytes_per_row=16,
        group_bytes=4,
        address_digits=8,
        collapse=True,
        collapse_min_rows=6,
        collapse_keep_edge_rows=1,
    ).write_svg("shared_memory_layout.svg")