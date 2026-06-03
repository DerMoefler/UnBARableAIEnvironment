# VIBECODED VISUALIZATION THINGY, see memory_layout_posix.py for usage
from __future__ import annotations

import argparse
import hashlib
import math
import pathlib
from dataclasses import dataclass
from html import escape
from typing import Iterable, Optional


@dataclass(frozen=True)
class Region:
    """
    Larger conceptual memory region.

    Example:
        partial segment table #0
        segment payload bytes
    """

    start: int
    size: int
    name: str
    color: Optional[str] = None
    note: str = ""
    collapsible: bool = False

    @property
    def end(self) -> int:
        return self.start + self.size

    def intersects(self, start: int, end: int) -> bool:
        return self.start < end and start < self.end


@dataclass(frozen=True)
class Field:
    """
    Concrete non-overlapping field shown in the right-hand field table.

    group:
        Name of the owning Region. Used for color grouping and legend grouping.

    role:
        Repeated logical field kind inside a group.
        For example:
            role="id"
            role="offset"

        All fields with the same group + role get the same color.
    """

    start: int
    size: int
    name: str
    group: Optional[str] = None
    role: Optional[str] = None
    color: Optional[str] = None
    collapsible: bool = False

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
    label: str


class HexLayoutSvg:
    """
    Two-table SVG memory-layout visualizer.

    Left table:
        actual byte values

    Right table:
        concrete field map with the same byte grid

    Below:
        larger conceptual regions, such as structs/tables/payload areas

    Important:
        The data table renders one text object per byte. This avoids text
        alignment drift from grouped multi-byte strings.
    """

    def __init__(
        self,
        data: bytes | bytearray | memoryview,
        fields: Iterable[Field] = (),
        regions: Iterable[Region] = (),
        *,
        title: str = "Memory layout",
        base_address: int = 0,
        bytes_per_row: int = 16,
        group_bytes: int = 4,
        address_digits: int = 8,
        collapse: bool = True,
        collapse_min_rows: int = 5,
        collapse_keep_edge_rows: int = 1,
        data_color_mode: str = "field",
        font_size: int = 13,
        byte_cell_w: int = 30,
        row_h: int = 30,
        group_gap: int = 10,
        table_gap: int = 42,
    ):
        self.data = bytes(data)
        self.fields = list(fields)
        self.regions = list(regions)

        self.title = title
        self.base_address = base_address
        self.bytes_per_row = bytes_per_row
        self.group_bytes = group_bytes
        self.address_digits = address_digits

        self.collapse = collapse
        self.collapse_min_rows = collapse_min_rows
        self.collapse_keep_edge_rows = collapse_keep_edge_rows

        self.data_color_mode = data_color_mode

        if self.data_color_mode not in {"field", "region", "none"}:
            raise ValueError(
                "data_color_mode must be one of: 'field', 'region', 'none'"
            )

        self.font_size = font_size
        self.byte_cell_w = byte_cell_w
        self.row_h = row_h
        self.group_gap = group_gap
        self.table_gap = table_gap

        self._validate()

        self._region_by_name = {
            region.name: region
            for region in self.regions
        }

        self._region_colors = {
            region: region.color or self._auto_color(region.name)
            for region in self.regions
        }

        self._region_color_by_name = {
            region.name: self._region_colors[region]
            for region in self.regions
        }

        self._field_colors = {}

        for field in self.fields:
            if field.color is not None:
                self._field_colors[field] = field.color
                continue

            if field.group is not None and field.group in self._region_color_by_name:
                base_color = self._region_color_by_name[field.group]

                if field.role is not None:
                    self._field_colors[field] = self._role_color(base_color, field.role)
                else:
                    self._field_colors[field] = self._mix(base_color, "#ffffff", 0.22)

                continue

            if field.role is not None:
                self._field_colors[field] = self._role_color(self._auto_color(field.role), field.role)
            else:
                self._field_colors[field] = self._auto_color(field.name)

        self._byte_owner = self._build_byte_owner()
        self._region_owner = self._build_region_owner()

    def _validate(self) -> None:
        if self.bytes_per_row <= 0:
            raise ValueError("bytes_per_row must be positive")

        if self.group_bytes <= 0:
            raise ValueError("group_bytes must be positive")

        if self.bytes_per_row % self.group_bytes != 0:
            raise ValueError("bytes_per_row must be divisible by group_bytes")

        for field in self.fields:
            if field.start < 0 or field.size < 0:
                raise ValueError(f"invalid field range: {field!r}")

        for region in self.regions:
            if region.start < 0 or region.size < 0:
                raise ValueError(f"invalid region range: {region!r}")

        self._validate_no_overlapping_fields()

    def _validate_no_overlapping_fields(self) -> None:
        """
        Concrete fields should not overlap.

        Larger structures should be represented as Region, not Field.
        """

        ordered = sorted(self.fields, key=lambda field: (field.start, field.end))

        for previous, current in zip(ordered, ordered[1:]):
            if previous.end > current.start:
                raise ValueError(
                    "overlapping concrete fields are not allowed: "
                    f"{previous.name} [{previous.start:#x}, {previous.end:#x}) "
                    f"overlaps "
                    f"{current.name} [{current.start:#x}, {current.end:#x}). "
                    "Use Region for larger wrapper objects."
                )

    def write_svg(self, path: str | pathlib.Path) -> None:
        pathlib.Path(path).write_text(self.to_svg(), encoding="utf-8")

    def to_svg(self) -> str:
        visible_rows, markers = self._visible_rows_and_markers()
        marker_by_row = {marker.row: marker for marker in markers}

        margin = 18
        title_h = 30
        table_title_h = 22
        header_h = 24
        legend_h = self._legend_height()

        addr_w = max(
            92,
            int(self.font_size * 0.68 * (self.address_digits + 2)),
        )

        table_w = addr_w + self._grid_w()

        x_data = margin
        x_fields = x_data + table_w + self.table_gap

        width = margin * 2 + table_w * 2 + self.table_gap

        body_lines = len(visible_rows) + len(markers)
        height = (
            margin * 2
            + title_h
            + table_title_h
            + header_h
            + body_lines * self.row_h
            + legend_h
        )

        parts: list[str] = []
        add = parts.append

        add(
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">'
        )

        self._render_style(add)

        add(
            f'<rect class="bg" x="0" y="0" '
            f'width="{width}" height="{height}"/>'
        )

        y = margin

        add(
            f'<text class="title" x="{margin}" '
            f'y="{y + self.font_size + 3}">'
            f'{escape(self.title)}</text>'
        )

        y += title_h

        add(
            f'<text class="tableTitle" x="{x_data + addr_w}" '
            f'y="{y + self.font_size}">data bytes</text>'
        )
        add(
            f'<text class="tableTitle" x="{x_fields + addr_w}" '
            f'y="{y + self.font_size}">field map</text>'
        )

        y += table_title_h

        self._render_header(add, x_data, y, addr_w)
        self._render_header(add, x_fields, y, addr_w)

        y += header_h

        for row in range(self._row_count()):
            marker = marker_by_row.get(row)

            if marker is not None:
                self._render_marker(
                    add=add,
                    marker=marker,
                    x_data=x_data,
                    x_fields=x_fields,
                    y=y,
                    table_w=table_w,
                )
                y += self.row_h

            if row not in visible_rows:
                continue

            self._render_data_row(add, row, x_data, y, addr_w)
            self._render_field_row(add, row, x_fields, y, addr_w)

            y += self.row_h

        self._render_region_legend(add, margin, y + 12, width - 2 * margin)

        add("</svg>")
        return "\n".join(parts)

    def _render_style(self, add) -> None:
        add("<style>")
        add(f"""
            .bg {{
                fill: #ffffff;
            }}

            .title {{
                font: 700 {self.font_size + 4}px ui-monospace,
                      SFMono-Regular, Menlo, Consolas, monospace;
                fill: #111827;
            }}

            .tableTitle {{
                font: 700 {self.font_size}px ui-monospace,
                      SFMono-Regular, Menlo, Consolas, monospace;
                fill: #374151;
            }}

            .mono {{
                font: {self.font_size}px ui-monospace,
                      SFMono-Regular, Menlo, Consolas, monospace;
                fill: #111827;
            }}

            .tiny {{
                font: {max(9, self.font_size - 2)}px ui-monospace,
                      SFMono-Regular, Menlo, Consolas, monospace;
                fill: #6b7280;
            }}

            .fieldText {{
                font: {max(9, self.font_size - 2)}px ui-monospace,
                      SFMono-Regular, Menlo, Consolas, monospace;
                fill: #111827;
            }}

            .legendText {{
                font: {max(9, self.font_size - 1)}px ui-monospace,
                      SFMono-Regular, Menlo, Consolas, monospace;
                fill: #374151;
            }}

            .cell {{
                stroke: #d1d5db;
                stroke-width: 0.8;
                rx: 3;
                ry: 3;
            }}

            .fieldSpan {{
                stroke: #d1d5db;
                stroke-width: 0.8;
                rx: 3;
                ry: 3;
            }}

            .emptyCell {{
                fill: #ffffff;
                stroke: #eef2f7;
                stroke-width: 0.8;
                rx: 3;
                ry: 3;
            }}

            .marker {{
                fill: #f9fafb;
                stroke: #cbd5e1;
                stroke-dasharray: 4 3;
                stroke-width: 1;
                rx: 5;
                ry: 5;
            }}

            .legendBox {{
                fill: #ffffff;
                stroke: #e5e7eb;
                stroke-width: 1;
                rx: 5;
                ry: 5;
            }}
            .regionSpan {{
                stroke: #d1d5db;
                stroke-width: 0.8;
                rx: 3;
                ry: 3;
            }}
        """)
        add("</style>")

    def _render_header(self, add, x: int, y: int, addr_w: int) -> None:
        add(f'<text class="tiny" x="{x}" y="{y + 16}">offset</text>')

        for col in range(self.bytes_per_row):
            cx = x + addr_w + self._col_x(col) + self.byte_cell_w / 2

            add(
                f'<text class="tiny" '
                f'text-anchor="middle" '
                f'dominant-baseline="middle" '
                f'x="{cx:.2f}" y="{y + 13}">'
                f'{col:02x}</text>'
            )

    def _render_data_row(self, add, row: int, x: int, y: int, addr_w: int) -> None:
        row_start = row * self.bytes_per_row
        row_end = min(row_start + self.bytes_per_row, len(self.data))
        cy = y + self.row_h / 2

        add(
            f'<text class="mono" x="{x}" y="{cy:.2f}" '
            f'dominant-baseline="middle">'
            f'{self.base_address + row_start:0{self.address_digits}x}'
            f'</text>'
        )

        for col in range(self.bytes_per_row):
            idx = row_start + col
            bx = x + addr_w + self._col_x(col)

            region = self._region_owner[idx] if idx < len(self._region_owner) else None
            color = "#ffffff"

            if idx < len(self.data):
                if self.data_color_mode == "field":
                    owner = self._byte_owner[idx] if idx < len(self._byte_owner) else None
                    color = self._field_colors.get(owner, "#ffffff") if owner else "#ffffff"

                elif self.data_color_mode == "region":
                    region = self._region_owner[idx] if idx < len(self._region_owner) else None
                    color = self._region_colors.get(region, "#ffffff") if region else "#ffffff"

                elif self.data_color_mode == "none":
                    color = "#ffffff"

            css_class = "cell" if idx < len(self.data) else "emptyCell"

            add(
                f'<rect class="{css_class}" '
                f'x="{bx:.2f}" y="{y + 2}" '
                f'width="{self.byte_cell_w}" '
                f'height="{self.row_h - 4}" '
                f'fill="{color}"/>'
            )

            if idx < row_end:
                add(
                    f'<text class="mono" '
                    f'text-anchor="middle" '
                    f'dominant-baseline="middle" '
                    f'x="{bx + self.byte_cell_w / 2:.2f}" '
                    f'y="{cy:.2f}">'
                    f'{self.data[idx]:02x}</text>'
                )

    def _render_field_row(self, add, row: int, x: int, y: int, addr_w: int) -> None:
        row_start = row * self.bytes_per_row
        row_end = min(row_start + self.bytes_per_row, len(self.data))
        cy = y + self.row_h / 2

        add(
            f'<text class="mono" x="{x}" y="{cy:.2f}" '
            f'dominant-baseline="middle">'
            f'{self.base_address + row_start:0{self.address_digits}x}'
            f'</text>'
        )

        # Draw faint byte grid.
        for col in range(self.bytes_per_row):
            bx = x + addr_w + self._col_x(col)

            add(
                f'<rect class="emptyCell" '
                f'x="{bx:.2f}" y="{y + 2}" '
                f'width="{self.byte_cell_w}" '
                f'height="{self.row_h - 4}"/>'
            )

        # Draw region background spans first.
        for region in self.regions:
            if not region.intersects(row_start, row_end):
                continue

            abs_a = max(region.start, row_start)
            abs_b = min(region.end, row_end)

            if abs_a >= abs_b:
                continue

            col_a = abs_a - row_start
            col_b = abs_b - row_start

            bx = x + addr_w + self._col_x(col_a)
            bw = self._span_w(col_a, col_b)
            color = self._region_colors[region]

            add(
                f'<rect class="regionSpan" '
                f'x="{bx:.2f}" y="{y + 2}" '
                f'width="{bw:.2f}" '
                f'height="{self.row_h - 4}" '
                f'fill="{color}"/>'
            )

        # Draw only concrete fields. No wrapper regions here.
        for field in self.fields:
            if not field.intersects(row_start, row_end):
                continue

            abs_a = max(field.start, row_start)
            abs_b = min(field.end, row_end)

            if abs_a >= abs_b:
                continue

            col_a = abs_a - row_start
            col_b = abs_b - row_start

            bx = x + addr_w + self._col_x(col_a)
            bw = self._span_w(col_a, col_b)
            color = self._field_colors[field]
            label = self._label_for_span(field, abs_a, bw)

            add(
                f'<rect class="fieldSpan" '
                f'x="{bx:.2f}" y="{y + 2}" '
                f'width="{bw:.2f}" '
                f'height="{self.row_h - 4}" '
                f'fill="{color}"/>'
            )

            if label:
                add(
                    f'<text class="fieldText" '
                    f'text-anchor="middle" '
                    f'dominant-baseline="middle" '
                    f'x="{bx + bw / 2:.2f}" '
                    f'y="{cy:.2f}">'
                    f'{escape(label)}</text>'
                )

    def _render_marker(
        self,
        *,
        add,
        marker: _CollapseMarker,
        x_data: int,
        x_fields: int,
        y: int,
        table_w: int,
    ) -> None:
        text = (
            f"⋮ {marker.hidden_rows} rows / "
            f"{marker.hidden_bytes} bytes omitted: {marker.label}"
        )

        for x in (x_data, x_fields):
            add(
                f'<rect class="marker" '
                f'x="{x}" y="{y + 2}" '
                f'width="{table_w}" '
                f'height="{self.row_h - 4}"/>'
            )
            add(
                f'<text class="tiny" '
                f'x="{x + 10}" '
                f'y="{y + self.row_h / 2:.2f}" '
                f'dominant-baseline="middle">'
                f'{escape(text)}</text>'
            )

    def _render_region_legend(self, add, x: int, y: int, width: int) -> None:
        if not self.regions:
            return

        line_h = 24
        box_h = self._legend_height()

        add(
            f'<rect class="legendBox" '
            f'x="{x}" y="{y}" '
            f'width="{width}" height="{box_h}"/>'
        )

        add(
            f'<text class="tableTitle" x="{x + 12}" '
            f'y="{y + 22}">regions</text>'
        )

        current_y = y + 48

        for region in self.regions:
            region_color = self._region_colors[region]

            label = region.name

            if region.note:
                label += f" — {region.note}"

            if region.collapsible:
                label += " — shortened in table"

            add(
                f'<rect x="{x + 14}" y="{current_y - 14}" '
                f'width="14" height="14" '
                f'fill="{region_color}" stroke="#d1d5db"/>'
            )

            add(
                f'<text class="legendText" x="{x + 36}" '
                f'y="{current_y}">{escape(label)}</text>'
            )

            current_y += line_h

            grouped_fields = [
                field
                for field in self.fields
                if field.group == region.name
            ]

            role_ranges: dict[str, list[Field]] = {}

            for field in grouped_fields:
                role = field.role or field.name
                role_ranges.setdefault(role, []).append(field)

            for role, role_fields in role_ranges.items():
                sample_field = role_fields[0]
                role_color = self._field_colors[sample_field]

                first = min(field.start for field in role_fields)
                last = max(field.end for field in role_fields)

                if len(role_fields) == 1:
                    role_label = f"  {role}: {role_fields[0].name}"
                else:
                    role_label = f"  {role}: {len(role_fields)} field(s)"

                add(
                    f'<rect x="{x + 36}" y="{current_y - 13}" '
                    f'width="12" height="12" '
                    f'fill="{role_color}" stroke="#d1d5db"/>'
                )

                add(
                    f'<text class="legendText" x="{x + 56}" '
                    f'y="{current_y}">{escape(role_label)}</text>'
                )

                current_y += line_h

            current_y += 6

    def _label_for_span(self, field: Field, absolute_start: int, width: float) -> str:
        if absolute_start != field.start:
            text = "…"
        else:
            text = field.name

        approx_char_w = self.font_size * 0.58
        max_chars = max(1, int(width / approx_char_w))

        if len(text) <= max_chars:
            return text

        if max_chars <= 1:
            return ""
        if max_chars <= 3:
            return "…"

        return text[: max_chars - 1] + "…"

    def _visible_rows_and_markers(self) -> tuple[set[int], list[_CollapseMarker]]:
        row_count = self._row_count()
        visible = set(range(row_count))
        markers: list[_CollapseMarker] = []

        if not self.collapse:
            return visible, markers

        collapsible_ranges: list[tuple[int, int, str]] = []

        for field in self.fields:
            if field.collapsible:
                collapsible_ranges.append((field.start, field.end, field.name))

        for region in self.regions:
            if region.collapsible:
                collapsible_ranges.append((region.start, region.end, region.name))

        for start, end, label in collapsible_ranges:
            if end <= start:
                continue

            first_full_row = math.ceil(start / self.bytes_per_row)
            last_full_row = math.floor(end / self.bytes_per_row) - 1
            full_rows = last_full_row - first_full_row + 1

            if full_rows < self.collapse_min_rows:
                continue

            hide_start = first_full_row + self.collapse_keep_edge_rows
            hide_end = last_full_row - self.collapse_keep_edge_rows

            if hide_start > hide_end:
                continue

            hidden_rows = hide_end - hide_start + 1

            for row in range(hide_start, hide_end + 1):
                visible.discard(row)

            markers.append(
                _CollapseMarker(
                    row=hide_start,
                    hidden_rows=hidden_rows,
                    hidden_bytes=hidden_rows * self.bytes_per_row,
                    label=label,
                )
            )

        markers.sort(key=lambda marker: marker.row)
        return visible, markers

    def _build_byte_owner(self) -> list[Optional[Field]]:
        owner: list[Optional[Field]] = [None] * len(self.data)

        for field in self.fields:
            lo = max(0, field.start)
            hi = min(len(self.data), field.end)

            for idx in range(lo, hi):
                owner[idx] = field

        return owner

    def _build_region_owner(self) -> list[Optional[Region]]:
        owner: list[Optional[Region]] = [None] * len(self.data)

        for region in self.regions:
            lo = max(0, region.start)
            hi = min(len(self.data), region.end)

            for idx in range(lo, hi):
                owner[idx] = region

        return owner

    def _legend_height(self) -> int:
        if not self.regions:
            return 0

        line_h = 24
        height = 48

        for region in self.regions:
            height += line_h

            roles = {
                field.role or field.name
                for field in self.fields
                if field.group == region.name
            }

            height += line_h * len(roles)
            height += 6

        return height + 12

    def _row_count(self) -> int:
        return max(1, math.ceil(len(self.data) / self.bytes_per_row))

    def _grid_w(self) -> int:
        groups = self.bytes_per_row // self.group_bytes

        return (
            self.bytes_per_row * self.byte_cell_w
            + (groups - 1) * self.group_gap
        )

    def _col_x(self, col: int) -> int:
        return (
            col * self.byte_cell_w
            + (col // self.group_bytes) * self.group_gap
        )

    def _span_w(self, col_a: int, col_b: int) -> float:
        if col_b <= col_a:
            return 0

        x0 = self._col_x(col_a)
        x1 = self._col_x(col_b - 1) + self.byte_cell_w

        return x1 - x0

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

    @staticmethod
    def _hex_to_rgb(color: str) -> tuple[int, int, int]:
        color = color.lstrip("#")

        if len(color) != 6:
            raise ValueError(f"expected #rrggbb color, got {color!r}")

        return (
            int(color[0:2], 16),
            int(color[2:4], 16),
            int(color[4:6], 16),
        )


    @staticmethod
    def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
        r, g, b = rgb

        return f"#{r:02x}{g:02x}{b:02x}"


    @classmethod
    def _mix(cls, a: str, b: str, amount: float) -> str:
        """
        Mix color a towards color b.

        amount = 0.0 gives a
        amount = 1.0 gives b
        """

        ar, ag, ab = cls._hex_to_rgb(a)
        br, bg, bb = cls._hex_to_rgb(b)

        r = round(ar * (1.0 - amount) + br * amount)
        g = round(ag * (1.0 - amount) + bg * amount)
        b = round(ab * (1.0 - amount) + bb * amount)

        return cls._rgb_to_hex((r, g, b))


    @classmethod
    def _role_color(cls, base_color: str, role: str) -> str:
        """
        Derive stable, related colors from a region base color.

        The output remains visually related to the owning region, but different
        roles get distinguishable shades.

        You can tweak these constants later.
        """

        normalized = role.lower()

        if normalized in {"id", "ids", "segment_id"}:
            return cls._mix(base_color, "#3b82f6", 0.30)

        if normalized in {"offset", "offsets", "position", "link"}:
            return cls._mix(base_color, "#10b981", 0.30)

        if normalized in {"payload", "data"}:
            return cls._mix(base_color, "#f59e0b", 0.24)

        if normalized in {"magic", "version", "header"}:
            return cls._mix(base_color, "#8b5cf6", 0.24)

        # Stable fallback for arbitrary roles.
        h = hashlib.sha1(role.encode("utf-8")).digest()[0]
        amount = 0.18 + (h % 5) * 0.06

        # Alternate between mixing towards blue and green-ish colors.
        target = "#3b82f6" if h % 2 == 0 else "#10b981"

        return cls._mix(base_color, target, amount)
