# VIBECODED VISUALIZATION THINGY, contains methods that construct some memory layout
from memory_layout_core import Field, Region, HexLayoutSvg

import argparse
import pathlib

def u32be(value: int) -> bytes:
    return value.to_bytes(4, "big")


def build_shared_memory_posix_write_segment_test_example(
    *,
    c_contiguous_segment_count: int = 1,
    magic: bytes = bytes.fromhex("55424152"),
    version: int = 1,
    segment_data: bytes = bytes.fromhex("ba52ab1e"),
    segment_count: int = 3,
    c_partial_table_link_id: int = 0xFFFFFFFF,
) -> tuple[bytes, list[Field], list[Region]]:
    """
    Example matching the current GTest:

        SharedMemoryPosix shm("/posix-test");

        data = { 0xba, 0x52, 0xab, 0x1e };

        shm.writeSegment(data);
        shm.writeSegment(data);
        shm.writeSegment(data);

    With c_contiguous_segment_count = 1, each partial segment table can hold
    exactly one normal id/offset entry plus one link entry. Therefore three
    written segments require three partial segment tables.

    This builder intentionally models the current write order of the skeleton:

        header
        partial table #0
        payload #0
        partial table #1
        payload #1
        partial table #2
        payload #2

    Payloads are concrete fields, not regions.
    The shared memory header is also not represented as a region.
    """

    if c_contiguous_segment_count <= 0:
        raise ValueError("c_contiguous_segment_count must be positive")

    if segment_count <= 0:
        raise ValueError("segment_count must be positive")

    id_size = 4
    offset_size = 4
    table_entry_size = id_size + offset_size

    # One extra entry is the partial-table link entry.
    partial_table_entries = c_contiguous_segment_count + 1
    partial_table_size = partial_table_entries * table_entry_size

    magic_offset = 0
    version_offset = magic_offset + 4
    head = version_offset + 4

    segment_size = len(segment_data)

    fields: list[Field] = [
        Field(
            magic_offset,
            4,
            "magic",
            role="magic",
        ),
        Field(
            version_offset,
            4,
            "version",
            role="version",
        ),
    ]

    regions: list[Region] = []

    chunks: list[bytes] = [
        magic,
        u32be(version),
    ]

    # Each element stores:
    #   table index,
    #   table offset,
    #   payload offset,
    #   next table offset
    table_infos: list[tuple[int, int, int, int]] = []

    for segment_idx in range(segment_count):
        table_idx = segment_idx // c_contiguous_segment_count

        # For c_contiguous_segment_count = 1 this creates one table per segment.
        # For larger values this still works, but the layout is intentionally
        # optimized for demonstrating c_contiguous_segment_count = 1.
        if segment_idx % c_contiguous_segment_count != 0:
            raise NotImplementedError(
                "This illustrative builder is intended for "
                "c_contiguous_segment_count = 1. For larger tables, use a "
                "different packed-table builder."
            )

        table_offset = head
        payload_offset = table_offset + partial_table_size

        next_table_offset = (
            payload_offset + segment_size
            if segment_idx + 1 < segment_count
            else 0
        )

        table_infos.append(
            (
                table_idx,
                table_offset,
                payload_offset,
                next_table_offset,
            )
        )

        head = payload_offset + segment_size

    for table_idx, table_offset, payload_offset, next_table_offset in table_infos:
        table_region = f"partial segment table #{table_idx}"

        regions.append(
            Region(
                table_offset,
                partial_table_size,
                table_region,
                color="#ccfbf1",
                note=(
                    "one id/offset entry plus link entry"
                    if c_contiguous_segment_count == 1
                    else (
                        f"{c_contiguous_segment_count} id/offset entries "
                        "plus link entry"
                    )
                ),
                collapsible=False,
            )
        )

        # With c_contiguous_segment_count = 1, segment_idx == table_idx.
        segment_id = table_idx

        table = bytearray()

        # Normal segment-table entry.
        table += u32be(segment_id)
        table += u32be(payload_offset)

        fields.append(
            Field(
                table_offset,
                4,
                f"id[{segment_id}]",
                group=table_region,
                role="id",
            )
        )
        fields.append(
            Field(
                table_offset + 4,
                4,
                f"offset[{segment_id}]",
                group=table_region,
                role="offset",
            )
        )

        # Link entry.
        link_entry_offset = table_offset + c_contiguous_segment_count * table_entry_size

        table += u32be(c_partial_table_link_id)
        table += u32be(next_table_offset)

        fields.append(
            Field(
                link_entry_offset,
                4,
                "link id",
                group=table_region,
                role="id",
            )
        )
        fields.append(
            Field(
                link_entry_offset + 4,
                4,
                "next table",
                group=table_region,
                role="offset",
            )
        )

        chunks.append(bytes(table))

        # Payload is intentionally a field, not a region.
        fields.append(
            Field(
                payload_offset,
                segment_size,
                f"payload[{segment_id}]",
                role="payload",
            )
        )

        chunks.append(segment_data)

    blob = b"".join(chunks)

    return blob, fields, regions

def build_shared_memory_posix_example() -> tuple[bytes, list[Field], list[Region]]:
    """
    Small illustrative layout:

        header
        partial table #0
            id[0] -> payload #0
            link  -> partial table #1
        payload #0
        partial table #1
            id[1] -> payload #1
            link  -> null
        payload #1

    This intentionally uses one segment entry per partial table to make the
    linking concept easy to see in the generated documentation.
    """

    c_contiguous_segment_count = 1
    id_size = 4

    partial_table_entries = c_contiguous_segment_count + 1
    partial_table_size = partial_table_entries * id_size * 2

    magic_offset = 0
    version_offset = magic_offset + 4
    table0_offset = version_offset + 4

    payload0_size = 128
    payload1_size = 160

    payload0_offset = table0_offset + partial_table_size
    table1_offset = payload0_offset + payload0_size
    payload1_offset = table1_offset + partial_table_size

    magic = bytes.fromhex("55424152")
    version = (1).to_bytes(4, "big")

    table0 = bytearray()
    table0 += (0).to_bytes(4, "big")
    table0 += payload0_offset.to_bytes(4, "big")
    table0 += (0xFFFFFFFF).to_bytes(4, "big")
    table0 += table1_offset.to_bytes(4, "big")

    payload0 = bytes(range(payload0_size))

    table1 = bytearray()
    table1 += (1).to_bytes(4, "big")
    table1 += payload1_offset.to_bytes(4, "big")
    table1 += (0xFFFFFFFF).to_bytes(4, "big")
    table1 += (0).to_bytes(4, "big")

    payload1 = bytes((0x80 + i) & 0xFF for i in range(payload1_size))

    blob = magic + version + bytes(table0) + payload0 + bytes(table1) + payload1

    header_region = "shared memory header"
    table0_region = "partial segment table #0"
    payload0_region = "segment payload #0"
    table1_region = "partial segment table #1"
    payload1_region = "segment payload #1"

    regions: list[Region] = [
        Region(
            magic_offset,
            8,
            header_region,
            color="#e0e7ff",
            note="magic + version",
        ),
        Region(
            table0_offset,
            partial_table_size,
            table0_region,
            color="#ccfbf1",
            note="one id/offset entry plus link to next table",
        ),
        Region(
            payload0_offset,
            payload0_size,
            payload0_region,
            color="#fef3c7",
            note="first segment data",
            collapsible=True,
        ),
        Region(
            table1_offset,
            partial_table_size,
            table1_region,
            color="#d1fae5",
            note="second partial table",
        ),
        Region(
            payload1_offset,
            payload1_size,
            payload1_region,
            color="#ffedd5",
            note="second segment data",
            collapsible=True,
        ),
    ]

    fields: list[Field] = [
        Field(
            magic_offset,
            4,
            "magic",
            group=header_region,
            role="magic",
        ),
        Field(
            version_offset,
            4,
            "version",
            group=header_region,
            role="version",
        ),

        Field(
            table0_offset,
            4,
            "id[0]",
            group=table0_region,
            role="id",
        ),
        Field(
            table0_offset + 4,
            4,
            "offset[0]",
            group=table0_region,
            role="offset",
        ),
        Field(
            table0_offset + 8,
            4,
            "link id",
            group=table0_region,
            role="id",
        ),
        Field(
            table0_offset + 12,
            4,
            "next table",
            group=table0_region,
            role="offset",
        ),

        Field(
            payload0_offset,
            16,
            "payload #0 head",
            group=payload0_region,
            role="payload",
        ),
        Field(
            payload0_offset + payload0_size - 16,
            16,
            "payload #0 tail",
            group=payload0_region,
            role="payload",
        ),

        Field(
            table1_offset,
            4,
            "id[1]",
            group=table1_region,
            role="id",
        ),
        Field(
            table1_offset + 4,
            4,
            "offset[1]",
            group=table1_region,
            role="offset",
        ),
        Field(
            table1_offset + 8,
            4,
            "link id",
            group=table1_region,
            role="id",
        ),
        Field(
            table1_offset + 12,
            4,
            "next table",
            group=table1_region,
            role="offset",
        ),

        Field(
            payload1_offset,
            16,
            "payload #1 head",
            group=payload1_region,
            role="payload",
        ),
        Field(
            payload1_offset + payload1_size - 16,
            16,
            "payload #1 tail",
            group=payload1_region,
            role="payload",
        ),
    ]

    return blob, fields, regions


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--out-dir",
        "--outdir",
        dest="out_dir",
        required=True,
        help="Output directory for generated image",
    )

    parser.add_argument(
        "--data-color-mode",
        choices=("field", "region", "none"),
        default="field",
        help=(
            "How to color the left data table: "
            "'field' uses concrete field colors, "
            "'region' uses region colors, "
            "'none' disables coloring."
        ),
    )

    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_file = out_dir / "shared_memory_layout.svg"

    blob, fields, regions = build_shared_memory_posix_write_segment_test_example()

    print(f"Writing to {output_file}")

    HexLayoutSvg(
        blob,
        fields,
        regions,
        title="SharedMemoryPosix writeSegment test layout",
        bytes_per_row=16,
        group_bytes=4,
        address_digits=8,
        collapse=True,
        collapse_min_rows=6,
        collapse_keep_edge_rows=1,
        data_color_mode=args.data_color_mode,
        byte_cell_w=30,
        row_h=30,
    ).write_svg(output_file)


if __name__ == "__main__":
    main()