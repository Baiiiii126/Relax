# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Patch the DeepEyesV2 RL parquet files for the Relax agentic runtime.

Does two things per row:

1. Inject ``extra_info["data_source"]`` (perception / reason / search /
   vstar-test) so ``reward_deepeyes_v2.py`` can route the scorer.
2. Rewrite the ``prompt`` column from the upstream dual schema
   (``<code>...</code>`` for python, ``<tool_call>...`` for search) to
   the unified single schema where every action goes through
   ``<tool_call>{"name": "python_exec"|"search"|"image_search", ...}``.
   This matches what the per-session agent (``app/agent.py``) and the
   smoke harness use, and lets Instruct-line VLMs (which are trained on
   ``<tool_call>`` but not ``<code>``) drive the agent without SFT.

Pass ``--no-unified-schema`` to keep the upstream dual schema in the
prompt column (useful if you have a checkpoint that was SFT-ed on the
upstream V2 prompt).

The ``cached_images/`` directory and ancillary JSON files
(e.g. ``fvqa_train_image_search_results_cache.json``) are **not**
touched here — handle those via ``cache_convert.py``.

Mapping rule (matches ``_SCORER_REGISTRY`` in ``reward_deepeyes_v2.py``):

    perception_all_*.parquet -> "perception"
    reason.parquet           -> "reason"
    search.parquet           -> "search"
    vstar_test.parquet       -> "vstar-test"   (hyphen, not underscore!)

Usage:

    python examples/deepeyes_v2/convert_tool/rl_data_convert.py \\
        --input  /path/to/DeepEyesV2_RL \\
        --output /path/to/DeepEyesV2_RL_with_datasource

Any unexpected condition (missing extra_info column, unknown filename,
empty input dir, etc.) raises an error -- this script is a single-purpose
data conversion tool, not a fault-tolerant pipeline.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


# Surface the canonical unified-schema prompt strings from app/prompt.py
# (single source of truth shared with the smoke harness + per-session agent).
_EXAMPLE_DIR = Path(__file__).resolve().parents[1]
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))
from app.prompt import UNIFIED_SYSTEM_PROMPT, rewrite_user_format_reminder  # noqa: E402


# ---------------------------------------------------------------------------
# Filename -> data_source tag mapping
# ---------------------------------------------------------------------------
# Keys are matched as compiled regex against the parquet file *stem* (without
# the ``.parquet`` suffix). Order matters: the first matching pattern wins,
# so put the more specific patterns first.
_FILENAME_TO_DATA_SOURCE: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^perception_all_\d+$"), "perception"),
    (re.compile(r"^reason$"), "reason"),
    (re.compile(r"^search$"), "search"),
    (re.compile(r"^vstar_test$"), "vstar-test"),
]


def resolve_data_source(stem: str) -> str:
    """Return the data_source tag for a parquet file stem, or raise on
    unknown."""
    for pattern, tag in _FILENAME_TO_DATA_SOURCE:
        if pattern.match(stem):
            return tag
    known = ", ".join(p.pattern for p, _ in _FILENAME_TO_DATA_SOURCE)
    raise ValueError(f"unknown parquet file (no rule for stem={stem!r}); known patterns: {known}")


# ---------------------------------------------------------------------------
# pyarrow helpers -- patch a struct column without going through pandas
# ---------------------------------------------------------------------------
#
# We *must* avoid pandas here:
#   1. ``ParquetFile.read()`` raises
#      ``ArrowNotImplementedError: Nested data conversions not implemented
#      for chunked array outputs`` when a column is a nested type
#      (struct / list) AND the file has more than one row group, because
#      pyarrow cannot materialise a chunked nested column for pandas.
#   2. ``Table.combine_chunks()`` works around (1) but fails with
#      ``ArrowInvalid: offset overflow while concatenating arrays`` when
#      the total payload of a binary column exceeds 2 GiB (perception_all_5
#      contains the entire base64 image bytes inline -- well over 2 GiB).
#
# The combination forces a row-group-by-row-group, chunked-array workflow
# that operates purely on pyarrow.StructArrays.
#
def _add_field_to_struct_chunk(
    struct_chunk: pa.StructArray,
    field_name: str,
    field_value: str,
    field_type: pa.DataType,
) -> pa.StructArray:
    """Return a copy of ``struct_chunk`` with ``field_name=field_value``
    appended.

    If the field already exists it is *overwritten*, so re-running the
    converter on an already-patched dataset is idempotent.
    """
    existing_type = struct_chunk.type
    existing_field_names = [existing_type.field(i).name for i in range(existing_type.num_fields)]

    n = len(struct_chunk)
    new_value_arr = pa.array([field_value] * n, type=field_type)

    arrays: list[pa.Array] = []
    fields: list[pa.Field] = []
    overwritten = False
    for i, name in enumerate(existing_field_names):
        if name == field_name:
            arrays.append(new_value_arr)
            fields.append(pa.field(field_name, field_type))
            overwritten = True
        else:
            arrays.append(struct_chunk.field(i))
            fields.append(existing_type.field(i))
    if not overwritten:
        arrays.append(new_value_arr)
        fields.append(pa.field(field_name, field_type))

    return pa.StructArray.from_arrays(arrays, fields=fields)


def _patch_extra_info_column(
    extra_info_col: pa.ChunkedArray,
    data_source: str,
) -> pa.ChunkedArray:
    """Add ``data_source=<tag>`` to every row of the ``extra_info`` struct
    column."""
    if not pa.types.is_struct(extra_info_col.type):
        raise TypeError(f"expected 'extra_info' to be a struct column, got {extra_info_col.type}")
    new_chunks = [
        _add_field_to_struct_chunk(chunk, "data_source", data_source, pa.string()) for chunk in extra_info_col.chunks
    ]
    return pa.chunked_array(new_chunks)


# ---------------------------------------------------------------------------
# Prompt rewrite — collapse upstream <code>/<tool_call> dual schema to unified
# ---------------------------------------------------------------------------
# The prompt column is `list<struct<role,content>>`; size is small (just system
# + user text), so per-row Python-side rewrite is fine memory-wise. The 2-GiB
# concern that drove the row-group streaming in `convert_file` applies only to
# the binary `images` column, not here.
def _rewrite_prompt_chunk(prompt_chunk: pa.ListArray) -> pa.ListArray:
    py = prompt_chunk.to_pylist()
    for msg_list in py:
        if not msg_list:
            continue
        # Replace the system message verbatim. If the upstream schema ever
        # moves the system message off index 0 we want a loud failure rather
        # than a silent miss, so we scan + assert.
        sys_indices = [i for i, m in enumerate(msg_list) if m.get("role") == "system"]
        if len(sys_indices) > 1:
            raise ValueError(f"prompt has {len(sys_indices)} system messages; only one expected")
        for i in sys_indices:
            msg_list[i]["content"] = UNIFIED_SYSTEM_PROMPT
        # Rewrite the "Format strictly as ..." sentence in every user turn
        # (idempotent — already-unified text is matched + replaced with the
        # same template).
        for m in msg_list:
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                m["content"] = rewrite_user_format_reminder(m["content"])
    return pa.array(py, type=prompt_chunk.type)


def _rewrite_prompt_column(prompt_col: pa.ChunkedArray) -> pa.ChunkedArray:
    if not pa.types.is_list(prompt_col.type):
        raise TypeError(f"expected 'prompt' to be a list column, got {prompt_col.type}")
    return pa.chunked_array([_rewrite_prompt_chunk(chunk) for chunk in prompt_col.chunks])


# ---------------------------------------------------------------------------
# Per-file conversion
# ---------------------------------------------------------------------------
def _patch_table(table: pa.Table, data_source: str, *, unified_schema: bool) -> pa.Table:
    """Return a copy of ``table`` with ``extra_info.data_source = <tag>`` set,
    and (when ``unified_schema``) the ``prompt`` column rewritten to the unified
    <tool_call>{name:python_exec,...} schema."""
    extra_info_col = table.column("extra_info")
    patched_col = _patch_extra_info_column(extra_info_col, data_source)
    extra_info_idx = table.column_names.index("extra_info")
    table = table.set_column(extra_info_idx, "extra_info", patched_col)

    if unified_schema and "prompt" in table.column_names:
        prompt_col = table.column("prompt")
        new_prompt = _rewrite_prompt_column(prompt_col)
        prompt_idx = table.column_names.index("prompt")
        table = table.set_column(prompt_idx, "prompt", new_prompt)

    return table


def convert_file(src: Path, dst: Path, data_source: str, *, unified_schema: bool) -> int:
    """Convert one parquet file row-group by row-group, preserving the source's
    row-group layout. Returns the number of rows written.

    Streaming is required: some row groups (e.g. perception_all_5) contain
    nested binary columns (images) whose total size exceeds the 2 GiB single-
    buffer limit of pyarrow's ChunkedArray-to-Array conversion. Writing
    everything as one big row group via ``pq.write_table`` would produce a file
    that ``pq.read_table`` itself cannot load.
    """
    pf = pq.ParquetFile(src)
    total_rows = pf.metadata.num_rows
    num_row_groups = pf.num_row_groups
    schema_names = pf.schema_arrow.names
    if "extra_info" not in schema_names:
        raise ValueError(
            f"{src}: parquet schema is missing 'extra_info' column (found: {schema_names}); refusing to convert."
        )

    print(
        f"  -> reading {total_rows:>7d} rows from {num_row_groups} row group(s), columns={schema_names}",
        flush=True,
    )

    dst.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    new_field_names: list[str] | None = None
    sample_value: str | None = None
    rows_written = 0
    try:
        for rg_idx in range(num_row_groups):
            rg_table = pf.read_row_group(rg_idx)
            rg_table = _patch_table(rg_table, data_source, unified_schema=unified_schema)

            if writer is None:
                # Initialize writer with the patched schema (extra_info now
                # has the data_source field). Subsequent row groups must
                # match this schema, which they will since every row group
                # goes through the same _patch_table().
                writer = pq.ParquetWriter(dst, rg_table.schema, compression="snappy")
                # Sanity-check the patch on the first row group only.
                first_chunk = rg_table.column("extra_info").chunks[0]
                new_field_names = [first_chunk.type.field(i).name for i in range(first_chunk.type.num_fields)]
                sample_value = first_chunk.field("data_source")[0].as_py()
                assert sample_value == data_source, f"BUG: expected data_source={data_source!r}, got {sample_value!r}"

            writer.write_table(rg_table)
            rows_written += rg_table.num_rows
    finally:
        if writer is not None:
            writer.close()

    print(
        f"     extra_info fields after patch: {new_field_names}, data_source={sample_value!r}",
        flush=True,
    )
    print(
        f"     wrote -> {dst} ({dst.stat().st_size / 1024 / 1024:.1f} MiB, {num_row_groups} row group(s))",
        flush=True,
    )
    return rows_written


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description=("Inject extra_info['data_source'] into DeepEyesV2 RL parquet files."),
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input directory containing the raw DeepEyesV2_RL parquet files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help=("Output directory for the patched parquet files. Will be created if it does not exist."),
    )
    parser.add_argument(
        "--no-unified-schema",
        dest="unified_schema",
        action="store_false",
        help=(
            "Skip the prompt rewrite. Leaves the upstream <code>...</code> dual schema in place. "
            "Default is to rewrite so that the converted parquets match the runtime agent's "
            "unified <tool_call>{name:python_exec,...} schema."
        ),
    )
    parser.set_defaults(unified_schema=True)
    args = parser.parse_args()

    input_dir: Path = args.input.resolve()
    output_dir: Path = args.output.resolve()

    if not input_dir.is_dir():
        raise NotADirectoryError(f"--input not a directory: {input_dir}")
    if input_dir == output_dir:
        raise ValueError("--input and --output must point to different directories")

    parquet_files = sorted(input_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"no *.parquet files under {input_dir}")

    print(f"input:           {input_dir}")
    print(f"output:          {output_dir}")
    print(f"unified_schema:  {args.unified_schema}")
    print(f"found {len(parquet_files)} parquet file(s)")
    print()

    t0 = time.time()
    total_rows = 0
    for src in parquet_files:
        data_source = resolve_data_source(src.stem)
        dst = output_dir / src.name
        print(f"[convert] {src.name}  ->  data_source={data_source!r}")
        total_rows += convert_file(src, dst, data_source, unified_schema=args.unified_schema)
        print()

    elapsed = time.time() - t0
    print(f"done: {len(parquet_files)} file(s), {total_rows} rows in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
