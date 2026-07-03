# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Generate a tiny synthetic smoke parquet for DeepEyes V2 local debug.

Writes 4 rows matching the post-convert V2 schema:
  prompt        list<struct<role: str, content: str>>
  images        list<binary>            (raw PNG bytes)
  reward_model  struct<ground_truth: str>
  extra_info    struct<index: int, data_source: str>

The 4 rows cover all four data_source values the reward router knows:
  row 0: perception     -- red square with "42", crop+OCR
  row 1: reason         -- two circles, count them
  row 2: search         -- triangle with label, web-search style
  row 3: vstar-test     -- yes/no perception question

Each image is generated programmatically (PIL only), no external assets.

Usage:
    python examples/deepeyes_v2_agentic/scripts/build_smoke_parquet.py \\
        --output /tmp/smoke.parquet

The output file can be fed directly to:
    python examples/deepeyes_v2_agentic/scripts/run_single_session.py \\
        --parquet /tmp/smoke.parquet --row 0
"""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _png_bytes(img: Image.Image) -> bytes:
    buf = BytesIO()
    (img if img.mode == "RGB" else img.convert("RGB")).save(buf, format="PNG")
    return buf.getvalue()


def _perception_image() -> Image.Image:
    img = Image.new("RGB", (512, 512), color="white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((180, 180, 332, 332), fill="red", outline="black", width=3)
    draw.text((220, 240), "42", fill="white", font=_load_font(24))
    return img


def _reason_image() -> Image.Image:
    img = Image.new("RGB", (512, 512), color="lightyellow")
    draw = ImageDraw.Draw(img)
    draw.ellipse((100, 200, 200, 300), fill="blue", outline="black", width=2)
    draw.ellipse((300, 200, 400, 300), fill="blue", outline="black", width=2)
    return img


def _search_image() -> Image.Image:
    img = Image.new("RGB", (512, 512), color="lightblue")
    draw = ImageDraw.Draw(img)
    draw.polygon([(256, 100), (150, 380), (362, 380)], fill="green", outline="black")
    draw.text((180, 420), "MysteryShape", fill="black", font=_load_font(20))
    return img


def _vstar_image() -> Image.Image:
    img = Image.new("RGB", (512, 512), color="white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((50, 50, 250, 250), fill="orange")  # left orange square
    draw.ellipse((280, 280, 460, 460), fill="purple")  # bottom-right purple circle
    return img


SAMPLES = [
    {
        "image": _perception_image(),
        "prompt": "There is a red square in the centre of this image with a number written inside it. Crop the red square out of `image_1` to see the number clearly, then tell me what number it is.",
        "ground_truth": "42",
        "data_source": "perception",
        "index": 0,
    },
    {
        "image": _reason_image(),
        "prompt": "How many blue circles are in this image? Use `image_1` to count them by writing code that detects the blue regions.",
        "ground_truth": "2",
        "data_source": "reason",
        "index": 1,
    },
    {
        "image": _search_image(),
        "prompt": "What is the shape in this image, and what is the label written below it?",
        "ground_truth": "triangle, MysteryShape",
        "data_source": "search",
        "index": 2,
    },
    {
        "image": _vstar_image(),
        "prompt": "Is there an orange square in this image? Answer yes or no.",
        "ground_truth": "yes",
        "data_source": "vstar-test",
        "index": 3,
    },
]


def build_table() -> pa.Table:
    prompts = []
    images = []
    rewards = []
    extras = []
    for s in SAMPLES:
        prompts.append([{"role": "user", "content": s["prompt"]}])
        images.append([_png_bytes(s["image"])])
        rewards.append({"ground_truth": s["ground_truth"]})
        extras.append({"index": s["index"], "data_source": s["data_source"]})

    schema = pa.schema(
        [
            ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
            ("images", pa.list_(pa.binary())),
            ("reward_model", pa.struct([("ground_truth", pa.string())])),
            ("extra_info", pa.struct([("index", pa.int64()), ("data_source", pa.string())])),
        ]
    )
    return pa.Table.from_pydict(
        {"prompt": prompts, "images": images, "reward_model": rewards, "extra_info": extras},
        schema=schema,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=Path("/tmp/smoke.parquet"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    table = build_table()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.output, compression="snappy")
    size_kb = args.output.stat().st_size / 1024
    print(f"wrote {args.output} ({table.num_rows} rows, {size_kb:.1f} KiB)")
    print()
    print("columns:", table.column_names)
    for i, row in enumerate(table.to_pylist()):
        print(
            f"  row {i}: data_source={row['extra_info']['data_source']:>12}  "
            f"ground_truth={row['reward_model']['ground_truth']!r:<25}  "
            f"image_bytes={len(row['images'][0])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
