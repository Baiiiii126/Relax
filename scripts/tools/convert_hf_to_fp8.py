# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Convert HF safetensors checkpoints to FP8 (block / channel / tensor
strategies).

Shards are quantized in parallel via a thread pool. The output ``config.json``
gets a ``quantization_config`` block written in either the fp8/e4m3 layout
(block/tensor) or the compressed-tensors layout (channel). Non-quantizable
modules (layernorm, embed, router, lm_head, …) are passed through and recorded
in ``modules_to_not_convert`` / ``ignore`` so downstream loaders skip them.
"""

import argparse
import gc
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import safetensors
import safetensors.torch
import torch
from tqdm import tqdm

from relax.utils.logging_utils import get_logger
from relax.utils.quant_cast.fp8 import build_quantization_config, quantize_hf_tensor, validate_fp8_options
from relax.utils.quant_cast.fp8_checkpoint import ConversionResult, write_safetensors_index


logger = get_logger(__name__)


def _process_file(
    input_path: str,
    output_path: str,
    filename: str,
    strategy: str,
    block_size: list[int] | None,
    result_collector: ConversionResult,
) -> None:
    if not filename.endswith(".safetensors"):
        return

    logger.info(f"Processing {filename}, memory usage: {torch.cuda.memory_allocated()}")
    q_weights: dict[str, torch.Tensor] = {}
    modules_to_not_convert: list[str] = []

    with safetensors.safe_open(os.path.join(input_path, filename), framework="pt", device="cuda") as f:
        for k in f.keys():
            converted, ignored_modules = quantize_hf_tensor(k, f.get_tensor(k), strategy, block_size)
            q_weights.update(converted)
            modules_to_not_convert.extend(ignored_modules)

    safetensors.torch.save_file(q_weights, os.path.join(output_path, filename), metadata={"format": "pt"})

    result_collector.add_result(filename, q_weights, modules_to_not_convert)


def convert_fp8(
    input_path: str,
    output_path: str,
    strategy: str,
    block_size: list[int] | None = None,
    max_workers: int = 4,
    scale_fmt: str | None = None,
) -> None:
    validate_fp8_options(strategy, block_size)
    input_path = os.path.abspath(input_path)
    os.makedirs(output_path, exist_ok=True)

    for filename in os.listdir(input_path):
        if not filename.endswith(".safetensors") and not os.path.isdir(os.path.join(input_path, filename)):
            shutil.copyfile(os.path.join(input_path, filename), os.path.join(output_path, filename))

    safetensors_files = [f for f in os.listdir(input_path) if f.endswith(".safetensors")]

    result_collector = ConversionResult()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for filename in safetensors_files:
            future = executor.submit(
                _process_file, input_path, output_path, filename, strategy, block_size, result_collector
            )
            futures.append(future)

        for future in tqdm(futures, desc="Processing files"):
            future.result()

    quantization_config = build_quantization_config(
        strategy,
        block_size,
        result_collector.modules_to_not_convert,
        scale_fmt,
    )

    config_path = Path(input_path) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        cfg["quantization_config"] = quantization_config
        with open(Path(output_path) / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)

    write_safetensors_index(output_path, result_collector)

    gc.collect()
    torch.cuda.empty_cache()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, help="Path to the directory of the HF safetensors model.")
    parser.add_argument("--save-dir", type=str, help="Path to the directory to save the converted model.")
    parser.add_argument("--strategy", type=str, default="block", choices=["block", "channel", "tensor"])
    parser.add_argument("--block-size", type=int, nargs="*", default=None, help="eg. --block-size 128 128")
    parser.add_argument("--max-workers", type=int, default=1, help="Number of worker threads for parallel processing")
    parser.add_argument("--scale-fmt", type=str, default=None, choices=["ue8m0"])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if not os.path.exists(args.save_dir):
        logger.info(f"Creating directory {args.save_dir}")
        os.makedirs(args.save_dir)
    elif not os.path.isdir(args.save_dir):
        raise ValueError("The save_dir should be a directory.")

    convert_fp8(args.model_dir, args.save_dir, args.strategy, args.block_size, args.max_workers, args.scale_fmt)


if __name__ == "__main__":
    main()
