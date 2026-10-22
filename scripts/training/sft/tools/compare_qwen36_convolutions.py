# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Compare canonical-token GDN convolution probes and CPU depthwise
references."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def reference_dw(reference: dict) -> Any:
    import torch

    x, dy, weight = (reference[k].double() for k in ("x", "dy", "weight"))
    width = weight.shape[-1]
    result = torch.zeros_like(weight)
    for sequence, (start, end) in enumerate(zip(reference["cu"][:-1], reference["cu"][1:], strict=True)):
        segment = x[0, start:end]
        initial = reference["initial_state"]
        history = initial[sequence].double().T[-(width - 1) :] if initial is not None and width > 1 else None
        if history is None:
            history = torch.zeros(width - 1, x.shape[-1], dtype=torch.float64)
        padded = torch.cat((history, segment))
        windows = [padded[k : k + len(segment)] for k in range(width)]
        pre = sum(value * weight[:, k] for k, value in enumerate(windows))
        if reference["bias"] is not None:
            pre = pre + reference["bias"].double()
        gradient = dy[0, start:end]
        if reference["activation"] in ("silu", "swish"):
            sigmoid = pre.sigmoid()
            gradient = gradient * sigmoid * (1 + pre * (1 - sigmoid))
        elif reference["activation"] is not None:
            raise ValueError(f"Unsupported reference activation: {reference['activation']}")
        for k, value in enumerate(windows):
            result[:, k] += (value * gradient).sum(0)
    return result


def _stats(base: Any, candidate: Any) -> dict:
    import torch

    x, y = base.double().reshape(-1), candidate.double().reshape(-1)
    if x.shape != y.shape or not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("Shape or nonfinite mismatch")
    return {
        "base_sq": float(x.square().sum()),
        "candidate_sq": float(y.square().sum()),
        "error_sq": float((x - y).square().sum()),
        "dot": float((x * y).sum()),
        "max_abs": float((x - y).abs().max()) if x.numel() else 0,
        "coordinates": x.numel(),
    }


def compare(folder: Path, candidate: str = "chunkflow") -> None:
    import torch

    torch.set_num_threads(1)
    rows, issues = [], []
    state_checks = []
    for rank in range(8):
        datasets = []
        for arm in ("baseline", candidate):
            path = folder / arm / "diagnostics" / f"conv_r0_rank{rank}.pt"
            if not path.exists():
                issues.append(f"Missing snapshot: {path}")
                datasets.append(None)
                continue
            datasets.append(torch.load(path, map_location="cpu", weights_only=True))
        if any(d is None for d in datasets):
            continue
        flattened = []
        for arm, data in zip(("baseline", candidate), datasets, strict=True):
            samples, raw_weights, coverage, channels = {}, {}, defaultdict(list), {}
            groups = defaultdict(list)
            for call in data["calls"]:
                layer, parent, values = call["layer"], call["parent"], call["values"]
                key = (layer, parent)
                coverage[key].extend(tuple(span) for span in call["ranges"])
                if layer in channels and channels[layer] != call["channels"]:
                    issues.append(f"Channel coordinates changed: {arm} rank{rank} layer{layer}")
                channels[layer] = call["channels"]
                for field in ("x", "y", "dy", "dx"):
                    if field not in values:
                        issues.append(f"Missing {field}: {arm} rank{rank} {key}")
                        continue
                    for position, value in zip(call["rows"], values[field], strict=True):
                        token = (layer, parent, field, position)
                        if token in samples:
                            issues.append(f"Duplicate token/call: {arm} rank{rank} {token}")
                        samples[token] = value
                if "dw" not in values:
                    issues.append(f"Missing dw: {arm} rank{rank} {key}")
                else:
                    raw_weights[key] = (
                        raw_weights.get(key, torch.zeros_like(values["dw"], dtype=torch.float64))
                        + values["dw"].double()
                    )
                reference = call["reference"]
                if reference is not None and "dy" in reference and "dw" in values:
                    expected = reference_dw(reference)
                    rows.append(
                        {
                            "comparison": arm + "_cpu_reference",
                            "rank": rank,
                            "layer": layer,
                            "field": "dw_reference",
                            **_stats(expected, values["dw"]),
                        }
                    )
                if call["group"] is not None:
                    groups[layer, parent, call["group"]].append(call)
            for key, calls in groups.items():
                calls.sort(key=lambda c: c["group_index"])
                for before, after in zip(calls, calls[1:]):
                    if after["group_index"] != before["group_index"] + 1:
                        issues.append(f"State group coverage gap: {arm} rank{rank} {key}")
                        continue
                    for source, target in (("hT", "h0"), ("dhT", "dh0")):
                        x, y = before["values"].get(source), after["values"].get(target)
                        if x is None or y is None:
                            issues.append(f"Missing state edge {source}/{target}: {arm} rank{rank} {key}")
                        else:
                            state_checks.append(
                                {"arm": arm, "rank": rank, "layer": key[0], "field": source, **_stats(x, y)}
                            )
            for key, spans in coverage.items():
                spans.sort()
                merged = []
                for start, end in spans:
                    if merged and start < merged[-1][1]:
                        issues.append(f"Overlapping convolution calls: {arm} rank{rank} {key}")
                    if merged and start == merged[-1][1]:
                        merged[-1][1] = end
                    else:
                        merged.append([start, end])
                coverage[key] = merged
            flattened.append((samples, raw_weights, coverage, channels))
        b, c = flattened
        if b[2] != c[2] or b[3] != c[3]:
            issues.append(f"Canonical coverage/channel mismatch at rank{rank}; rank comparison skipped")
            continue
        if b[0].keys() != c[0].keys():
            issues.append(f"Unmatched canonical samples at rank{rank}: {len(b[0].keys() ^ c[0].keys())}")
        grouped = defaultdict(list)
        for key in b[0].keys() & c[0].keys():
            grouped[key[0], key[2]].append((b[0][key], c[0][key]))
        for (layer, field), pairs in grouped.items():
            rows.append(
                {
                    "comparison": candidate,
                    "rank": rank,
                    "layer": layer,
                    "field": field,
                    **_stats(torch.stack([v[0] for v in pairs]), torch.stack([v[1] for v in pairs])),
                }
            )
        if b[1].keys() != c[1].keys():
            issues.append(f"Unmatched dw parent groups at rank{rank}")
        for layer in sorted(datasets[0]["final"].keys() | datasets[1]["final"].keys()):
            x, y = (d["final"].get(layer) for d in datasets)
            if x is None or y is None:
                issues.append(f"Missing finalized dw at rank{rank} layer{layer}")
                continue
            rows.append({"comparison": candidate, "rank": rank, "layer": layer, "field": "final_dw", **_stats(x, y)})
            parents = sorted(key for key in b[1].keys() & c[1].keys() if key[0] == layer)
            if parents:
                rows.append(
                    {
                        "comparison": candidate,
                        "rank": rank,
                        "layer": layer,
                        "field": "raw_dw_sum",
                        **_stats(
                            torch.stack([b[1][key] for key in parents]), torch.stack([c[1][key] for key in parents])
                        ),
                    }
                )
    suffix = "convolutions" if candidate == "chunkflow" else "convolutions-baseline-repeat"
    if rows:
        with (folder / f"{suffix}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# GDN convolution diagnosis",
        "",
        f"Comparison: baseline vs {candidate}.",
        "x/y/dy/dx: sampled canonical token positions and Q/K/V channels; padding excluded.",
        "raw_dw_sum: sum unit contributions per parent; final_dw: full rank-local finalized convolution gradient.",
        "CPU reference: full time axis for selected channels at layer 15, including padding and initial states.",
        "CPU double-precision reference is not bitwise equivalent to fused BF16/Triton arithmetic.",
        "Differences in x or dy prevent blaming convolution backward from dw alone. All timings include diagnostic overhead.",
        "",
        "| Comparison | Layer | Field | Relative L2 | Cosine | Max absolute difference |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    groups = defaultdict(list)
    for row in rows:
        groups[row["comparison"], row["layer"], row["field"]].append(row)
    for (arm, layer, field), subset in sorted(groups.items()):
        bn, cn, error, dot = (sum(r[k] for r in subset) for k in ("base_sq", "candidate_sq", "error_sq", "dot"))
        rel = f"{math.sqrt(error / bn):.6g}" if bn else "undefined"
        cosine = f"{dot / math.sqrt(bn * cn):.6g}" if bn * cn else "undefined"
        lines.append(f"| {arm} | {layer} | {field} | {rel} | {cosine} | {max(r['max_abs'] for r in subset):.6g} |")
    lines += [
        "",
        "## Cross-chunk state continuity",
        "",
        f"Checked state edges: {len(state_checks)}; max absolute mismatch: {max((r['max_abs'] for r in state_checks), default=0):.6g}.",
        "State checks compare each arm's adjacent chunks, not baseline and ChunkFlow state layouts.",
    ]
    if candidate == "chunkflow" and not state_checks:
        issues.append("No cross-chunk state edges checked")
    lines += ["", "## Coverage problems", *[f"- {issue}" for issue in issues]]
    if not rows:
        lines.append("No comparable convolution data; this is not a passing result.")
    (folder / f"{suffix}.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--candidate", choices=("chunkflow", "baseline_repeat"), default="chunkflow")
    args = parser.parse_args()
    compare(args.folder, args.candidate)
