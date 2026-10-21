# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Compare canonical-token layer traces across TP placement and ChunkFlow
packing."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def compare(folder: Path) -> None:
    import torch

    issues = []

    def load(arm: str) -> dict:
        records = {}
        conflicts = set()
        paths = list((folder / arm / "diagnostics").glob("layers_r0_rank*.pt"))
        if len(paths) != 8:
            issues.append(f"{arm}: expected 8 rank files, found {len(paths)}")
        for path in paths:
            for event in torch.load(path, map_location="cpu", weights_only=True):
                for row, value in zip(event["rows"], event["values"], strict=True):
                    key = (event["layer"], event["kind"], event["parent"], row)
                    if key in records and not torch.equal(records[key], value):
                        issues.append(f"Conflicting repeated token: {arm}: {key}")
                        conflicts.add(key)
                    records[key] = value
        return {key: value for key, value in records.items() if key not in conflicts}

    base, chunk = load("baseline"), load("chunkflow")
    groups = defaultdict(list)
    missing = base.keys() ^ chunk.keys()
    if missing:
        issues.append(f"Unmatched layer/token events: {len(missing)} (sample: {sorted(missing)[:5]})")
    for key in sorted(base.keys() & chunk.keys()):
        b, c = base[key], chunk[key]
        if b.shape != c.shape or not torch.isfinite(b).all() or not torch.isfinite(c).all():
            issues.append(f"Shape/nonfinite mismatch: {key}")
            continue
        groups[key[:2]].append((b, c))
    rows = []
    for (layer, kind), values in sorted(groups.items()):
        b, c = (torch.stack([v[i] for v in values]) for i in range(2))
        if kind == "router_map":
            b, c = b.bool(), c.bool()
            bn, cn = float(b.sum()) ** 0.5, float(c.sum()) ** 0.5
            error = float((b != c).sum()) ** 0.5
            dot, maximum = float((b & c).sum()), float((b != c).any())
        else:
            b, c = b.double(), c.double()
            bn, cn = float(b.norm()), float(c.norm())
            error, dot, maximum = float((b - c).norm()), float((b * c).sum()), float((b - c).abs().max())
        rows.append(
            {
                "layer": layer,
                "event": kind,
                "tokens": len(values),
                "relative_l2": error / bn if bn else None,
                "cosine": dot / (bn * cn) if bn and cn else None,
                "max_abs": maximum,
                "changed_tokens": int((b != c).any(dim=1).sum()),
            }
        )
    if rows:
        with (folder / "layers.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# First-step layer localization",
        "",
        "Layer numbers come from Megatron layer_number (normally 1-based).",
        "Tokens are paired by parent input hash and original token position, even when TP placement changes.",
        "Layer 1 and routing maps cover all valid tokens. Layer 1 retains every feature and adds component boundaries/logits.",
        "Other layers sample 32 token positions and 64 activation/gradient features; sampled router inputs/probabilities retain all features.",
        "Small floating-point differences are not automatically bugs. Routing differences alone do not prove faulty backward.",
        "",
        "| Layer | Event | Tokens | Relative L2 | Cosine | Max absolute difference | Changed tokens |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        numbers = [
            f"{row[k]:.6g}" if row[k] is not None else "undefined" for k in ("relative_l2", "cosine", "max_abs")
        ]
        lines.append(
            f"| {row['layer']} | {row['event']} | {row['tokens']} | "
            + " | ".join(numbers)
            + f" | {row['changed_tokens']} |"
        )
    lines += ["", "## Coverage problems", *[f"- {issue}" for issue in issues]]
    if not rows:
        lines.append("No comparable events; this is not a passing result.")
    (folder / "layers.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    compare(parser.parse_args().folder)
