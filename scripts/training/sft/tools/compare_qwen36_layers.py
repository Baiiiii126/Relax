# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Compare canonical-token layer traces across TP placement and ChunkFlow
packing."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def compare(folder: Path, candidate: str = "chunkflow", prefix: str = "layers") -> None:
    import torch

    torch.set_num_threads(1)
    issues = []
    supervised = {}
    suffix = prefix if candidate == "chunkflow" else prefix + "-" + candidate.replace("_", "-")

    def load(arm: str) -> dict:
        records = {}
        conflicts = set()
        paths = list((folder / arm / "diagnostics").glob(f"{prefix}_r0_rank*.pt"))
        if len(paths) != 8:
            issues.append(f"{arm}: expected 8 rank files, found {len(paths)}")
        for path in paths:
            for event in torch.load(path, map_location="cpu", weights_only=True):
                for index, (row, value) in enumerate(zip(event["rows"], event["values"], strict=True)):
                    key = (event["layer"], event["kind"], event["parent"], row)
                    if "supervised" in event:
                        supervised[arm, key] = bool(event["supervised"][index])
                    if key in records and not torch.equal(records[key], value):
                        issues.append(f"Conflicting repeated token: {arm}: {key}")
                        conflicts.add(key)
                    records[key] = value
        return {key: value for key, value in records.items() if key not in conflicts}

    base, chunk = load("baseline"), load(candidate)
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
                "baseline_nonzero_tokens": int(b.ne(0).any(dim=1).sum()),
                "candidate_nonzero_tokens": int(c.ne(0).any(dim=1).sum()),
                "supervised_tokens": sum(
                    supervised.get(("baseline", key), False) and supervised.get((candidate, key), False)
                    for key in base.keys() & chunk.keys()
                    if key[:2] == (layer, kind)
                ),
            }
        )
    if prefix == "tail":
        for row in rows:
            if row["event"] == "log_probs_grad":
                if not row["baseline_nonzero_tokens"] or not row["candidate_nonzero_tokens"]:
                    issues.append(
                        "All sampled log-probability gradients are zero in at least one arm; upstream equality is inconclusive. Sample supervised answer tokens before locating an operator."
                    )
                if not row["supervised_tokens"]:
                    issues.append(
                        "No supervised-token metadata at the loss boundary; this trace cannot certify answer-token coverage."
                    )
    if rows:
        with (folder / f"{suffix}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if prefix == "tail":
        required = {
            (39, "layer_input"),
            (39, "layer_output"),
            (40, "layer_input"),
            (40, "layer_output"),
            (40, "attention_input"),
            (40, "attention_output"),
            (40, "moe_input"),
            (40, "moe_output"),
            (40, "final_norm_input"),
            (40, "final_norm_output"),
            (40, "lm_head_input_tp0"),
            (40, "lm_head_input_tp1"),
            (40, "lm_head_logits_tp0"),
            (40, "lm_head_logits_tp1"),
            (40, "log_probs"),
        }
        required.update({(40, "linear_proj_output"), (40, "linear_qkv_input")})
        for tp_rank in range(2):
            for kind in (
                "linear_proj_input",
                "linear_qkv_output",
                "q_layernorm_input",
                "q_layernorm_output",
                "k_layernorm_input",
                "k_layernorm_output",
                "core_query",
                "core_key",
                "core_value",
                "core_output",
            ):
                required.add((40, kind + f"_tp{tp_rank}"))
        for arm, events in (("baseline", base), (candidate, chunk)):
            present = {key[:2] for key in events}
            for layer, kind in sorted(required):
                for event in (kind, kind + "_grad"):
                    if (layer, event) not in present:
                        issues.append(f"Missing required boundary: {arm}: layer {layer}: {event}")
    lines = [
        "# First-step layer localization",
        "",
        f"Comparison: baseline vs {candidate}; trace: {prefix}.",
        "Layer numbers come from Megatron layer_number (normally 1-based).",
        "Tokens are paired by parent input hash and original token position, even when TP placement changes.",
        "Layer 1 and routing maps cover all valid tokens. Layer 1 retains every feature and adds component boundaries/logits.",
        "Other layers sample 32 token positions and 64 activation/gradient features; sampled router inputs/probabilities retain all features.",
        "Small floating-point differences are not automatically bugs. Routing differences alone do not prove faulty backward.",
        "",
        "| Layer | Event | Tokens | Relative L2 | Cosine | Max absolute difference | Changed tokens | Nonzero tokens B/C | Supervised tokens |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    if prefix == "tail":
        lines[5:8] = [
            "Tail probes retain all hidden features at 32 uniform parent positions plus up to 16 supervised positions per sequence; padding excluded.",
            "Head logits retain 64 fixed local-vocabulary coordinates plus the target coordinate (zero on non-owning TP rank).",
            "Head input gradients are before the dtype cast and kept separate by TP rank; logit gradients are at the CE input after temperature scaling.",
            "Read backward from log_probs_grad -> lm_head_logits_*_grad -> lm_head_input_*_grad -> final_norm_*_grad -> layer40 -> layer39.",
            "Attention/MoE residual branches mean an increase in relative error alone does not prove the first faulty operator.",
            "Compare with tail-baseline-repeat.md or tail-chunkflow-control.md. Full loss equality or matching samples do not prove full gradient equality.",
        ]
    for row in rows:
        numbers = [
            f"{row[k]:.6g}" if row[k] is not None else "undefined" for k in ("relative_l2", "cosine", "max_abs")
        ]
        lines.append(
            f"| {row['layer']} | {row['event']} | {row['tokens']} | "
            + " | ".join(numbers)
            + f" | {row['changed_tokens']} | {row['baseline_nonzero_tokens']}/{row['candidate_nonzero_tokens']} | {row['supervised_tokens']} |"
        )
    lines += ["", "## Coverage problems", *[f"- {issue}" for issue in issues]]
    if not rows:
        lines.append("No comparable events; this is not a passing result.")
    (folder / f"{suffix}.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument(
        "--candidate", choices=("chunkflow", "baseline_repeat", "chunkflow_control"), default="chunkflow"
    )
    parser.add_argument("--prefix", choices=("layers", "tail"), default="layers")
    args = parser.parse_args()
    compare(args.folder, args.candidate, args.prefix)
