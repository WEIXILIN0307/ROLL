"""Render a reviewer table and exact TSV snapshots from checkpoint/Tempo readback."""
import csv
import json
from pathlib import Path


root = Path(__file__).resolve().parent
dashboards = root / "dashboards"
body = [
    "# Verify the restart at each original training step",
    "",
    "The eligible-scalar counts are from the resumed Tempo execution. Every point was checked against the saved",
    "checkpoint by original step, metric name, and value. The selected chart values below are rounded for reading.",
    "The linked TSV and JSON retain the values returned by Tempo.",
]
for kind, title in (("sft", "SFT"), ("vision", "Vision RLVR")):
    data = json.loads((dashboards / f"{kind}-readback.json").read_text())
    samples = data["series"]
    metrics = list(dict.fromkeys(p["metric"] for p in samples))
    rows = {}
    for point in samples:
        row = rows.setdefault(point["step"], {})
        assert point["value"] == point["checkpoint_value"]
        row[point["metric"]] = point["value"]
        row["origin"] = point["origin"]
    counts = {r["step"]: r["metric_count"] for r in data["matched_counts"] if r["phase"] == "resumed"}
    assert rows.keys() == counts.keys()
    with (dashboards / f"{kind}-readback.tsv").open("w", newline="") as file:
        tsv = csv.writer(file, delimiter="\t", lineterminator="\n")
        tsv.writerow(["step", "origin", "matched_scalars", *metrics])
        for step, row in sorted(rows.items()):
            tsv.writerow([step, row["origin"], counts[step], *(row[m] for m in metrics)])
    body += [
        "", f"## {title}", "",
        f"Execution `{data['execution']}` on ROLL `{data['roll_commit'][:7]}`. "
        f"[Exact TSV](dashboards/{kind}-readback.tsv) · [Trace samples](dashboards/{kind}-readback.json)",
        "", "| Original step | Origin | Matched scalar points | " + " | ".join(f"`{m}`" for m in metrics) + " |",
        "| ---: | --- | ---: |" + " ---: |" * len(metrics),
    ]
    for step, row in sorted(rows.items()):
        values = " | ".join(f"{row[m]:.6g}" for m in metrics)
        body.append(f"| {step} | {row['origin']} | {counts[step]} | {values} |")
    body.extend(["", f"Total in the resumed execution: **{sum(counts.values()):,}**."])

skip = json.loads((dashboards / "vision-skip-readback.json").read_text())
body.extend([
    "", "## Vision RLVR zero-valid-response run", "",
    "In a separate real rollout, the driver skipped actor training for both steps because no response tokens",
    "were valid. After exit, Tempo returned `system/step` for both original steps.",
    "", "| Step | `system/step` |", "| ---: | ---: |",
    *(f"| {step} | {value:g} |" for step, value in skip["points"]),
    "", "[Skip-path readback](dashboards/vision-skip-readback.json)", "",
])
(root / "Readback.md").write_text("\n".join(body))
