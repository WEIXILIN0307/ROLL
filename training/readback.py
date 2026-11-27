"""Compare saved WorkerState scalars against Tempo after both real driver exits."""
import argparse
import json
import math
from pathlib import Path
import re
import time

import requests

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run", type=Path)
parser.add_argument("--tempo-url", default="http://127.0.0.1:3200")
args = parser.parse_args()
root = args.run
tempo = args.tempo_url.rstrip("/")
output = []
with requests.Session() as client:
    client.trust_env = False
    for phase, last, new_steps in (("first", 2, (0, 1, 2)), ("resumed", 4, (3, 4))):
        log = (root / f"{phase}.log").read_text()
        executed = tuple(int(i) for i in re.findall(r"\] pipeline step (\d+) start", log))
        assert executed == new_steps and "pipeline complete!" in log, executed
        name, project = re.search(
            r"RL-Insight experiment: (\S+) \(project: (\S+)\)", log
        ).groups()
        ckpt = Path(re.search(r"add timestamp to output_dir (.*)", log)[1]) / f"checkpoint-{last}"
        state = json.loads((ckpt / "pipeline" / "worker_state_pipeline.json").read_text())
        assert state["step"] == last
        assert [m["system/step"] for m in state["log_history"]] == list(range(last + 1))
        points = []
        for metrics in state["log_history"]:
            step = metrics["system/step"]
            names = {key.rsplit("@", 1)[0]: value for key, value in metrics.items()}
            expected = {key: value for key, value in names.items() if type(value) in (int, float) and math.isfinite(value)}
            query = (
                '{resource.service.name = "rl_insight_monitor"'
                f' && span.project = "{project}" && span.experiment_name = "{name}" && span.global_step = {step}'
                ' && span."monitor.trace_segment" = "metric_log"}'
                ' | select(span.global_step, span."metric.name", span."metric.value")'
            )
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                response = client.get(f"{tempo}/api/search", params={"q": query, "limit": len(expected) + 10}, timeout=5)
                response.raise_for_status()
                found = {}
                for trace in response.json().get("traces") or []:
                    for group in trace["spanSets"]:
                        for span in group["spans"]:
                            tags = {a["key"]: a["value"] for a in span["attributes"]}
                            assert int(tags["global_step"]["intValue"]) == step
                            found[tags["metric.name"]["stringValue"]] = tags["metric.value"]["doubleValue"]
                if found == expected:
                    break
                time.sleep(0.5)
            else:
                raise AssertionError(
                    f"{phase} step {step}: missing {expected.keys() - found.keys()}, unexpected {found.keys() - expected.keys()},"
                    f" different {[k for k in expected.keys() & found.keys() if expected[k] != found[k]]}"
                )
            points.append({"step": step, "count": len(found), "examples": {
                k: v for k, v in found.items() if k in (
                    "system/step", "sft_train/loss", "sft_train/grad_norm", "actor_train/grad_norm", "actor/pg_loss"
                )
            }})
        output.append({"phase": phase, "execution": name, "checkpoint": ckpt.name, "points": points})
assert output[0]["execution"] != output[1]["execution"]
print(json.dumps(output, indent=2))
