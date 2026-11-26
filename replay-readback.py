"""Re-import the recorded Grafana series into a local Tempo OTLP/HTTP receiver."""
import argparse
import json
from pathlib import Path
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--otlp-url", default="http://127.0.0.1:4318/v1/traces")
p.add_argument("--project", default="roll-529-figure-replay")
p.add_argument("--execution", default=f"recorded-resume-{uuid4().hex[:8]}")
p.add_argument("--input", type=Path, default=Path(__file__).with_name("readback.json"))
a = p.parse_args()
if urlsplit(a.otlp_url).hostname not in ("127.0.0.1", "localhost", "::1"):
    p.error("--otlp-url must point to a local OTLP/HTTP receiver")

series = json.loads(a.input.read_text())["series"]
begin = time.time_ns() - 1_000_000_000


def attr(key, field, value):
    return {"key": key, "value": {field: value}}


def span(i, point):
    start = begin + i * 10_000_000
    return {
        "traceId": uuid4().hex, "spanId": uuid4().hex[:16],
        "name": point["metric"], "kind": 1,
        "startTimeUnixNano": str(start), "endTimeUnixNano": str(start + 1_000_000),
        "attributes": [
            attr("monitor.trace_segment", "stringValue", "metric_log"),
            attr("project", "stringValue", a.project),
            attr("experiment_name", "stringValue", a.execution),
            attr("global_step", "intValue", str(point["step"])),
            attr("metric.name", "stringValue", point["metric"]),
            attr("metric.value", "doubleValue", point["value"]),
        ],
    }


body = {"resourceSpans": [{
    "resource": {"attributes": [attr("service.name", "stringValue", "rl_insight_monitor")]},
    "scopeSpans": [{"scope": {"name": "roll.figure-replay"},
                    "spans": [span(i, point) for i, point in enumerate(series)]}],
}]}
req = Request(a.otlp_url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
with urlopen(req, timeout=10) as resp:
    result = json.load(resp)
if result.get("partialSuccess", {}).get("rejectedSpans", 0):
    raise RuntimeError(result)
print(json.dumps({"project": a.project, "execution": a.execution, "points": len(series)}))
