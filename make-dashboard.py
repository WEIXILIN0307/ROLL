"""Create a Grafana dashboard for one ROLL execution in RL-Insight Tempo."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--project", required=True)
p.add_argument("--execution", required=True)
p.add_argument("--tempo-uid", default="tempo")
p.add_argument("--from-utc")
p.add_argument("--to-utc")
p.add_argument("--metric", action="append", help="Driver metric, repeat to select multiple")
p.add_argument("--output", type=Path, default=Path("grafana-roll-rlvr.json"))
a = p.parse_args()

ds = {"type": "tempo", "uid": a.tempo_uid}

def metric_panel(i, metric, label, unit):
    query = (
        '{resource.service.name = "rl_insight_monitor" && span."monitor.trace_segment" = "metric_log"'
        ' && span.project =~ `^${project:regex}$` && span.experiment_name =~ `^${execution:regex}$`'
        f' && span."metric.name" = {json.dumps(metric)}'
        ' && span.global_step != nil && span."metric.value" != nil}'
        ' | select(span.global_step, span."metric.value")'
    )
    return {
        "id": i, "type": "barchart", "title": label,
        "description": f'Tempo metric_log spans for {metric}. The X axis is the original ROLL global_step.',
        "datasource": ds,
        "gridPos": {"x": (i-2) % 2 * 12, "y": 3 + (i-2)//2 * 14, "w": 12, "h": 14},
        "targets": [{
            "datasource": ds, "refId": "A", "queryType": "traceql",
            "tableType": "spans", "query": query, "limit": 1000, "spss": 100,
        }],
        "transformations": [
            {"id": "convertFieldType", "options": {"conversions": [
                {"targetField": field, "destinationType": "number"}
                for field in ("global_step", "metric.value")
            ]}},
            {"id": "sortBy", "options": {"sort": [{"field": "global_step", "desc": False}]}},
            {"id": "convertFieldType", "options": {"conversions": [
                {"targetField": "global_step", "destinationType": "string"}
            ]}},
            {"id": "filterFieldsByName", "options": {"include": {"pattern": r"^(global_step|metric\.value)$"}}},
        ],
        "fieldConfig": {
            "defaults": {"unit": unit, "displayName": metric, "color": {"mode": "palette-classic"},
                         "custom": {"lineWidth": 0, "fillOpacity": 76}},
            "overrides": [{"matcher": {"id": "byName", "options": "global_step"},
                           "properties": [{"id": "unit", "value": "none"},
                                          {"id": "decimals", "value": 0},
                                          {"id": "displayName", "value": "global_step"}]}],
        },
        "options": {"xField": "global_step", "orientation": "vertical",
                    "groupWidth": 0.62, "barWidth": 0.83, "tooltip": {"mode": "single"},
                    "legend": {"showLegend": True, "displayMode": "list", "placement": "bottom"}},
    }


def utc(value, default):
    if value is None:
        return default
    return datetime.fromisoformat(value).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def var(name, label, value):
    return {"type": "textbox", "name": name, "label": label, "query": value,
            "current": {"text": value, "value": value, "selected": True}}


panels = [{
    "id": 1, "type": "text",
    "gridPos": {"x": 0, "y": 0, "w": 24, "h": 3},
    "options": {"mode": "markdown", "content":
                "**X axis · original ROLL training step**\n\n"
                "For the checkpoint-2 acceptance run, steps **0–2** are replayed and steps **3–4** are new. "
                "The panels query RL-Insight's Tempo spans for the selected execution."},
}]
metrics = [(m, m, "none") for m in a.metric] if a.metric else [
    ("time/step_generate", "Rollout generation time", "s"),
    ("actor/lr", "Actor learning rate", "none"),
]
panels.extend(metric_panel(i, *m) for i, m in enumerate(metrics, 2))
a.output.write_text(json.dumps({
    "id": None, "uid": "roll-rlvr-resume", "schemaVersion": 41, "version": 1,
    "title": "ROLL text RLVR • Checkpoint restart", "timezone": "utc",
    "tags": ["roll", "rl-insight", "rlvr"], "refresh": "", "panels": panels,
    "time": {"from": utc(a.from_utc, "now-1h"), "to": utc(a.to_utc, "now")},
    "templating": {"list": [var("project", "Project", a.project), var("execution", "Execution", a.execution)]},
}, indent=2) + "\n")
