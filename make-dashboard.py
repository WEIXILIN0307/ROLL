"""Create a Grafana dashboard for one ROLL execution in RL-Insight Tempo."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--kind", default="sft", choices=("sft", "vision"))
p.add_argument("--uid", help="Override the dashboard UID, e.g. for an offline replay")
p.add_argument("--project", required=True)
p.add_argument("--execution", required=True)
p.add_argument("--tempo-uid", default="tempo")
p.add_argument("--from-utc")
p.add_argument("--to-utc")
p.add_argument("--checkpoint-step", type=int, help="Color steps through this checkpoint as replayed")
p.add_argument("--metric", action="append", help="Driver metric, repeat to select multiple")
p.add_argument("--output", type=Path, default=Path("grafana-roll-rlvr.json"))
a = p.parse_args()
if a.checkpoint_step is not None and a.checkpoint_step < 0:
    p.error("--checkpoint-step must be non-negative")

ds = {"type": "tempo", "uid": a.tempo_uid}
replayed_color = "#2563EB"
new_color = "#BA4A00"


def step_field_color():
    if a.checkpoint_step is None:
        return []
    return [{"id": "mappings", "value": [
        {"type": "range", "options": {"from": 0, "to": a.checkpoint_step,
                                      "result": {"color": replayed_color, "index": 0}}},
        {"type": "range", "options": {"from": a.checkpoint_step + 1, "to": None,
                                      "result": {"color": new_color, "index": 1}}},
    ]}]


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
                         **({"decimals": 3} if metric == "sft_train/loss" else {}),
                         "custom": {"lineWidth": 0, "fillOpacity": 76,
                                    **({"axisSoftMax": 14} if metric == "sft_train/loss" else {})}},
            "overrides": [{"matcher": {"id": "byName", "options": "global_step"},
                           "properties": [{"id": "unit", "value": "none"},
                                          {"id": "decimals", "value": 0},
                                          {"id": "displayName", "value": "global_step"},
                                          *step_field_color()]}],
        },
        "options": {"xField": "global_step", "orientation": "vertical",
                    **({"colorByField": "global_step"} if a.checkpoint_step is not None else {}),
                    "groupWidth": 0.62, "barWidth": 0.83, "showValue": "always", "tooltip": {"mode": "single"},
                    "legend": {"showLegend": a.checkpoint_step is None,
                               "displayMode": "list", "placement": "bottom"}},
    }


def utc(value, default):
    if value is None:
        return default
    return datetime.fromisoformat(value).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def var(name, label, value):
    return {"type": "textbox", "name": name, "label": label, "query": value,
            "current": {"text": value, "value": value, "selected": True}}


run_label = f"ROLL {'SFT' if a.kind == 'sft' else 'Vision RLVR'} · original training step"
phase_legend = ""
if a.checkpoint_step is not None:
    phase_legend = (
        f"**Blue:** checkpoint history through step {a.checkpoint_step}. "
        f"**Orange:** new training from step {a.checkpoint_step + 1}. "
    )
panels = [{
    "id": 1, "type": "text",
    "gridPos": {"x": 0, "y": 0, "w": 24, "h": 3},
    "options": {"mode": "markdown", "content":
                f"**{run_label}**\n\n{phase_legend}"
                "The panels query RL-Insight's Tempo spans for the selected execution."},
}]
title, metrics = {
    "sft": ("SFT", [("sft_train/loss", "SFT training loss", "none"),
                    ("time/step_train", "SFT batch update time", "s")]),
    "vision": ("Vision RLVR", [("time/step_generate", "Vision rollout generation time", "s"),
                              ("actor/lr", "Actor learning rate", "none")]),
}[a.kind]
if a.metric:
    metrics = [(m, m, "none") for m in a.metric]
panels.extend(metric_panel(i, *m) for i, m in enumerate(metrics, 2))
a.output.write_text(json.dumps({
    "id": None, "uid": a.uid or f"roll-{a.kind}-resume", "schemaVersion": 41, "version": 1,
    "title": f"ROLL {title} • Checkpoint restart" + (" · replay" if a.uid else ""), "timezone": "utc",
    "tags": ["roll", "rl-insight", "rlvr"], "refresh": "", "panels": panels,
    "time": {"from": utc(a.from_utc, "now-1h"), "to": utc(a.to_utc, "now")},
    "templating": {"list": [var("project", "Project", a.project), var("execution", "Execution", a.execution)]},
}, indent=2) + "\n")
