"""Builds docker/grafana/dashboards/credit-scoring.json (provisioned as code, so the dashboard redeploys unchanged).

  uv run python docker/grafana/build_dashboard.py

Why the queries look the way they do: the simulated weeks are LABELS on the metrics (`week="5"`), not timestamps, so
the weekly panels use instant queries in table format and plot by the week label (Trend panels, a numeric x axis). Weeks are zero-padded in the query
(`05`, `10`) so they sort as numbers. Label-dependent panels (performance, slices) already show week N-1 at run N,
because the DAG pushes them under the week they describe: they appear one week late by construction.
"""
import json
from pathlib import Path

OUT = Path(__file__).parent / "dashboards" / "credit-scoring.json"
DS = {"type": "prometheus", "uid": "prometheus"}
GREEN, YELLOW, ORANGE, RED = "green", "yellow", "orange", "red"
BASELINE_AUC_PR = 0.246  # the champion's validation AUC-PR; re-run this script when a new champion is promoted


def padded(expr: str) -> str:
    """Zero-pad single-digit weeks so '10' does not sort before '2'."""
    return f'label_replace({expr}, "week", "0$1", "week", `^(\\d)$`)'


def table_target(expr: str, ref: str) -> dict:
    return {"refId": ref, "datasource": DS, "expr": expr, "instant": True, "range": False, "format": "table"}


def steps(*pairs) -> dict:
    """thresholds from (color, value) pairs; the first value is the base (None)."""
    return {"mode": "absolute", "steps": [{"color": c, "value": v} for c, v in pairs]}


def rename(ref: str, name: str, single: bool = False) -> dict:
    """With a single query the value field is called "Value"; with several it is "Value #A", "Value #B"..."""
    field = "Value" if single else f"Value #{ref}"
    return {"matcher": {"id": "byName", "options": field}, "properties": [{"id": "displayName", "value": name}]}


def row(title: str, y: int) -> dict:
    return {"type": "row", "title": title, "collapsed": False, "gridPos": {"h": 1, "w": 24, "x": 0, "y": y}, "panels": []}


def text(title: str, body: str, x: int, y: int, w: int, h: int) -> dict:
    return {"type": "text", "title": title, "gridPos": {"x": x, "y": y, "w": w, "h": h},
            "options": {"mode": "markdown", "content": body}}


def stat(title: str, expr: str, x: int, y: int, w: int, h: int, thresholds: dict, decimals: int | None = None,
         desc: str = "") -> dict:
    return {"type": "stat", "title": title, "description": desc, "datasource": DS, "gridPos": {"x": x, "y": y, "w": w, "h": h},
            "targets": [{"refId": "A", "datasource": DS, "expr": expr, "instant": True, "range": False}],
            "fieldConfig": {"defaults": {"thresholds": thresholds, "decimals": decimals, "noValue": "0",
                                         "color": {"mode": "thresholds"}}, "overrides": []},
            "options": {"colorMode": "background", "graphMode": "none", "justifyMode": "center", "textMode": "value",
                        "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False}}}


def weekly_lines(title: str, series: list[tuple[str, str]], x: int, y: int, w: int, h: int, thresholds: dict | None = None,
                unit: str = "none", decimals: int | None = 3, minimum: float | None = 0, desc: str = "",
                colors: list[str] | None = None) -> dict:
    """One line per series, the week number on the x axis (a Trend panel: numeric x, not time). `series` is a list of
    (metric expression, legend)."""
    targets = [table_target(padded(f"sum by (week) ({expr})"), chr(65 + i)) for i, (expr, _) in enumerate(series)]
    overrides = [rename(chr(65 + i), legend, single=len(series) == 1) for i, (_, legend) in enumerate(series)]
    if colors:
        overrides += [{"matcher": {"id": "byName", "options": legend},
                       "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": colors[i]}}]}
                      for i, (_, legend) in enumerate(series)]
    defaults = {"unit": unit, "decimals": decimals, "min": minimum,
                "custom": {"drawStyle": "line", "lineWidth": 2, "fillOpacity": 0, "showPoints": "always", "pointSize": 7,
                           "spanNulls": True},  # week 9 has no labels: connect across it instead of breaking the line
                "color": {"mode": "palette-classic"}}
    overrides.append({"matcher": {"id": "byName", "options": "week"},
                      "properties": [{"id": "min", "value": 1}, {"id": "max", "value": 16}, {"id": "decimals", "value": 0}]})
    if thresholds:
        defaults["thresholds"] = thresholds
        defaults["custom"]["thresholdsStyle"] = {"mode": "line"}
    return {
        "type": "trend", "title": title, "description": desc, "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": h}, "targets": targets,
        "transformations": [
            {"id": "joinByField", "options": {"byField": "week", "mode": "outer"}},
            {"id": "sortBy", "options": {"fields": {}, "sort": [{"field": "week"}]}},
            {"id": "convertFieldType", "options": {"conversions": [{"targetField": "week", "destinationType": "number"}]}},
            {"id": "filterFieldsByName", "options": {"include": {"pattern": "^(week|Value.*)$"}}},
        ],
        "fieldConfig": {"defaults": defaults, "overrides": overrides},
        "options": {"xField": "week",
                    "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
                    "tooltip": {"mode": "multi", "sort": "none"}},
    }


def slice_matrix(x: int, y: int, w: int, h: int) -> dict:
    expr = padded("slice_auc_pr_baseline - slice_auc_pr")
    return {
        "type": "table", "title": "Slice AUC-PR drop against the champion's own baseline (weeks across, slices down)",
        "description": ("Each cell = the slice's validation AUC-PR minus its AUC-PR over the last two labelled weeks. "
                        "Green is fine, red is a clear drop. Slices are fixed cut points from the training window; "
                        "the injected segment (age <= 45, income <= median) is not hand-picked into any slice."),
        "datasource": DS, "gridPos": {"x": x, "y": y, "w": w, "h": h}, "targets": [table_target(expr, "A")],
        "transformations": [
            {"id": "groupingToMatrix", "options": {"columnField": "week", "rowField": "slice", "valueField": "Value",
                                                    "emptyValue": "null"}},
        ],
        "fieldConfig": {"defaults": {"decimals": 3, "custom": {"cellOptions": {"type": "color-background", "mode": "basic"},
                                                             "align": "center", "width": 72},
                                     "thresholds": steps((GREEN, None), (YELLOW, 0.03), (ORANGE, 0.06), (RED, 0.09)),
                                     "color": {"mode": "thresholds"}},
                        "overrides": [{"matcher": {"id": "byName", "options": "slice\\week"},
                                       "properties": [{"id": "custom.width", "value": 210}]}]},
        "options": {"showHeader": True, "cellHeight": "sm"},
    }


def feature_matrix(x: int, y: int, w: int, h: int) -> dict:
    """PSI of every monitored feature, features down and weeks across: which feature moved, and when."""
    expr = padded("sum by (week, feature) (drift_feature_psi)")
    return {
        "type": "table", "title": "PSI by feature and week (scroll for all 32 features)",
        "description": ("Each cell is one feature's PSI against the champion's training window for that week. Green is under "
                        "0.10 (stable), orange 0.10 to 0.25 (moderate), red above 0.25 (major). A week with no cell was blocked "
                        "by the contract. Some injected shifts, like employment, stay under 0.10."),
        "datasource": DS, "gridPos": {"x": x, "y": y, "w": w, "h": h}, "targets": [table_target(expr, "A")],
        "transformations": [
            {"id": "groupingToMatrix", "options": {"columnField": "week", "rowField": "feature", "valueField": "Value",
                                                    "emptyValue": "null"}},
            {"id": "sortBy", "options": {"fields": {}, "sort": [{"field": "08", "desc": True}]}},
        ],
        "fieldConfig": {"defaults": {"decimals": 2, "custom": {"cellOptions": {"type": "color-background", "mode": "basic"},
                                                             "align": "center", "width": 40},
                                     "thresholds": steps((GREEN, None), (ORANGE, 0.10), (RED, 0.25)),
                                     "color": {"mode": "thresholds"}},
                        "overrides": [{"matcher": {"id": "byName", "options": "feature\\week"},
                                       "properties": [{"id": "custom.width", "value": 185}]}]},
        "options": {"showHeader": True, "cellHeight": "sm"},
    }


def colored_table(title: str, expr: str, columns: list[str], x: int, y: int, w: int, h: int, color_field: str,
                  mapping: dict[str, str], desc: str = "") -> dict:
    mappings = [{"type": "value",
                 "options": {k: {"color": c, "index": i} for i, (k, c) in enumerate(mapping.items())}}]
    return {
        "type": "table", "title": title, "description": desc, "datasource": DS, "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [table_target(padded(expr), "A")],
        "transformations": [
            {"id": "sortBy", "options": {"fields": {}, "sort": [{"field": "week"}]}},
            {"id": "organize", "options": {
                "excludeByName": {"Time": True, "Value": True, "job": True, "stage": True},
                "indexByName": {c: i for i, c in enumerate(columns)}, "renameByName": {}}},
        ],
        "fieldConfig": {"defaults": {"custom": {"align": "left"}}, "overrides": [
            {"matcher": {"id": "byName", "options": color_field},
             "properties": [{"id": "mappings", "value": mappings},
                            {"id": "custom.cellOptions", "value": {"type": "color-text"}}]}]},
        "options": {"showHeader": True, "cellHeight": "sm"},
    }


def timeseries(title: str, expr: str, legend: str, x: int, y: int, w: int, h: int, unit: str = "none", desc: str = "") -> dict:
    return {
        "type": "timeseries", "title": title, "description": desc, "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [{"refId": "A", "datasource": DS, "expr": expr, "legendFormat": legend, "range": True}],
        "fieldConfig": {"defaults": {"unit": unit, "custom": {"lineWidth": 2, "fillOpacity": 12}}, "overrides": []},
        "options": {"legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
                    "tooltip": {"mode": "multi", "sort": "none"}},
    }


def build() -> dict:
    psi_lines = steps((GREEN, None), (ORANGE, 0.10), (RED, 0.25))
    panels = [
        text("How to read this dashboard",
             "**Three acts, top to bottom:** is the data valid (the contract), did the inputs move, did the model get worse. "
             "The last row, collapsed, holds the slices, the diagnosis and the alerts. "
             "**Weeks are simulated** (1 to 16) and are plotted by week number on the x axis, not as time. Inputs and score describe "
             "week N the day it runs. **Performance and slices need labels, which arrive a week late**, so they show week N "
             "only after week N+1 has run (week 9's labels never arrive: its batch was blocked). Alerts repeat from week 5 "
             "because the drift baseline stays the champion's training window until a new model is promoted.",
             0, 0, 24, 4),
        stat("Weeks monitored", "count(batch_rows)", 0, 3, 4, 4, steps((GREEN, None)), 0),
        stat("Batches blocked by the contract", "sum(batch_blocked)", 4, 3, 4, 4, steps((GREEN, None), (RED, 1)), 0),
        stat("Worst drift (max PSI)", "max(drift_max_psi)", 8, 3, 4, 4, psi_lines, 2,
             "Worst monitored-feature PSI in any week. Under 0.10 none, 0.10 to 0.25 moderate, above 0.25 major."),
        stat("Worst AUC-PR drop", "max(model_auc_pr_drop)", 12, 3, 4, 4, steps((GREEN, None), (ORANGE, 0.04)), 3,
             "Largest gap between the champion's validation AUC-PR and a week's AUC-PR. The alert fires at 0.04."),
        stat("Alerts raised", "sum(alert_firing)", 16, 3, 4, 4, steps((GREEN, None), (ORANGE, 1)), 0),
        stat("Applicants scored by the server", "sum(predict_rows_total)", 20, 3, 4, 4, steps((GREEN, None)), 0),

        row("Inputs and score (describe week N, no labels needed)", 7),
        weekly_lines("Drift severity by week (PSI)", [("drift_max_psi", "worst feature"), ("drift_mean_psi", "mean of monitored")],
                    0, 8, 8, 8, psi_lines, decimals=2, colors=["red", "blue"],
                    desc="PSI against the champion's training window. The lines mark 0.10 (moderate) and 0.25 (major)."),
        weekly_lines("Monitored features past the limits", [("drift_n_psi_moderate", "PSI >= 0.10"), ("drift_n_psi_major", "PSI >= 0.25")],
                    8, 8, 8, 8, decimals=0, colors=["orange", "red"],
                    desc="Of the 32 monitored features. Two at >= 0.10 raises a moderate alert; one at >= 0.25 a major one."),
        weekly_lines("Score drift", [("drift_score_psi", "score PSI"), ("drift_score_ks", "score KS")], 16, 8, 8, 8,
                    steps((GREEN, None), (ORANGE, 0.10)), decimals=3, colors=["purple", "blue"],
                    desc="How far the model's own score distribution moved from its held-out baseline."),

        row("Performance (labels arrive a week late)", 16),
        weekly_lines("AUC-PR by week", [("model_auc_pr", "AUC-PR")], 0, 17, 8, 8,
                    steps(("transparent", None), (GREEN, BASELINE_AUC_PR)), decimals=3, colors=["blue"],
                    desc=f"The green line is the champion's validation AUC-PR ({BASELINE_AUC_PR}). A random model scores 0.081. "
                         "One week of 5,000 applicants is noisy (about +-0.04)."),
        weekly_lines("Brier score by week", [("model_brier", "Brier")], 8, 17, 8, 8,
                    steps(("transparent", None), (GREEN, 0.0745)), decimals=4, colors=["purple"],
                    desc="Lower is better. The line is the constant-default-rate guess (0.0745): a useful model stays below it."),
        weekly_lines("Predicted vs observed default rate", [("model_mean_pred", "predicted"), ("model_observed_rate", "observed")],
                    16, 17, 8, 8, decimals=3, minimum=0, colors=["blue", "orange"],
                    desc="Both rise after the week-5 shock (0.090 predicted, 0.088 observed) and stay within 20% of each other; a wider gap would be a calibration alert."),

        row("Where is it getting worse? (14 slices over two weeks)", 25),
        slice_matrix(0, 26, 24, 16),

        row("What happened and why", 42),
        colored_table("Diagnosed cause by week", "sum by (week, cause) (diagnosis)", ["week", "cause"], 0, 43, 8, 14, "cause",
                      {"covariate_shift": "orange", "concept_drift": "red", "technical": "purple", "label_shift": "blue",
                       "none": "green"},
                      "Week-over-week signals, not versus the Reference: features changed, score changed, performance dropped."),
        colored_table("Alerts raised", "sum by (week, rule, severity) (alert_firing)", ["week", "rule", "severity"], 8, 43, 16, 14,
                      "severity", {"critical": "red", "high": "orange", "medium": "yellow"},
                      "Each row is an alert about that week. Performance alerts appear a week after the week they describe."),

        row("Contract and serving", 57),
        weekly_lines("Contract outcome by week", [("batch_blocked", "blocked (1 = rejected)"), ("batch_contract_warnings", "warnings")],
                    0, 58, 8, 7, decimals=0, colors=["red", "yellow"]),
        timeseries("Server requests per minute", 'sum by (outcome) (rate(predict_requests_total[5m])) * 60', "{{outcome}}",
                   8, 58, 8, 7, desc="Scraped straight from the model server (the pull side of the push/pull pair)."),
        timeseries("Server latency, 95th percentile", "histogram_quantile(0.95, sum by (le) (rate(predict_latency_seconds_bucket[5m])))",
                   "p95", 16, 58, 8, 7, unit="s"),
    ]
    def find(title, kind=None):
        return next(p for p in panels if p["title"] == title and (kind is None or p["type"] == kind))

    def put(panel, x, y, w, h):
        panel["gridPos"] = {"x": x, "y": y, "w": w, "h": h}
        return panel

    # The dashboard follows the story: act 1 the contract, act 2 the inputs, act 3 the performance; the rest is collapsed.
    tiles = [put(p, p["gridPos"]["x"], p["gridPos"]["y"] + (1 if p["type"] == "stat" else 0), p["gridPos"]["w"],
                 p["gridPos"]["h"]) for p in panels if p["type"] in ("text", "stat")]
    act1 = [row("Act 1 · Is the data valid? (the contract, and the server's health)", 8),
            put(find("Contract outcome by week"), 0, 9, 8, 7),
            put(find("Server requests per minute"), 8, 9, 8, 7),
            put(find("Server latency, 95th percentile"), 16, 9, 8, 7)]
    act2 = [row("Act 2 · Did the inputs move? (no labels needed)", 16),
            put(find("Drift severity by week (PSI)"), 0, 17, 8, 8),
            put(find("Monitored features past the limits"), 8, 17, 8, 8),
            put(find("Score drift"), 16, 17, 8, 8),
            feature_matrix(0, 25, 16, 13),
            text("How to read the heat map",
                 "Rows are the monitored features, sorted by how much they moved; columns are weeks. "
                 "**Green** is under 0.10 (stable), **orange** 0.10 to 0.25, **red** above 0.25. "
                 "Scroll inside the table to see all 32 features. A week with no column was blocked by the contract (week 9). "
                 "Only four features cross 0.10: the annuity ratios, income and the annuity; the rest stay green.",
                 16, 25, 8, 13)]
    act3 = [row("Act 3 · Did the model get worse? (labels arrive a week late)", 38),
            put(find("AUC-PR by week"), 0, 39, 8, 8),
            put(find("Brier score by week"), 8, 39, 8, 8),
            put(find("Predicted vs observed default rate"), 16, 39, 8, 8)]
    details = row("Details: where it got worse, why, and which alerts fired (click to open)", 47)
    details["collapsed"] = True
    details["panels"] = [
        put(find("Slice AUC-PR drop against the champion's own baseline (weeks across, slices down)"), 0, 48, 24, 16),
        put(find("Diagnosed cause by week"), 0, 64, 8, 14),
        put(find("Alerts raised", "table"), 8, 64, 16, 14)]
    panels = tiles + act1 + act2 + act3 + [details]
    n = 0
    for panel in panels:
        for q in [panel, *panel.get("panels", [])]:
            n += 1
            q["id"] = n
    return {
        "uid": "credit-scoring-monitoring", "title": "Credit scoring: drift and reliability", "tags": ["credit-drift"],
        "timezone": "browser", "schemaVersion": 41, "version": 1, "editable": True, "graphTooltip": 1, "refresh": "",
        "time": {"from": "now-24h", "to": "now"}, "templating": {"list": []}, "annotations": {"list": []},
        "panels": panels,
    }


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(build()['panels'])} panels)")
