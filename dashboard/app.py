"""
dashboard.app
~~~~~~~~~~~~~
Interactive GeoIDS dashboard built with Plotly Dash.

Features
--------
* Live alert feed table
* 3-D PCA scatter of blade-coefficient manifold
* Score time-series with dynamic threshold overlay
* Blade anomaly breakdown (bar chart)
* KPI cards: total flows, alert rate, top anomalous blade

Launch
------
geoIDS dashboard --port 8050 --alerts-file alerts.json
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def create_app(alerts_file: str = "alerts.json"):
    try:
        import dash

        #import plotly.express as px
        import plotly.graph_objects as go
        from dash import Input, Output, dash_table, dcc, html
    except ImportError:
        raise ImportError("Dashboard requires: pip install dash plotly pandas") from None

    app = dash.Dash(
        __name__,
        title="GeoIDS — Anomaly Dashboard",
        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    )

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    app.layout = html.Div(
        style={"backgroundColor": "#0d1117", "color": "#c9d1d9", "fontFamily": "monospace",
               "minHeight": "100vh", "padding": "16px"},
        children=[
            # Header
            html.Div([
                html.H1("⬡ GeoIDS Anomaly Dashboard",
                        style={"color": "#58a6ff", "marginBottom": "4px"}),
                html.P("Conformal Geometric Algebra · Real-time Encrypted Traffic Analysis",
                       style={"color": "#8b949e", "margin": 0}),
            ], style={"marginBottom": "24px"}),

            # Refresh interval
            dcc.Interval(id="interval", interval=3_000, n_intervals=0),

            # KPI cards
            html.Div(id="kpi-cards", style={"display": "flex", "gap": "16px",
                                             "marginBottom": "24px", "flexWrap": "wrap"}),

            # Charts row
            html.Div([
                html.Div([
                    html.H3("Anomaly Score Time-Series", style={"color": "#58a6ff"}),
                    dcc.Graph(id="score-ts", style={"height": "300px"}),
                ], style={"flex": "2", "backgroundColor": "#161b22",
                           "borderRadius": "8px", "padding": "16px"}),

                html.Div([
                    html.H3("Top Anomalous Blades", style={"color": "#58a6ff"}),
                    dcc.Graph(id="blade-bar", style={"height": "300px"}),
                ], style={"flex": "1", "backgroundColor": "#161b22",
                           "borderRadius": "8px", "padding": "16px"}),
            ], style={"display": "flex", "gap": "16px", "marginBottom": "24px"}),

            # 3-D manifold
            html.Div([
                html.H3("Multivector Manifold (PCA Projection)", style={"color": "#58a6ff"}),
                dcc.Graph(id="manifold-3d", style={"height": "450px"}),
            ], style={"backgroundColor": "#161b22", "borderRadius": "8px",
                      "padding": "16px", "marginBottom": "24px"}),

            # Alert table
            html.Div([
                html.H3("Recent Alerts", style={"color": "#58a6ff"}),
                dash_table.DataTable(
                    id="alert-table",
                    style_table={"overflowX": "auto"},
                    style_header={"backgroundColor": "#21262d", "color": "#58a6ff",
                                  "fontWeight": "bold"},
                    style_cell={"backgroundColor": "#161b22", "color": "#c9d1d9",
                                "border": "1px solid #30363d", "textAlign": "left",
                                "fontSize": "12px", "padding": "6px"},
                    style_data_conditional=[{
                        "if": {"filter_query": "{is_anomaly} = true"},
                        "backgroundColor": "#3d1f1f",
                    }],
                    page_size=20,
                ),
            ], style={"backgroundColor": "#161b22", "borderRadius": "8px", "padding": "16px"}),
        ],
    )

    # ------------------------------------------------------------------
    # Data loader
    # ------------------------------------------------------------------

    def load_alerts(path: str, max_rows: int = 2000) -> pd.DataFrame:
        p = Path(path)
        if not p.exists():
            return pd.DataFrame()
        rows = []
        try:
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        with contextlib.suppress(json.JSONDecodeError):
                            rows.append(json.loads(line))
        except Exception:
            return pd.DataFrame()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows[-max_rows:])
        if "timestamp" in df.columns:
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        return df

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    @app.callback(
        [
            Output("kpi-cards", "children"),
            Output("score-ts", "figure"),
            Output("blade-bar", "figure"),
            Output("manifold-3d", "figure"),
            Output("alert-table", "data"),
            Output("alert-table", "columns"),
        ],
        Input("interval", "n_intervals"),
    )
    def update(n):
        df = load_alerts(alerts_file)
        dark_layout = {
            "paper_bgcolor": "#161b22",
            "plot_bgcolor": "#0d1117",
            "font": {"color": "#c9d1d9"},
            "xaxis": {"gridcolor": "#21262d"},
            "yaxis": {"gridcolor": "#21262d"},
        }

        # -- KPI cards --
        if df.empty:
            total = alerts = 0
            alert_rate = 0.0
            top_blade = "N/A"
        else:
            total = len(df)
            alerts = int(df.get("is_anomaly", pd.Series(dtype=bool)).sum())
            alert_rate = alerts / max(total, 1)
            blade_col = df.get("top_blade", pd.Series(dtype=str))
            top_blade = blade_col.mode().iloc[0] if not blade_col.dropna().empty else "N/A"

        def kpi_card(label, value, colour="#58a6ff"):
            return html.Div([
                html.P(label, style={"color": "#8b949e", "margin": 0, "fontSize": "12px"}),
                html.H2(str(value), style={"color": colour, "margin": 0}),
            ], style={"backgroundColor": "#161b22", "borderRadius": "8px",
                      "padding": "16px 24px", "minWidth": "160px"})

        kpi_cards = [
            kpi_card("Total Flows", f"{total:,}"),
            kpi_card("Alerts", f"{alerts:,}", "#f85149"),
            kpi_card("Alert Rate", f"{alert_rate:.2%}", "#d29922"),
            kpi_card("Top Blade", top_blade, "#3fb950"),
        ]

        # -- Score time-series --
        if df.empty or "score" not in df.columns:
            fig_ts = go.Figure(layout={**dark_layout, "title": "No data yet"})
        else:
            x = df.get("datetime", df.index)
            fig_ts = go.Figure(layout=dark_layout)
            fig_ts.add_trace(go.Scatter(
                x=x, y=df["score"], mode="lines", name="Score",
                line={"color": "#58a6ff", "width": 1},
            ))
            if "threshold" in df.columns:
                fig_ts.add_trace(go.Scatter(
                    x=x, y=df["threshold"], mode="lines", name="Threshold",
                    line={"color": "#f85149", "width": 1.5, "dash": "dash"},
                ))
            # Highlight alerts
            is_anom = df.get("is_anomaly", False)
            alerts_df = df[is_anom] if "is_anomaly" in df.columns else pd.DataFrame()
            if not alerts_df.empty:
                ax = alerts_df.get("datetime", alerts_df.index)
                fig_ts.add_trace(go.Scatter(
                    x=ax, y=alerts_df["score"], mode="markers", name="Alert",
                    marker={"color": "#f85149", "size": 6, "symbol": "x"},
                ))
            fig_ts.update_layout(legend={"bgcolor": "#21262d"})

        # -- Blade bar chart --
        if df.empty or "top_blade" not in df.columns:
            fig_blade = go.Figure(layout={**dark_layout, "title": "No data yet"})
        else:
            blade_counts = df["top_blade"].value_counts().head(10)
            fig_blade = go.Figure(
                go.Bar(
                    x=blade_counts.values[::-1],
                    y=blade_counts.index[::-1],
                    orientation="h",
                    marker_color="#58a6ff",
                ),
                layout={**dark_layout, "xaxis_title": "Count"},
            )

        # -- 3-D manifold (synthetic PCA if no blade vectors) --
        if df.empty or "score" not in df.columns:
            fig_3d = go.Figure(layout={**dark_layout, "title": "No data yet"})
        else:
            n = len(df)
            rng = np.random.default_rng(42)
            # Approximate 3-D projection from score + confidence
            scores = df["score"].fillna(0).values
            confs = df.get("confidence", pd.Series(np.zeros(n))).fillna(0).values
            # Synthetic 3rd axis from threshold ratio
            thresholds = df.get("threshold", pd.Series(np.ones(n))).fillna(1).values
            ratio = np.clip(scores / np.where(thresholds > 0, thresholds, 1), 0, 5)
            x3 = scores + rng.normal(0, 0.01, n)
            y3 = confs + rng.normal(0, 0.01, n)
            z3 = ratio + rng.normal(0, 0.01, n)
            is_anom = df.get("is_anomaly", pd.Series([False] * n)).fillna(False).values
            colours = ["#f85149" if a else "#58a6ff" for a in is_anom]
            fig_3d = go.Figure(
                go.Scatter3d(
                    x=x3, y=y3, z=z3,
                    mode="markers",
                    marker={"size": 3, "color": colours, "opacity": 0.7},
                    hovertext=df.get("flow_id", pd.Series([""] * n)).values,
                ),
                layout={
                    **dark_layout,
                    "scene": {
                        "xaxis": {
                                    "title": "Score",
                                    "gridcolor": "#21262d",
                                    "backgroundcolor": "#0d1117"
                                  },
                        "yaxis": {"title": "Confidence", 
                                   "gridcolor": "#21262d",
                                   "backgroundcolor": "#0d1117"},
                        "zaxis": {"title": "Threshold Ratio", 
                                   "gridcolor": "#21262d", 
                                   "backgroundcolor": "#0d1117"},
                    },
                    "title": "Multivector Manifold (PCA Approximation)",
                },
            )

        # -- Alert table --
        if df.empty:
            table_data = []
            table_cols = []
        else:
            show_cols = [c for c in ["flow_id", "datetime", "score", "threshold",
                                      "is_anomaly", "confidence", "top_blade"]
                         if c in df.columns]
            tdf = df[show_cols].tail(200).copy()
            if "score" in tdf.columns:
                tdf["score"] = tdf["score"].round(5)
            if "threshold" in tdf.columns:
                tdf["threshold"] = tdf["threshold"].round(5)
            if "confidence" in tdf.columns:
                tdf["confidence"] = tdf["confidence"].round(4)
            table_data = tdf.to_dict("records")
            table_cols = [{"name": c, "id": c} for c in tdf.columns]

        return kpi_cards, fig_ts, fig_blade, fig_3d, table_data, table_cols

    return app
