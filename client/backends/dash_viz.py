import base64
import io
import threading
from typing import Any, Dict, List, Optional, Tuple
import logging, sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

import numpy as np

# Import Dash and dependencies with backward compatibility (dash<2.0)
DASH_AVAILABLE = False
GO_AVAILABLE = False
try:
    from dash import Dash, dcc, html, no_update  # Dash >= 2.x
    try:
        from dash import Output, Input  # Dash >= 2.10
    except Exception:
        from dash.dependencies import Input, Output  # Older Dash
    DASH_AVAILABLE = True
except Exception:
    # Keep import-time failures isolated so the rest of the system can run.
    Dash = None  # type: ignore
    dcc = None  # type: ignore
    html = None  # type: ignore
    Output = None  # type: ignore
    Input = None  # type: ignore
    no_update = None  # type: ignore

try:
    import plotly.graph_objects as go
    GO_AVAILABLE = True
except Exception:
    go = None  # type: ignore


def _np_image_to_base64(img: np.ndarray) -> str:
    """Encode an RGB numpy image as base64 PNG data URI for Plotly layout.images."""
    import cv2

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    success, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if not success:
        raise RuntimeError("Failed to encode image for Dash display")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


class LiveViz:
    """Lightweight Dash app for real-time frame + detection overlays.

    - Main view: camera frame with bounding boxes + labels using Plotly shapes/annotations
    - Side view: causal graph figure (if provided via logic chains)
    """

    def __init__(
        self,
        *,
        title: str = "OpenPI Visualization",
        host: str = "127.0.0.1",
        port: int = 8060,
        logic_chains: Optional[List[Dict[str, Any]]] = None,
        interval_ms: int = 300,
    ) -> None:
        self.title = title
        self.host = host
        self.port = port
        self.logic_chains = logic_chains or []
        self.interval_ms = int(max(10, interval_ms))

        self._lock = threading.Lock()
        self._frame_index: int = 0
        self._frame: Optional[np.ndarray] = None  # RGB HxWx3
        self._results: List[Dict[str, Any]] = []
        self._server_thread: Optional[threading.Thread] = None
        self._running = False

        if not DASH_AVAILABLE:
            raise RuntimeError("Dash is not available. Please install 'dash' >= 1.0.")
        if not GO_AVAILABLE:
            raise RuntimeError("Plotly is not available. Please install 'plotly'.")

        self._app = Dash(__name__, serve_locally=True, compress=True)
        self._app.enable_dev_tools(dev_tools_hot_reload=False, dev_tools_ui=False)
        self._app.title = self.title
        self._app.layout = html.Div(
            [
                html.Div(
                    [
                        html.H2(self.title, style={"textAlign": "center"})
                    ]
                ),
                html.Div(
                    [
                        dcc.Graph(id="main-graph", style={"height": "60vh", "width": "100%", "margin": "0", "padding": "0"}, config={"displayModeBar": False, "responsive": True}),
                        dcc.Graph(id="causal-graph", style={"height": "60vh", "width": "100%"}),
                    ],
                    style={"display": "grid", "gridTemplateColumns": "40% 60%", "gap": "12px"},
                ),
                html.Div(
                    id="info-panel",
                    children="booting...",  # 默认就有字
                    style={
                        "whiteSpace": "pre-wrap",
                        "fontFamily": "monospace",
                        "fontSize": "12px",
                        "border": "1px dashed #bbb",
                        "padding": "6px",
                        "marginTop": "8px",
                    },
                ),
                dcc.Interval(id="tick", interval=self.interval_ms, n_intervals=0),
            ],
            style={"padding": "8px"},
        )

        @self._app.callback(
            Output("main-graph", "figure"),
            Output("causal-graph", "figure"),
            Output("info-panel", "children"),
            Input("tick", "n_intervals"),
            prevent_initial_call=False,
        )
        def _update(n):  # 注意接收 n
            print(f"[dash] tick={n}", flush=True)

            # 总会变化的“心跳点”，用来证明前后端在说话
            hb_fig = go.Figure(go.Scatter(x=[0, 1], y=[0, (n or 0) % 2],
                                        mode="markers+lines", name="heartbeat"))
            hb_fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))

            frame, results, idx = self._snapshot()
            if frame is None:
                # 关键：不要 no_update，返回占位（证明回调在触发）
                return hb_fig, go.Figure(), f"Waiting for frames... ticks={n}"

            main_fig = self._build_main_figure(frame, results)
            causal_fig = self._build_causal_figure(results)
            info_text = self._build_info_text(results, idx) + f"\n[tick={n}]"
            return main_fig, causal_fig, info_text
        
        @self._app.server.route("/health")
        def _health():
            return "ok", 200

    def start(self) -> None:
        if self._running:
            return
        self._running = True

        def _serve():
            # Use the stable Dash API for running a server in scripts
            # (works consistently across Dash 1.x and 2.x).
            self._app.run(
                debug=False, host=self.host, port=self.port, use_reloader=False
            )

        self._server_thread = threading.Thread(target=_serve, daemon=True)
        self._server_thread.start()

    def stop(self) -> None:
        # Dash does not expose a direct stop; rely on process shutdown.
        self._running = False

    def update(self, frame_rgb: np.ndarray, results: List[Dict[str, Any]], frame_index: int) -> None:
        with self._lock:
            self._frame = frame_rgb
            self._results = results
            self._frame_index = frame_index
        print(f"[producer] fed frame={frame_index} shape={getattr(frame_rgb, 'shape', None)}", flush=True)

    def _snapshot(self) -> Tuple[Optional[np.ndarray], List[Dict[str, Any]], int]:
        with self._lock:
            return (None if self._frame is None else self._frame.copy(), list(self._results), int(self._frame_index))

    def _build_main_figure(self, frame: np.ndarray, results: List[Dict[str, Any]]):
        h, w = frame.shape[:2]
        img_uri = _np_image_to_base64(frame)

        fig = go.Figure()
        # Add background image
        fig.add_layout_image(
            dict(
                source=img_uri,
                x=0,                # 左边界
                y=0,                # 顶边（注意 y 轴是 [h,0]）
                sizex=w,
                sizey=h,
                xref="x",
                yref="y",
                xanchor="left",
                yanchor="top",
                sizing="stretch",   # 按给定 sizex/sizey 拉伸，避免留白
                layer="below",
            )
        )

        # # Boxes and labels
        # shapes = []
        # annotations = []
        # colors = ["#e74c3c", "#2ecc71", "#3498db", "#f1c40f", "#9b59b6"]
        # for i, item in enumerate(results):
        #     bbox = item.get("bbox", [0, 0, 0, 0])
        #     label = str(item.get("label", ""))
        #     x1, y1, x2, y2 = bbox
            # color = colors[i % len(colors)]
            # shapes.append(
            #     dict(type="rect", x0=x1, y0=y1, x1=x2, y1=y2, line=dict(color=color, width=2), fillcolor="rgba(0,0,0,0)")
            # )
            # annotations.append(
            #     dict(
            #         x=x1,
            #         y=max(0, y1 - 8),
            #         xref="x",
            #         yref="y",
            #         text=label,
            #         showarrow=False,
            #         font=dict(color="white", size=12),
            #         align="left",
            #         bgcolor=color,
            #     )
            # )

            # # Top attributes/affordances (first 3)
            # top_attrs = list(item.get("top_attrs", []))[:3]
            # top_affs = list(item.get("top_affs", []))[:3]
            # attr_probs = list(item.get("attr_probs", []))[:3]
            # aff_probs = list(item.get("aff_probs", []))[:3]
            # lines = []
            # for a, p in zip(top_attrs, attr_probs):
            #     lines.append(f"A: {a} ({float(p):.2f})")
            # for f, p in zip(top_affs, aff_probs):
            #     lines.append(f"F: {f} ({float(p):.2f})")
            # if lines:
            #     annotations.append(
            #         dict(
            #             x=x1 + 5,
            #             y=y1 + 15,
            #             xref="x",
            #             yref="y",
            #             text="<br>".join(lines),
            #             showarrow=False,
            #             font=dict(color=color, size=10),
            #             align="left",
            #             bgcolor="rgba(255,255,255,0.6)",
            #         )
            #     )

        fig.update_layout(
            xaxis=dict(visible=False, range=[0, w], constrain="domain"),
            yaxis=dict(visible=False, range=[h, 0], scaleanchor="x", scaleratio=1),
            margin=dict(l=0, r=0, t=0, b=0),
            # shapes=shapes,
            # annotations=annotations,
            uirevision="stream",  # keep viewport stable across updates
        )
        return fig

    def _build_causal_figure(self, results: List[Dict[str, Any]]):
        if not self.logic_chains or not results:
            return go.Figure().update_layout(margin=dict(l=0, r=0, t=0, b=0))

        first = results[0]
        top_attrs = list(first.get("top_attrs", []))
        top_affs = list(first.get("top_affs", []))
        attr_probs = list(first.get("attr_probs", []))
        aff_probs = list(first.get("aff_probs", []))

        if not top_affs:
            return go.Figure().update_layout(margin=dict(l=0, r=0, t=0, b=0))

        affordance_name = str(top_affs[0])
        attr_prob_dict = {a: float(p) for a, p in zip(top_attrs, attr_probs[: len(top_attrs)])}
        aff_prob_dict = {a: float(p) for a, p in zip(top_affs, aff_probs[: len(top_affs)])}

        try:
            from .graph import build_causal_graph_figure

            return build_causal_graph_figure(
                logic_chains=self.logic_chains,
                attr_prob_dict=attr_prob_dict,
                aff_prob_dict=aff_prob_dict,
                affordance_name=affordance_name,
            )
        except Exception as e:
            # Fall back to a simple placeholder figure
            import traceback
            print(f"Error building causal graph: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            fig = go.Figure()
            fig.add_annotation(text="Causal graph unavailable", showarrow=False)
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            return fig

    def _build_info_text(self, results: List[Dict[str, Any]], frame_index: int) -> str:
        lines = [f"Frame: {frame_index} | Objects: {len(results)}"]
        for i, item in enumerate(results[:3]):
            label = str(item.get("label", ""))
            # Safely get first values without relying on truthiness of numpy arrays
            top_attrs_raw = item.get("top_attrs")
            if top_attrs_raw is None:
                top_attrs = []
            elif isinstance(top_attrs_raw, np.ndarray):
                top_attrs = top_attrs_raw.tolist()
            else:
                top_attrs = list(top_attrs_raw)

            top_affs_raw = item.get("top_affs")
            if top_affs_raw is None:
                top_affs = []
            elif isinstance(top_affs_raw, np.ndarray):
                top_affs = top_affs_raw.tolist()
            else:
                top_affs = list(top_affs_raw)

            attr_probs_raw = item.get("attr_probs")
            if attr_probs_raw is None:
                attr_probs = []
            elif isinstance(attr_probs_raw, np.ndarray):
                attr_probs = attr_probs_raw.tolist()
            else:
                attr_probs = list(attr_probs_raw)

            aff_probs_raw = item.get("aff_probs")
            if aff_probs_raw is None:
                aff_probs = []
            elif isinstance(aff_probs_raw, np.ndarray):
                aff_probs = aff_probs_raw.tolist()
            else:
                aff_probs = list(aff_probs_raw)

            top_attr = str(top_attrs[0]) if top_attrs else "none"
            top_aff = str(top_affs[0]) if top_affs else "none"
            attr_prob = float(attr_probs[0]) if attr_probs else 0.0
            aff_prob = float(aff_probs[0]) if aff_probs else 0.0
            lines.append(f"{i+1}. {label}: {top_attr}({attr_prob:.2f}), {top_aff}({aff_prob:.2f})")
        return "\n".join(lines)
