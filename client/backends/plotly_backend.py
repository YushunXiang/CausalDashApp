from __future__ import annotations

from typing import Any, Dict, List, Optional
import os
import re
import copy

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio


class PlotlyBackend:
    name = "plotly"

    def __init__(
        self,
        *,
        output_dir: str,
        refresh_ms: int = 1000,
        auto_open: bool = False,
        logic_chains: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.output_dir = output_dir
        self.refresh_ms = int(max(50, refresh_ms))
        self.auto_open = bool(auto_open)
        self.logic_chains = logic_chains or []

        os.makedirs(self.output_dir, exist_ok=True)
        self._html_out = os.path.join(self.output_dir, "plotly_viz.html")
        self._frame_png = os.path.join(self.output_dir, "plotly_viz_frame.png")
        self._html_written = False

    def start(self) -> None:
        # Nothing to start; HTML is written lazily on first update
        pass

    def stop(self) -> None:
        # Nothing persistent to stop
        pass

    def update(self, frame_rgb: np.ndarray, results: List[Dict[str, Any]], frame_index: int) -> None:  # noqa: ANN401
        import cv2 as _cv2

        # Persist current frame for HTML to refresh via timestamp param
        _cv2.imwrite(self._frame_png, _cv2.cvtColor(frame_rgb, _cv2.COLOR_RGB2BGR))

        if not self._html_written:
            rel_frame = os.path.relpath(self._frame_png, start=os.path.dirname(self._html_out) or ".").replace(os.sep, "/")
            combined = self._build_combined_figure(
                frame_rgb,
                results,
                image_source_override=rel_frame + "?ts=0",
            )
            combined.update_layout(title=f"Frame {frame_index}")
            self._write_html(combined, self._html_out, frame_png=self._frame_png, refresh_ms=self.refresh_ms, auto_open=self.auto_open)
            self._html_written = True

    @staticmethod
    def _np_image_to_base64(img: np.ndarray) -> str:
        import cv2 as _cv2
        import base64 as _b64

        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        success, buf = _cv2.imencode(".png", _cv2.cvtColor(img, _cv2.COLOR_RGB2BGR))
        if not success:
            raise RuntimeError("Failed to encode image for Plotly display")
        b64 = _b64.b64encode(buf.tobytes()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    def _build_main_figure(self, frame: np.ndarray, results: List[Dict[str, Any]], image_source_override: Optional[str] = None):
        h, w = frame.shape[:2]
        img_uri = image_source_override if image_source_override is not None else self._np_image_to_base64(frame)

        fig = go.Figure()
        fig.add_layout_image(dict(source=img_uri, x=0, y=h, sizex=w, sizey=h, xref="x", yref="y", layer="below", yanchor="bottom"))

        shapes = []
        annotations = []
        colors = ["#e74c3c", "#2ecc71", "#3498db", "#f1c40f", "#9b59b6"]
        for i, item in enumerate(results):
            bbox = item.get("bbox", [0, 0, 0, 0])
            label = str(item.get("label", ""))
            x1, y1, x2, y2 = bbox
            color = colors[i % len(colors)]
            shapes.append(dict(type="rect", x0=x1, y0=y1, x1=x2, y1=y2, line=dict(color=color, width=2), fillcolor="rgba(0,0,0,0)"))
            annotations.append(
                dict(
                    x=x1,
                    y=max(0, y1 - 8),
                    xref="x",
                    yref="y",
                    text=label,
                    showarrow=False,
                    font=dict(color="white", size=12),
                    align="left",
                    bgcolor=color,
                )
            )

        fig.update_layout(
            xaxis=dict(visible=False, range=[0, w], constrain="domain"),
            yaxis=dict(visible=False, range=[h, 0], scaleanchor="x", scaleratio=1),
            margin=dict(l=0, r=0, t=0, b=0),
            shapes=shapes,
            annotations=annotations,
            uirevision="stream",
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
            from openpi.utils.graph import build_causal_graph_figure

            return build_causal_graph_figure(
                logic_chains=self.logic_chains,
                attr_prob_dict=attr_prob_dict,
                aff_prob_dict=aff_prob_dict,
                affordance_name=affordance_name,
            )
        except Exception:
            fig = go.Figure()
            fig.add_annotation(text="Causal graph unavailable", showarrow=False)
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            return fig

    def _build_combined_figure(self, frame: np.ndarray, results: List[Dict[str, Any]], image_source_override: Optional[str] = None):
        main_fig = self._build_main_figure(frame, results, image_source_override=image_source_override)
        causal_fig = self._build_causal_figure(results)

        fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.05, specs=[[{"type": "xy"}, {"type": "xy"}]])

        for tr in main_fig.data:
            fig.add_trace(tr, row=1, col=1)
        shapes: List[Dict[str, Any]] = []
        for s in (main_fig.layout.shapes or []):
            ss = copy.deepcopy(s)
            ss["xref"] = "x"
            ss["yref"] = "y"
            shapes.append(ss)
        images: List[Dict[str, Any]] = []
        for im in (main_fig.layout.images or []):
            ii = copy.deepcopy(im)
            ii["xref"] = "x"
            ii["yref"] = "y"
            images.append(ii)
        annos: List[Dict[str, Any]] = []
        for a in (main_fig.layout.annotations or []):
            aa = copy.deepcopy(a)
            aa["xref"] = "x"
            aa["yref"] = "y"
            annos.append(aa)

        for tr in causal_fig.data:
            fig.add_trace(tr, row=1, col=2)
        for s in (causal_fig.layout.shapes or []):
            ss = copy.deepcopy(s)
            ss["xref"] = "x2"
            ss["yref"] = "y2"
            shapes.append(ss)
        for im in (causal_fig.layout.images or []):
            ii = copy.deepcopy(im)
            ii["xref"] = "x2"
            ii["yref"] = "y2"
            images.append(ii)
        for a in (causal_fig.layout.annotations or []):
            aa = copy.deepcopy(a)
            aa["xref"] = "x2"
            aa["yref"] = "y2"
            annos.append(aa)

        fig.update_layout(shapes=shapes, images=images, annotations=annos, margin=dict(l=0, r=0, t=30, b=0))

        W = 1200
        fig.update_layout(width=W)
        left_dom = fig.layout.xaxis.domain
        left_px = (left_dom[1] - left_dom[0]) * W
        need_h = int(left_px * (frame.shape[0] / frame.shape[1]))
        mt, mb = fig.layout.margin.t or 0, fig.layout.margin.b or 0
        fig.update_layout(height=need_h + mt + mb)
        return fig

    def _write_html(self, fig, path: str, frame_png: str, refresh_ms: int = 1000, auto_open: bool = False, base_url: Optional[str] = None) -> None:
        rel = os.path.relpath(frame_png, start=os.path.dirname(path) or ".").replace(os.sep, "/")
        img_url = (base_url.rstrip("/") + "/" + rel) if base_url else rel

        if hasattr(fig.layout, "images") and fig.layout.images:
            fig.layout.images[0].source = img_url + "?ts=0"

        html = pio.to_html(fig, include_plotlyjs="full", full_html=True, config={"displayModeBar": False})
        m = re.search(r'<div id="([^"]+)" class="plotly-graph-div"', html)
        div_id = m.group(1) if m else None
        if div_id:
            updater = f"""
<script>
(function(){{
  var div = document.getElementById('{div_id}');
  function updateImage(){{
    var ts = Date.now();
    Plotly.relayout(div, {{'images[0].source': '{img_url}' + '?ts=' + ts}});
  }}
  updateImage();
  setInterval(updateImage, {max(100, int(refresh_ms))});
}})();
</script>
"""
            html = html.replace("</body>", updater + "\n</body>") if "</body>" in html else html + updater

        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        if auto_open:
            import webbrowser

            webbrowser.open((base_url.rstrip("/") + "/" + os.path.basename(path)) if base_url else os.path.abspath(path))

