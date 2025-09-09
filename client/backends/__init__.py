"""Visualization backends factory and common interface.

Provides light wrappers around cv2, Dash, and Plotly visualizations
so the server can enable multiple backends at once.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List
import logging
import numpy as np

logger = logging.getLogger(__name__)


class VisualizationBackend:
    """Simple interface each backend implements."""

    name: str = "base"

    def start(self) -> None:  # pragma: no cover - thin wrapper
        pass

    def update(self, frame_rgb: np.ndarray, results: List[Dict[str, Any]], frame_index: int) -> None:  # noqa: ANN401
        raise NotImplementedError

    def stop(self) -> None:  # pragma: no cover - thin wrapper
        pass


def make_backends(
    names: Iterable[str],
    *,
    output_dir: str,
    show_causal_graph: bool,
    logic_chains: List[Dict[str, Any]] | None,
    dash_interval_ms: int,
    plotly_refresh_ms: int,
    plotly_auto_open: bool,
) -> List[VisualizationBackend]:
    """Create visualization backends from name list.

    Unknown names are ignored with a warning.
    """
    from .cv2_backend import Cv2Backend
    from .dash_backend import DashBackend
    from .plotly_backend import PlotlyBackend

    backends: List[VisualizationBackend] = []
    normalized = [n.strip().lower() for n in names]

    for n in normalized:
        try:
            if n == "cv2":
                b = Cv2Backend(output_dir=output_dir, show_causal_graph=show_causal_graph, logic_chains=(logic_chains or []) if show_causal_graph else [])
            elif n == "dash":
                b = DashBackend(
                    logic_chains=(logic_chains or []) if show_causal_graph else [],
                    interval_ms=int(max(10, dash_interval_ms)),
                )
            elif n == "plotly":
                b = PlotlyBackend(
                    output_dir=output_dir,
                    refresh_ms=int(max(50, plotly_refresh_ms)),
                    auto_open=bool(plotly_auto_open),
                    logic_chains=(logic_chains or []) if show_causal_graph else [],
                )
            else:
                logger.warning("Unknown visualization backend: %s", n)
                continue
            backends.append(b)
        except Exception as e:  # pragma: no cover - environment specific
            logger.warning("Failed to init backend '%s': %s", n, e)

    # Start those that need startup
    for b in backends:
        try:
            b.start()
        except Exception as e:  # pragma: no cover - environment specific
            logger.warning("Failed to start backend '%s': %s", getattr(b, "name", type(b).__name__), e)
    return backends
