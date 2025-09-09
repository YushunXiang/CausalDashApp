from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


class DashBackend:
    name = "dash"

    def __init__(
        self,
        *,
        logic_chains: Optional[List[Dict[str, Any]]] = None,
        interval_ms: int = 300,
        title: str | None = None,
        host: str = "127.0.0.1",
        port: int = 8060,
    ) -> None:
        # Defer import of LiveViz to avoid hard dependency
        from .dash_viz import LiveViz

        self._viz = LiveViz(
            title="任务场景因果可视化",
            host=host,
            port=int(port),
            logic_chains=logic_chains or [],
            interval_ms=int(max(10, interval_ms)),
        )

    def start(self) -> None:
        self._viz.start()

    def stop(self) -> None:
        try:
            self._viz.stop()
        except Exception:
            pass

    def update(self, frame_rgb: np.ndarray, results: List[Dict[str, Any]], frame_index: int) -> None:  # noqa: ANN401
        self._viz.update(frame_rgb, results, frame_index)

