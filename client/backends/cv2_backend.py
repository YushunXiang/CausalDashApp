from __future__ import annotations

from typing import Any, Dict, List
import os
import cv2
import numpy as np


class Cv2Backend:
    name = "cv2"

    def __init__(self, *, output_dir: str, window_name: str = "任务场景因果可视化", show_causal_graph: bool = False, logic_chains: Any = None) -> None:
        self.output_dir = output_dir
        self.window_name = window_name
        self.show_causal_graph = show_causal_graph
        self.logic_chains = logic_chains
        self._fullscreen = False
        self._closed = False

    def start(self) -> None:
        if self._closed:
            return
        # Create a window and size it to 80% of 1920x1080 as a sane default
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        screen_width, screen_height = 1920, 1080
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.8)
        cv2.resizeWindow(self.window_name, window_width, window_height)
        cv2.moveWindow(self.window_name, int(screen_width * 0.1), int(screen_height * 0.1))

    def stop(self) -> None:
        if not self._closed:
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass
        self._closed = True

    def _draw_overlay(self, frame_rgb: np.ndarray, results: List[Dict[str, Any]]) -> np.ndarray:
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        colors = [
            (0, 0, 255),
            (0, 255, 0),
            (255, 0, 0),
            (255, 255, 0),
            (255, 0, 255),
        ]

        for i, item in enumerate(results):
            bbox = item.get("bbox", [0, 0, 0, 0])
            label = str(item.get("label", ""))
            top_attrs = list(item.get("top_attrs", []))[:3]
            top_affs = list(item.get("top_affs", []))[:3]
            attr_probs = list(item.get("attr_probs", []))[:3]
            aff_probs = list(item.get("aff_probs", []))[:3]

            color = colors[i % len(colors)]
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 1)

            # Label background and text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            label_text = f"{label}"
            text_size = cv2.getTextSize(label_text, font, font_scale, thickness)[0]
            cv2.rectangle(frame_bgr, (x1, max(0, y1 - 25)), (x1 + text_size[0], y1), color, -1)
            cv2.putText(frame_bgr, label_text, (x1, max(12, y1 - 5)), font, font_scale, (255, 255, 255), thickness)

            # Attributes and affordances inside or below the box
            y_offset = y1 + 20
            for j, (attr, prob) in enumerate(zip(top_attrs, attr_probs)):
                cv2.putText(
                    frame_bgr,
                    f"A: {attr} ({float(prob):.2f})",
                    (x1 + 5, y_offset + j * 15),
                    font,
                    0.5,
                    color,
                    1,
                )
            for j, (aff, prob) in enumerate(zip(top_affs, aff_probs)):
                cv2.putText(
                    frame_bgr,
                    f"F: {aff} ({float(prob):.2f})",
                    (x1 + 5, y_offset + (len(top_attrs) + j) * 15),
                    font,
                    0.5,
                    color,
                    1,
                )

        # Right-side info panel
        panel_width = 380
        panel = np.ones((frame_bgr.shape[0], panel_width, 3), dtype=np.uint8) * 255
        y_pos = 20
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(panel, f"Objects: {len(results)}", (10, y_pos), font, 0.6, (0, 0, 0), 1)
        y_pos += 24
        for i, item in enumerate(results[:5]):
            label = str(item.get("label", ""))
            top_attrs = list(item.get("top_attrs", []))
            top_affs = list(item.get("top_affs", []))
            attr_probs = list(item.get("attr_probs", []))
            aff_probs = list(item.get("aff_probs", []))
            top_attr = f"{top_attrs[0]}({float(attr_probs[0]):.2f})" if top_attrs and attr_probs else "-"
            top_aff = f"{top_affs[0]}({float(aff_probs[0]):.2f})" if top_affs and aff_probs else "-"
            cv2.putText(panel, f"{i+1}. {label}: {top_attr}, {top_aff}", (10, y_pos), font, 0.5, (0, 0, 0), 1)
            y_pos += 18

        canvas = np.concatenate([frame_bgr, panel], axis=1)

        # Optional: Add causal graph on the right side
        if self.show_causal_graph and self.logic_chains and len(results) > 0:
            try:
                # Lazy import to avoid errors if module is not available
                from .graph import draw_causal_graph
                
                # Use the first detected object's attributes/affordances for demonstration
                first = results[0]
                top_attrs = first.get('top_attrs', [])
                top_affs = first.get('top_affs', [])
                attr_probs = first.get('attr_probs', [])
                aff_probs = first.get('aff_probs', [])

                # Assemble top attributes and affordances into dictionaries (name -> probability)
                attr_prob_dict = {a: float(p) for a, p in zip(top_attrs, attr_probs[:len(top_attrs)])}
                aff_prob_dict = {a: float(p) for a, p in zip(top_affs, aff_probs[:len(top_affs)])}
                affordance_name = top_affs[0] if len(top_affs) > 0 else None

                if affordance_name is not None:
                    graph_img_bytes = draw_causal_graph(
                        self.logic_chains,
                        attr_prob_dict,
                        aff_prob_dict,
                        affordance_name,
                    )
                    graph_img_array = np.frombuffer(graph_img_bytes, np.uint8)
                    graph_img = cv2.imdecode(graph_img_array, cv2.IMREAD_COLOR)

                    if graph_img is not None:
                        # Resize causal graph to match the height of the main canvas
                        target_h = canvas.shape[0]
                        target_w = int(target_h * graph_img.shape[1] / max(1, graph_img.shape[0]))
                        graph_resized = cv2.resize(graph_img, (target_w, target_h))

                        # Expand canvas and attach causal graph to the right side
                        expanded = np.ones((canvas.shape[0], canvas.shape[1] + target_w, 3), dtype=np.uint8) * 255
                        expanded[:, :canvas.shape[1]] = canvas
                        expanded[:target_h, canvas.shape[1]:canvas.shape[1] + target_w] = graph_resized
                        canvas = expanded

            except ImportError:
                # If openpi module is not available, silently continue without causal graph
                pass
            except Exception as e:
                # Log error but don't crash the main visualization
                print(f"Error drawing causal graph: {e}")

        # Resize to a reasonable on-screen size (up to 80% HD)
        screen_w, screen_h = 1920, 1080
        target_w, target_h = int(screen_w * 0.8), int(screen_h * 0.8)
        scale = min(target_w / canvas.shape[1], target_h / canvas.shape[0], 1.5)
        new_w, new_h = int(canvas.shape[1] * scale), int(canvas.shape[0] * scale)
        return cv2.resize(canvas, (new_w, new_h))

    def update(self, frame_rgb: np.ndarray, results: List[Dict[str, Any]], frame_index: int) -> None:  # noqa: ANN401
        if self._closed:
            return
        canvas = self._draw_overlay(frame_rgb, results)
        cv2.imshow(self.window_name, canvas)

        # Non-blocking key handling
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # q or ESC
            self.stop()
        elif key == ord("s"):
            os.makedirs(self.output_dir, exist_ok=True)
            filename = f"detection_frame_{frame_index:06d}.png"
            save_path = os.path.join(self.output_dir, filename)
            cv2.imwrite(save_path, canvas)
        elif key == ord("f"):
            # Toggle fullscreen
            prop = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN)
            if prop == cv2.WINDOW_FULLSCREEN:
                cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                self._fullscreen = False
            else:
                cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                self._fullscreen = True