from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import numpy as np


@dataclass
class VGGTReconConfig:
    model_path: str = "./models/facebook/VGGT-1B"
    device: Optional[str] = None  # "cuda" or "cpu"; autodetect if None
    dtype: Optional[str] = None   # "bfloat16" | "float16" | None (auto)
    preprocess_mode: str = "crop"  # "crop" | "pad"
    conf_percentile: float = 50.0
    max_points: int = 80000
    single_history: int = 8  # window size for single-view stream


class VGGTReconstructor:
    """Lightweight wrapper around VGGT to produce a point cloud from images.

    - Accepts single- or multi-view RGB images (numpy HxWx3 in 0..255)
    - Returns downsampled point cloud with per-point RGB colors in 0..255
    - Thread-safe; loads model lazily on first use
    """

    def __init__(self, cfg: Optional[VGGTReconConfig] = None) -> None:
        self.cfg = cfg or VGGTReconConfig()
        self._model = None
        self._device = None
        self._dtype = None
        self._lock = threading.Lock()
        self._single_buffer: List[np.ndarray] = []

    # -------------------------- Public API --------------------------
    def warmup(self) -> bool:
        """Eagerly load VGGT so we don't block later.

        Returns True if the model is ready after this call, False otherwise.
        """
        try:
            with self._lock:
                self._ensure_model()
            return self._model is not None
        except Exception:
            return False

    def update_single_and_reconstruct(self, img_rgb: np.ndarray) -> Optional[Dict[str, Any]]:
        """Append a frame to the single-view buffer and reconstruct when full.

        Returns a dict with 'points' (Nx3 float), 'colors' (Nx3 uint8) when ready, else None.
        """
        if img_rgb is None or img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
            return None
        with self._lock:
            self._single_buffer.append(img_rgb)
            if len(self._single_buffer) < max(2, int(self.cfg.single_history)):
                return None
            imgs = list(self._single_buffer)
            self._single_buffer.clear()
        try:
            return self._reconstruct(imgs, source_mode="single")
        except Exception as e:
            return {"error": f"VGGT single-view reconstruction failed: {e}"}

    def reconstruct_multi(self, imgs_rgb: List[np.ndarray]) -> Optional[Dict[str, Any]]:
        """Reconstruct from a list of multi-view images for the current frame."""
        if not imgs_rgb:
            return None
        try:
            return self._reconstruct(imgs_rgb, source_mode="multi")
        except Exception as e:
            return {"error": f"VGGT multi-view reconstruction failed: {e}"}

    # ------------------------ Internal helpers ----------------------
    def _ensure_model(self):
        if self._model is not None:
            return
        # Lazy import to avoid hard dependency on environments without Torch
        import torch
        from vggt.models.vggt import VGGT

        device = self.cfg.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        if self.cfg.dtype is not None:
            if self.cfg.dtype.lower() == "bfloat16":
                self._dtype = torch.bfloat16
            elif self.cfg.dtype.lower() == "float16":
                self._dtype = torch.float16
            else:
                self._dtype = None
        else:
            if device == "cuda":
                # Use bfloat16 on Ada/Hopper, else float16
                major = torch.cuda.get_device_capability()[0]
                self._dtype = torch.bfloat16 if major >= 8 else torch.float16
            else:
                self._dtype = None

        self._model = VGGT.from_pretrained(self.cfg.model_path).to(device).eval()

    def _preprocess_images(self, imgs_rgb: List[np.ndarray]):
        """Preprocess numpy images to a torch tensor [S,3,H,W] in 0..1.

        Mirrors the logic of vggt.vggt.utils.load_fn.load_and_preprocess_images for in-memory arrays.
        """
        import torch
        from PIL import Image
        from torchvision import transforms as TF

        mode = self.cfg.preprocess_mode
        if mode not in ("crop", "pad"):
            mode = "crop"

        to_tensor = TF.ToTensor()
        target_size = 518

        images_t: List[torch.Tensor] = []
        shapes = set()

        for arr in imgs_rgb:
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr, mode="RGB")
            width, height = img.size

            if mode == "pad":
                if width >= height:
                    new_width = target_size
                    new_height = round(height * (new_width / width) / 14) * 14
                else:
                    new_height = target_size
                    new_width = round(width * (new_height / height) / 14) * 14
            else:  # crop
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14

            img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
            t = to_tensor(img)  # 0..1

            if mode == "crop" and new_height > target_size:
                start_y = (new_height - target_size) // 2
                t = t[:, start_y : start_y + target_size, :]

            if mode == "pad":
                h_padding = target_size - t.shape[1]
                w_padding = target_size - t.shape[2]
                if h_padding > 0 or w_padding > 0:
                    pad_top = h_padding // 2
                    pad_bottom = h_padding - pad_top
                    pad_left = w_padding // 2
                    pad_right = w_padding - pad_left
                    t = torch.nn.functional.pad(
                        t, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                    )

            shapes.add((t.shape[1], t.shape[2]))
            images_t.append(t)

        if len(shapes) > 1:
            max_h = max(s[0] for s in shapes)
            max_w = max(s[1] for s in shapes)
            padded: List["torch.Tensor"] = []
            for t in images_t:
                h_padding = max_h - t.shape[1]
                w_padding = max_w - t.shape[2]
                if h_padding > 0 or w_padding > 0:
                    pad_top = h_padding // 2
                    pad_bottom = h_padding - pad_top
                    pad_left = w_padding // 2
                    pad_right = w_padding - pad_left
                    t = torch.nn.functional.pad(
                        t, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                    )
                padded.append(t)
            images_t = padded

        images = torch.stack(images_t)  # [S,3,H,W]
        return images

    def _build_point_cloud_from_predictions(
        self, predictions, images_t, conf_percentile: float, max_points: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (points Nx3, colors Nx3 in 0..1) from VGGT predictions and input images tensor [S,3,H,W]."""
        import torch
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        from vggt.utils.geometry import unproject_depth_map_to_point_map

        pts = predictions.get("world_points", None)
        conf = predictions.get("world_points_conf", None)

        if pts is None or conf is None:
            if ("depth" in predictions) and ("pose_enc" in predictions):
                _, S, H, W, _ = predictions["depth"].shape
                extrinsic_b, intrinsic_b = pose_encoding_to_extri_intri(predictions["pose_enc"], (H, W))
                extrinsic = extrinsic_b[0]
                intrinsic = intrinsic_b[0]
                depth0 = predictions["depth"][0]
                world_from_depth = unproject_depth_map_to_point_map(depth0, extrinsic, intrinsic)
                pts = world_from_depth
                if "depth_conf" in predictions:
                    dc = predictions["depth_conf"][0]
                    conf = dc.detach().cpu().numpy() if torch.is_tensor(dc) else np.asarray(dc)
                else:
                    conf = np.ones((S, H, W), dtype=np.float32)
            else:
                raise ValueError("Predictions lack both world_points and depth/pose_enc for unprojection.")

        if hasattr(pts, "detach"):
            pts = pts.detach().cpu().numpy()
        if hasattr(conf, "detach"):
            conf = conf.detach().cpu().numpy()
        img = images_t.detach().cpu().numpy()  # [S,3,H,W]

        if pts.ndim == 5:
            pts = pts[0]
        if conf.ndim == 4:
            conf = conf[0]

        S, H, W = img.shape[0], img.shape[2], img.shape[3]
        pred_H, pred_W = pts.shape[1], pts.shape[2]

        if (H, W) != (pred_H, pred_W):
            try:
                import cv2
                img_resized = []
                for s in range(S):
                    hwc = np.transpose(img[s], (1, 2, 0))
                    hwc = cv2.resize(hwc, (pred_W, pred_H), interpolation=cv2.INTER_LINEAR)
                    img_resized.append(np.transpose(hwc, (2, 0, 1)))
                img = np.stack(img_resized, axis=0)
            except Exception:
                img = img[:, :, :pred_H, :pred_W]

        points = pts.reshape(-1, 3)
        conf_f = conf.reshape(-1)
        colors = np.transpose(img, (0, 2, 3, 1)).reshape(-1, 3)

        if conf_f.size > 0 and conf_percentile is not None:
            thr = float(np.percentile(conf_f, float(conf_percentile)))
            keep = (conf_f >= thr) & (conf_f > 1e-5)
        else:
            keep = np.ones(points.shape[0], dtype=bool)
        points = points[keep]
        colors = colors[keep]

        if points.shape[0] > max_points:
            idx = np.random.choice(points.shape[0], size=max_points, replace=False)
            points = points[idx]
            colors = colors[idx]
        return points.astype(np.float64), colors.astype(np.float64)

    def _reconstruct(self, imgs_rgb: List[np.ndarray], source_mode: str) -> Dict[str, Any]:
        """Core reconstruction pipeline."""
        try:
            import torch
        except Exception:
            # Torch not available: return a small dummy cloud
            pts, cols = self._dummy_point_cloud(imgs_rgb)
            return {
                "points": pts,
                "colors": cols,
                "num_points": int(pts.shape[0]),
                "source": source_mode,
                "status": "dummy",
            }

        self._ensure_model()

        images = self._preprocess_images(imgs_rgb).to(self._device)
        with torch.no_grad():
            if self._device == "cuda" and self._dtype is not None:
                with torch.cuda.amp.autocast(dtype=self._dtype):
                    predictions = self._model(images)
            else:
                predictions = self._model(images)

        # Attach images for color extraction if not present
        if "images" not in predictions:
            predictions["images"] = images

        points, colors01 = self._build_point_cloud_from_predictions(
            predictions, images, self.cfg.conf_percentile, self.cfg.max_points
        )
        colors255 = np.clip(colors01 * 255.0, 0, 255).astype(np.uint8)
        return {
            "points": points,
            "colors": colors255,
            "num_points": int(points.shape[0]),
            "source": source_mode,
            "status": "ok",
        }

    @staticmethod
    def _dummy_point_cloud(imgs_rgb: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        n = 2000
        pts = np.random.uniform(-1, 1, size=(n, 3)).astype(np.float64)
        # colors from first image average
        if imgs_rgb:
            img = imgs_rgb[0]
            mean_rgb = np.mean(img.reshape(-1, 3), axis=0)
            cols = np.tile(mean_rgb, (n, 1))
        else:
            cols = np.random.randint(0, 255, size=(n, 3)).astype(np.uint8)
        return pts, cols.astype(np.uint8)
