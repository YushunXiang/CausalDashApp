import os
import argparse
from typing import List, Tuple

import torch
import numpy as np
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map


def extract_frames_from_video(video_path: str, out_dir: str, every_n: int = 10, max_frames: int = 30) -> List[str]:
    """Extract frames from a video and save as PNGs, returning the paths.

    - every_n: sample every N-th frame
    - max_frames: upper bound on number of frames saved
    """
    import cv2

    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_idx = 0
    saved = 0
    saved_paths: List[str] = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if frame_idx % max(1, every_n) == 0:
            # Convert BGR->RGB then save
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            # Save
            out_path = os.path.join(out_dir, f"frame_{frame_idx:06d}.png")
            cv2.imwrite(out_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            saved_paths.append(out_path)
            saved += 1
            if max_frames > 0 and saved >= max_frames:
                break
        frame_idx += 1
    cap.release()
    print(f"[video] total={total} extracted={len(saved_paths)} from {video_path}")
    return saved_paths


def build_point_cloud_from_predictions(predictions, images_t: torch.Tensor, conf_percentile: float = 50.0,
                                       max_points: int = 80000) -> Tuple[np.ndarray, np.ndarray]:
    """Return (points Nx3, colors Nx3 in 0..1) from VGGT predictions and input images tensor [S,3,H,W]."""
    # Prefer pointmap prediction; fallback to compute from depth
    pts = predictions.get("world_points", None)
    conf = predictions.get("world_points_conf", None)

    if pts is None or conf is None:
        # Need extrinsic/intrinsic to unproject depth
        if ("depth" in predictions) and ("pose_enc" in predictions):
            # depth: [B,S,H,W,1]
            _, S, H, W, _ = predictions["depth"].shape
            extrinsic_b, intrinsic_b = pose_encoding_to_extri_intri(predictions["pose_enc"], (H, W))
            # Remove batch dim
            extrinsic = extrinsic_b[0]
            intrinsic = intrinsic_b[0]
            depth0 = predictions["depth"][0]
            world_from_depth = unproject_depth_map_to_point_map(depth0, extrinsic, intrinsic)
            pts = world_from_depth  # (S,H,W,3) numpy
            # depth_conf: [B,S,H,W] -> [S,H,W]
            if "depth_conf" in predictions:
                dc = predictions["depth_conf"][0]
                conf = dc.detach().cpu().numpy() if torch.is_tensor(dc) else np.asarray(dc)
            else:
                conf = np.ones((S, H, W), dtype=np.float32)
        else:
            raise ValueError("Predictions lack both world_points and depth/pose_enc for unprojection.")

    # Move to numpy
    if torch.is_tensor(pts):
        pts = pts.detach().cpu().numpy()
    if torch.is_tensor(conf):
        conf = conf.detach().cpu().numpy()
    img = images_t.detach().cpu().numpy()  # [S,3,H,W]

    # Shapes: pts [B,S,H,W,3] or [S,H,W,3], conf [B,S,H,W] or [S,H,W]
    if pts.ndim == 5:  # [B,S,H,W,3]
        pts = pts[0]
    if conf.ndim == 4:
        conf = conf[0]
    # img [S,3,H,W]
    S, H, W = img.shape[0], img.shape[2], img.shape[3]

    # If needed, resize images to match (H,W) from predictions
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
            H, W = pred_H, pred_W
        except Exception:
            img = img[:, :, :pred_H, :pred_W]
            H, W = pred_H, pred_W

    # Flatten S,H,W
    points = pts.reshape(-1, 3)
    conf_f = conf.reshape(-1)
    colors = np.transpose(img, (0, 2, 3, 1)).reshape(-1, 3)  # 0..1 floats

    # Confidence filter
    if conf_f.size > 0 and conf_percentile is not None:
        thr = np.percentile(conf_f, float(conf_percentile))
        keep = (conf_f >= thr) & (conf_f > 1e-5)
    else:
        keep = np.ones(points.shape[0], dtype=bool)
    points = points[keep]
    colors = colors[keep]

    # Downsample
    if points.shape[0] > max_points:
        idx = np.random.choice(points.shape[0], size=max_points, replace=False)
        points = points[idx]
        colors = colors[idx]
    return points.astype(np.float64), colors.astype(np.float64)


def main():
    parser = argparse.ArgumentParser(description="VGGT quick visualization for single-view video or multi-view images/videos")
    g_in = parser.add_argument_group("input")
    g_in.add_argument("--images", nargs="*", default=None, help="Image files (multi-view)")
    g_in.add_argument("--dir", default=None, help="Directory containing images")
    g_in.add_argument("--video", default=None, help="Single-view video path")
    g_in.add_argument("--videos", nargs="*", default=None, help="Multiple videos (multi-view)")
    g_in.add_argument("--frame-every", type=int, default=10, help="Sample every N frames from video(s)")
    g_in.add_argument("--max-frames", type=int, default=30, help="Max frames to sample from video(s)")
    g_in.add_argument("--mode", choices=["crop", "pad"], default="crop", help="Preprocess mode for images")

    g_viz = parser.add_argument_group("viz")
    g_viz.add_argument("--no-open3d", action="store_true", help="Disable Open3D window (still writes PLY)")
    g_viz.add_argument("--out-ply", default="vggt_pointcloud.ply", help="Output PLY path")
    g_viz.add_argument("--conf-percentile", type=float, default=50.0, help="Drop points below this confidence percentile")
    g_viz.add_argument("--max-points", type=int, default=80000, help="Max points to render/save")

    args = parser.parse_args()

    # Collect image paths
    image_paths: List[str] = []
    if args.images:
        image_paths.extend([p for p in args.images if os.path.isfile(p)])
    if args.dir and os.path.isdir(args.dir):
        for fn in sorted(os.listdir(args.dir)):
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                image_paths.append(os.path.join(args.dir, fn))
    tmp_dirs = []
    if args.video and os.path.isfile(args.video):
        tmp = os.path.join(os.path.dirname(__file__), ".vggt_video_frames")
        image_paths.extend(extract_frames_from_video(args.video, tmp, args.frame_every, args.max_frames))
        tmp_dirs.append(tmp)
    if args.videos:
        for i, vp in enumerate(args.videos):
            if os.path.isfile(vp):
                tmp = os.path.join(os.path.dirname(__file__), f".vggt_video_frames_{i}")
                image_paths.extend(extract_frames_from_video(vp, tmp, args.frame_every, args.max_frames))
                tmp_dirs.append(tmp)

    if not image_paths:
        # Fallback to a demo image in this folder
        fallback = os.path.join(os.path.dirname(__file__), "vggt-test.jpg")
        if os.path.isfile(fallback):
            image_paths = [fallback]
            print(f"[input] No inputs given; falling back to {fallback}")
        else:
            raise SystemExit("No input images/videos found.")

    print(f"[input] Using {len(image_paths)} frame(s)/image(s)")

    # Model + device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    print(f"[device] {device}, autocast dtype={dtype}")

    model = VGGT.from_pretrained("./models/facebook/VGGT-1B").to(device).eval()

    # Load & preprocess images
    images = load_and_preprocess_images(image_paths, mode=args.mode).to(device)  # [S,3,H,W]
    print(f"[prep] images tensor shape: {tuple(images.shape)}")

    # Inference
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(device == "cuda"), dtype=dtype):
            predictions = model(images)
    print("[pred] keys and shapes:")
    for k, v in predictions.items():
        try:
            print(f"  - {k}: {tuple(v.shape)}")
        except Exception:
            print(f"  - {k}: {type(v)}")

    # Build point cloud
    points, colors = build_point_cloud_from_predictions(predictions, images, args.conf_percentile, args.max_points)
    print(f"[pc] points={points.shape} colors={colors.shape}")

    # Save and visualize via Open3D
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(args.out_ply, pcd)
        print(f"[o3d] wrote {args.out_ply}")
        if not args.no_open3d:
            o3d.visualization.draw_geometries([pcd], window_name="VGGT Point Cloud")
    except Exception as e:
        print(f"[o3d] skipped or failed: {e}")

    # Optional cleanup of temp dirs
    for d in tmp_dirs:
        try:
            for fn in os.listdir(d):
                os.remove(os.path.join(d, fn))
            os.rmdir(d)
        except Exception:
            pass


if __name__ == "__main__":
    main()
