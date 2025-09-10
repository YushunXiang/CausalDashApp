import asyncio
import json
import logging
import base64
import io
import numpy as np
from PIL import Image
from sympy import true
import websockets
import argparse
import sys
import os
import socket
import time
import threading

logger = logging.getLogger(__name__)


class WebSocketVisualizationClient:
    """
    WebSocket client that connects to the policy server's visualization stream
    and displays results using various backends (cv2, dash, plotly).
    """
    
    def __init__(
        self,
        server_host: str = "localhost",
        server_port: int = 8766,
        visualization_backends: list[str] = ["cv2"],
        output_dir: str = "remote_visualization_output",
        show_causal_graph: bool = True,
        dash_interval_ms: int = 1000,
        plotly_refresh_ms: int = 1000,
        plotly_auto_open: bool = True,
        wait_for_server: bool = True,
        max_wait_time: int = 300,
        retry_interval: int = 5,
        vggt_recon_mode: str = "off",  # off | single | multi
        vggt_conf_percentile: float = 50.0,
        vggt_max_points: int = 40000,
        vggt_preprocess_mode: str = "crop",
        vggt_single_history: int = 6,
    ):
        self.server_host = server_host
        self.server_port = server_port
        self.visualization_backends = visualization_backends
        self.output_dir = output_dir
        self.show_causal_graph = show_causal_graph
        self.dash_interval_ms = dash_interval_ms
        self.plotly_refresh_ms = plotly_refresh_ms
        self.plotly_auto_open = plotly_auto_open
        self.wait_for_server = wait_for_server
        self.max_wait_time = max_wait_time
        self.retry_interval = retry_interval
        # VGGT options
        self.vggt_recon_mode = vggt_recon_mode
        self.vggt_conf_percentile = float(vggt_conf_percentile)
        self.vggt_max_points = int(vggt_max_points)
        self.vggt_preprocess_mode = vggt_preprocess_mode
        self.vggt_single_history = int(vggt_single_history)
        
        self._viz_backends = []
        self._config_received = False
        self._running = True
        self._reconstructor = None
        
    def _check_server_availability(self) -> bool:
        """Check if the server is available by attempting a socket connection"""
        try:
            with socket.create_connection((self.server_host, self.server_port), timeout=5):
                return True
        except (socket.error, ConnectionRefusedError, OSError):
            return False
    
    async def _wait_for_server(self) -> bool:
        """Wait for the server to become available"""
        if not self.wait_for_server:
            print("⚡ 跳过服务器等待检查")
            return True
            
        logger.info(f"Checking if server is available at {self.server_host}:{self.server_port}")
        print(f"🔍 检查服务器是否可用: {self.server_host}:{self.server_port}")
        
        if self._check_server_availability():
            logger.info("Server is already available")
            print("✅ 服务器已经可用")
            return True
        
        logger.info(f"Server not available, waiting up to {self.max_wait_time} seconds (checking every {self.retry_interval} seconds)")
        print(f"⏳ 服务器暂不可用，等待最多 {self.max_wait_time} 秒 (每 {self.retry_interval} 秒检查一次)")
        
        start_time = time.time()
        while time.time() - start_time < self.max_wait_time:
            if not self._running:
                print("🛑 客户端停止运行")
                return False
                
            await asyncio.sleep(self.retry_interval)
            
            if self._check_server_availability():
                elapsed_time = time.time() - start_time
                logger.info(f"Server became available after {elapsed_time:.1f} seconds")
                print(f"✅ 服务器在 {elapsed_time:.1f} 秒后变为可用")
                return True
            else:
                elapsed_time = time.time() - start_time
                logger.info(f"Server still not available after {elapsed_time:.1f} seconds, continuing to wait...")
                print(f"⏳ 等待 {elapsed_time:.1f} 秒后服务器仍不可用，继续等待...")
        
        logger.error(f"Server did not become available within {self.max_wait_time} seconds")
        print(f"❌ 服务器在 {self.max_wait_time} 秒内未变为可用")
        return False
        
    async def connect_and_run(self):
        """Connect to the server and keep running, auto-reconnecting on drop"""
        # Pre-initialize VGGT reconstructor before attempting server connection,
        # so model can load while we wait and avoid reloading on reconnects.
        try:
            if (self.vggt_recon_mode or "off").lower() in ("single", "multi") and self._reconstructor is None:
                from client.reconstruction.vggt_recon import VGGTReconstructor, VGGTReconConfig
                recon_cfg = VGGTReconConfig(
                    model_path="./models/facebook/VGGT-1B",
                    preprocess_mode=self.vggt_preprocess_mode,
                    conf_percentile=self.vggt_conf_percentile,
                    max_points=self.vggt_max_points,
                    single_history=self.vggt_single_history,
                )
                self._reconstructor = VGGTReconstructor(recon_cfg)
                # Warmup in a background thread so we can still await server
                def _warmup():
                    try:
                        ok = self._reconstructor.warmup()
                        if ok:
                            print("🔥 VGGT 模型已加载完成")
                        else:
                            print("⚠️ VGGT 模型预加载失败，将在首次使用时重试")
                    except Exception as e:
                        print(f"⚠️ VGGT 预加载异常: {e}")
                t = threading.Thread(target=_warmup, daemon=True)
                t.start()
        except Exception as e:
            print(f"⚠️ 初始化 VGGT 重建器失败: {e}")

        # Initial wait for server (respect configured max wait time)
        if not await self._wait_for_server():
            logger.error("Cannot connect to server - server is not available")
            return

        uri = f"ws://{self.server_host}:{self.server_port}"

        while self._running:
            try:
                logger.info(f"Connecting to visualization server at {uri}")
                print(f"🌐 正在连接到可视化服务器: {uri}")

                async with websockets.connect(uri) as websocket:
                    logger.info(f"Successfully connected to visualization server at {uri}")
                    print(f"✅ 成功连接到可视化服务器: {uri}")
                    print(f"🎯 开始监听数据包...")
                    print("=" * 60)

                    # Reset config state so we accept a fresh config if needed
                    # but avoid reinitializing backends if they're already running
                    # (handled inside _handle_config)
                    self._config_received = False

                    async for message in websocket:
                        if not self._running:
                            break

                        try:
                            data = json.loads(message)
                            await self._handle_message(data)
                        except Exception as e:
                            logger.error(f"Error processing message: {e}")

                # If the async for loop exits without exception, connection closed
                if not self._running:
                    break
                logger.info("Connection to server closed; will attempt to reconnect")
                print("🔌 连接关闭，准备重连…")
            except websockets.exceptions.ConnectionClosed as e:
                logger.info(f"Connection closed: {e}")
                print("🔌 与服务器的连接已关闭，稍后重试…")
            except (OSError, ConnectionRefusedError) as e:
                # TCP-level connection issues (refused, DNS, etc.)
                logger.warning(f"Network connection failed: {e}")
                print("❌ 无法连接到服务器，稍后重试…")
            except Exception as e:
                logger.error(f"Connection error: {e}")
                print(f"❌ 连接错误: {e}")

            # Wait until server becomes available again (indefinitely)
            if not self._running:
                break
            print("⏳ 等待服务器恢复可用… (按 Ctrl+C 退出)")
            while self._running and not self._check_server_availability():
                await asyncio.sleep(self.retry_interval)
                print("🔄 继续等待服务器…")

            if self._running:
                print("✅ 检测到服务器恢复，正在重连…")

        # Cleanup when fully stopping
        print("🧹 开始清理资源…")
        self._cleanup()
    
    async def _handle_message(self, data):
        """Handle incoming messages from the server"""
        msg_type = data.get("type", "unknown")
        
        print(f"🎯 处理消息类型: {msg_type}")
        print(f"📊 消息数据结构:")
        # for key, value in data.items():
        #     if key == "image":
        #         # 图像数据太长，只显示前50字符
        #         print(f"  {key}: {str(value)[:50]}... (base64 image data)")
        #     elif isinstance(value, (list, dict)):
        #         print(f"  {key}: {type(value).__name__} with {len(value)} items")
        #     else:
        #         print(f"  {key}: {value}")
        
        if msg_type == "config":
            await self._handle_config(data)
        elif msg_type == "frame":
            await self._handle_frame(data)
        else:
            logger.warning(f"Unknown message type: {msg_type}")
            print(f"⚠️ 未知消息类型: {msg_type}")
            print(f"🔍 完整数据: {json.dumps(data, indent=2, ensure_ascii=False)}")
    
    async def _handle_config(self, config):
        """Handle initial configuration from server"""
        logger.info("Received configuration from server")
        print(f"⚙️ 配置数据:")
        print(json.dumps(config, indent=2, ensure_ascii=False))
        
        # Initialize visualization backends
        try:
            # Import visualization backends (assuming they exist)
            from .backends import make_backends
            with open("client/logic_chains.json", "r", encoding="utf-8") as f:
                logic_chains = json.load(f)
            # Avoid re-initializing backends on reconnect (e.g., Dash port conflicts)
            if self._viz_backends:
                print("⚙️ 已存在可视化后端，跳过重新初始化")
            else:
                self._viz_backends = make_backends(
                    self.visualization_backends,
                    output_dir=self.output_dir,
                    show_causal_graph=self.show_causal_graph,
                    logic_chains=logic_chains,  # transfer from config if needed
                    dash_interval_ms=self.dash_interval_ms,
                    plotly_refresh_ms=self.plotly_refresh_ms,
                    plotly_auto_open=self.plotly_auto_open,
                )
            # Initialize VGGT reconstructor if enabled
            if (self.vggt_recon_mode or "off").lower() in ("single", "multi") and self._reconstructor is None:
                try:
                    from client.reconstruction.vggt_recon import VGGTReconstructor, VGGTReconConfig
                    recon_cfg = VGGTReconConfig(
                        model_path="./models/facebook/VGGT-1B",
                        preprocess_mode=self.vggt_preprocess_mode,
                        conf_percentile=self.vggt_conf_percentile,
                        max_points=self.vggt_max_points,
                        single_history=self.vggt_single_history,
                    )
                    self._reconstructor = VGGTReconstructor(recon_cfg)
                    print("✅ 已启用 VGGT 三维重建模块")
                except Exception as e:
                    self._reconstructor = None
                    print(f"⚠️ VGGT 重建模块初始化失败: {e}")
            
            self._config_received = True
            logger.info(f"Initialized {len(self._viz_backends)} visualization backends")
            
        except ImportError as e:
            logger.error(f"Failed to import visualization backends: {e}")
            # Fallback to basic CV2 implementation
            self._init_basic_cv2_backend()
    
    def _init_basic_cv2_backend(self):
        """Initialize basic CV2 backend if full backends are not available"""
        try:
            import cv2
            self._cv2_available = True
            logger.info("Using basic CV2 visualization")
        except ImportError:
            logger.error("CV2 not available, visualization disabled")
            self._cv2_available = False
    
    async def _handle_frame(self, frame_data):
        """Handle incoming frame data with multiple images"""
        if not self._config_received:
            logger.warning("Received frame before configuration, skipping")
            print("⚠️ 在配置之前收到帧数据，跳过处理")
            return
        
        print(f"🖼️ 处理帧数据:")
        
        try:
            # Print frame metadata
            frame_index = frame_data.get("frame_index", "unknown")
            results = frame_data.get("results", [])
            
            print(f"  帧索引: {frame_index}")
            print(f"  检测结果数量: {len(results)}")
            
            # Print detection results
            for i, result in enumerate(results):
                print(f"  检测结果 {i+1}:")
                for key, value in result.items():
                    if isinstance(value, list) and len(value) > 5:
                        print(f"    {key}: [{', '.join(map(str, value[:3]))}...] (length: {len(value)})")
                    else:
                        print(f"    {key}: {value}")
            
            # Handle multiple images or single image
            if "images" in frame_data:
                # Multiple images handling (special layout for 3 images)
                images_base64 = frame_data["images"]
                print(f"  接收到 {len(images_base64)} 张图像")

                # Decode to PIL and ensure RGB
                imgs_pil = []
                for idx, img_base64 in enumerate(images_base64):
                    img_bytes = base64.b64decode(img_base64)
                    img_pil = Image.open(io.BytesIO(img_bytes))
                    w, h = img_pil.size   # PIL 的 size 顺序是 (width, height)
                    top, bottom = 40, h - 40
                    img_pil = img_pil.crop((0, top, w, bottom))
                    if img_pil.mode != 'RGB':
                        img_pil = img_pil.convert('RGB')
                    imgs_pil.append(img_pil)
                    print(f"    原始图像 {idx+1} 尺寸: {np.array(img_pil).shape}")

                if len(imgs_pil) == 3:
                    # Desired layout:
                    # - Top: first image resized to [480*2, 640*2] => (960, 1280)
                    # - Bottom: second and third images side-by-side, each [480, 640]
                    target_small_w, target_small_h = 640, 400
                    target_big_w, target_big_h = 640 * 2, 400 * 2

                    # Resize images accordingly
                    top_img = imgs_pil[0].resize((target_big_w, target_big_h), Image.BILINEAR)
                    bottom_left = imgs_pil[1].resize((target_small_w, target_small_h), Image.BILINEAR)
                    bottom_right = imgs_pil[2].resize((target_small_w, target_small_h), Image.BILINEAR)

                    top_arr = np.array(top_img)
                    bl_arr = np.array(bottom_left)
                    br_arr = np.array(bottom_right)

                    print(f"    顶部图像尺寸(放大): {top_arr.shape}")
                    print(f"    底部左图尺寸: {bl_arr.shape}")
                    print(f"    底部右图尺寸: {br_arr.shape}")

                    # Bottom row: concatenate horizontally to width 640*2
                    bottom_row = np.hstack([bl_arr, br_arr])

                    # Final: stack vertically -> [480*3, 640*2]
                    img_array = np.vstack([top_arr, bottom_row])
                    print(f"  拼接后图像尺寸: {img_array.shape} (期望 [1440, 1280, 3])")
                else:
                    # Fallback: simple vertical stack for non-3-image cases
                    img_arrays = [np.array(img) for img in imgs_pil]
                    for idx, arr in enumerate(img_arrays):
                        print(f"    图像 {idx+1} 尺寸: {arr.shape}")
                    img_array = np.vstack(img_arrays)
                    print(f"  竖直拼接后图像尺寸: {img_array.shape}")
                # VGGT multi-view reconstruction (optional)
                vggt_pc = None
                if self._reconstructor is not None and (self.vggt_recon_mode or "off").lower() == "multi":
                    try:
                        imgs_np = [np.array(p) for p in imgs_pil]
                        vggt_pc = self._reconstructor.reconstruct_multi(imgs_np)
                    except Exception as e:
                        print(f"⚠️ VGGT 多视图重建失败: {e}")
                
            elif "image" in frame_data:
                # Single image (backward compatibility)
                img_base64 = frame_data["image"]
                print(f"  图像数据长度: {len(img_base64)} 字符")
                
                img_bytes = base64.b64decode(img_base64)
                print(f"  解码后图像字节数: {len(img_bytes)}")
                
                img_pil = Image.open(io.BytesIO(img_bytes))
                img_array = np.array(img_pil)
                print(f"  图像尺寸: {img_array.shape}")
                # VGGT single-view (windowed) reconstruction (optional)
                vggt_pc = None
                if self._reconstructor is not None and (self.vggt_recon_mode or "off").lower() == "single":
                    try:
                        vggt_pc = self._reconstructor.update_single_and_reconstruct(np.array(img_pil.convert('RGB')))
                    except Exception as e:
                        print(f"⚠️ VGGT 单视图重建失败: {e}")
            else:
                logger.error("No image data in frame")
                return
            
            # Extract results
            results = frame_data["results"]
            frame_index = frame_data["frame_index"]
            
            # Inject VGGT point cloud into results if any
            try:
                vggt_pc_local = locals().get('vggt_pc', None)
                if isinstance(vggt_pc_local, dict) and vggt_pc_local.get("points") is not None and vggt_pc_local.get("colors") is not None:
                    pts = np.asarray(vggt_pc_local["points"]).tolist()
                    cols = np.asarray(vggt_pc_local["colors"]).tolist()
                    results = list(results) + [{
                        "point_cloud": {
                            "points": pts,
                            "colors": cols,
                            "num_points": int(vggt_pc_local.get("num_points", len(pts))),
                            "source": vggt_pc_local.get("source", "unknown"),
                        }
                    }]
            except Exception:
                pass

            # Update visualization backends
            if self._viz_backends:
                for backend in self._viz_backends:
                    try:
                        backend.update(img_array, results, frame_index)
                    except Exception as e:
                        logger.warning(f"Backend update failed: {e}")
            elif hasattr(self, '_cv2_available') and self._cv2_available:
                self._basic_cv2_display(img_array, results, frame_index)
                
        except Exception as e:
            logger.error(f"Error handling frame: {e}")
    
    def _basic_cv2_display(self, img_array, results, frame_index):
        """Basic CV2 visualization when full backends are not available"""
        try:
            import cv2
            
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Create window and position it on the left side of the screen
            window_name = 'Remote Visualization'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            
            # Position window on the left side (x=0) with some top margin (y=50)
            cv2.moveWindow(window_name, 0, 50)
            
            # Resize window to fit the concatenated images nicely
            # For 3 images of [480, 640], the concatenated size is [1440, 640]
            # Scale down for better display
            display_height = 720  # Adjust as needed
            aspect_ratio = img_bgr.shape[1] / img_bgr.shape[0]
            display_width = int(display_height * aspect_ratio)
            cv2.resizeWindow(window_name, display_width, display_height)
            
            # Draw bounding boxes and labels
            for result in results:
                bbox = result.get('bbox', [])
                label = result.get('label', 'unknown')
                
                if len(bbox) == 4:
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # Draw bounding box
                    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label
                    cv2.putText(img_bgr, label, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Show top attributes and affordances
                    top_attrs = result.get('top_attrs', [])[:3]  # Show top 3
                    top_affs = result.get('top_affs', [])[:3]
                    attr_probs = result.get('attr_probs', [])[:len(top_attrs)]
                    aff_probs = result.get('aff_probs', [])[:len(top_affs)]
                    
                    # Draw attributes
                    for i, (attr, prob) in enumerate(zip(top_attrs, attr_probs)):
                        text = f"{attr}: {prob:.2f}"
                        cv2.putText(img_bgr, text, (x1, y2 + 15 + i*15),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                    
                    # Draw affordances
                    for i, (aff, prob) in enumerate(zip(top_affs, aff_probs)):
                        text = f"{aff}: {prob:.2f}"
                        cv2.putText(img_bgr, text, (x2 - 150, y2 + 15 + i*15),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            # Display frame index
            cv2.putText(img_bgr, f"第 {frame_index} 帧", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Show image
            cv2.imshow(window_name, img_bgr)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self._running = False
                
        except Exception as e:
            logger.error(f"CV2 display error: {e}")
    
    def _cleanup(self):
        """Clean up resources"""
        if self._viz_backends:
            for backend in self._viz_backends:
                try:
                    backend.stop()
                except Exception:
                    pass
        
        if hasattr(self, '_cv2_available') and self._cv2_available:
            try:
                import cv2
                cv2.destroyAllWindows()
            except ImportError:
                pass
        
        logger.info("Visualization client cleaned up")
    
    def stop(self):
        """Stop the client"""
        self._running = False


async def main():
    parser = argparse.ArgumentParser(description="Remote Visualization Client")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8766, help="Server port")
    parser.add_argument("--backends", nargs="+", default=["cv2"], 
                       choices=["cv2", "dash", "plotly"],
                       help="Visualization backends to use")
    parser.add_argument("--output-dir", default="remote_visualization_output",
                       help="Output directory for saved visualizations")
    parser.add_argument("--no-causal-graph", action="store_true",
                       help="Disable causal graph display")
    parser.add_argument("--dash-interval", type=int, default=1000,
                       help="Dash update interval in milliseconds")
    parser.add_argument("--plotly-refresh", type=int, default=1000,
                       help="Plotly refresh interval in milliseconds")
    parser.add_argument("--no-plotly-auto-open", action="store_true",
                       help="Disable auto-opening Plotly browser")
    parser.add_argument("--no-wait", action="store_true",
                       help="Don't wait for server to become available")
    parser.add_argument("--max-wait-time", type=int, default=300,
                       help="Maximum time to wait for server (seconds)")
    parser.add_argument("--retry-interval", type=int, default=5,
                       help="Interval between server availability checks (seconds)")
    parser.add_argument("--vggt-recon", type=str, default="off", choices=["off", "single", "multi"],
                       help="Enable VGGT 3D reconstruction: off | single | multi")
    parser.add_argument("--vggt-conf", type=float, default=50.0,
                       help="Confidence percentile filter for points (0..100)")
    parser.add_argument("--vggt-max-points", type=int, default=40000,
                       help="Max number of points to render")
    parser.add_argument("--vggt-preprocess", type=str, default="crop", choices=["crop", "pad"],
                       help="Preprocess mode for VGGT input")
    parser.add_argument("--vggt-single-history", type=int, default=6,
                       help="Window size for single-view reconstruction")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🎬 可视化客户端启动")
    print("=" * 50)
    print(f"🌐 服务器地址: {args.host}:{args.port}")
    print(f"🎨 可视化后端: {args.backends}")
    print(f"📁 输出目录: {args.output_dir}")
    print(f"⏰ 等待服务器: {not args.no_wait}")
    if not args.no_wait:
        print(f"⏱️ 最大等待时间: {args.max_wait_time}秒")
        print(f"🔄 重试间隔: {args.retry_interval}秒")
    print("=" * 50)
    
    logging.info(args)
    client = WebSocketVisualizationClient(
        server_host=args.host,
        server_port=args.port,
        visualization_backends=args.backends,
        output_dir=args.output_dir,
        show_causal_graph=not args.no_causal_graph,
        dash_interval_ms=args.dash_interval,
        plotly_refresh_ms=args.plotly_refresh,
        plotly_auto_open=not args.no_plotly_auto_open,
        wait_for_server=not args.no_wait,
        max_wait_time=args.max_wait_time,
        retry_interval=args.retry_interval,
        vggt_recon_mode=args.vggt_recon,
        vggt_conf_percentile=args.vggt_conf,
        vggt_max_points=args.vggt_max_points,
        vggt_preprocess_mode=args.vggt_preprocess,
        vggt_single_history=args.vggt_single_history,
    )
    
    try:
        await client.connect_and_run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        client.stop()


if __name__ == "__main__":
    asyncio.run(main())
