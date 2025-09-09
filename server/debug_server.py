import asyncio
import json
import logging
import base64
import numpy as np
from PIL import Image
import websockets
import argparse
from datetime import datetime
import io
import os
import random

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DebugVisualizationServer:
    """
    Debug server that publishes config and frame messages for client testing.
    Simulates the policy server's visualization stream.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8766,
        image_paths: list = None,
        frame_interval: float = 1.0,
        loop_frames: bool = True,
        max_frames: int = None
    ):
        self.host = host
        self.port = port
        self.image_paths = image_paths or []
        self.frame_interval = frame_interval
        self.loop_frames = loop_frames
        self.max_frames = max_frames
        
        self.clients = set()
        self.frame_index = 0
        self.running = False
        
    def generate_config(self):
        """Generate a sample configuration message"""
        config = {
            "type": "config",
            "timestamp": datetime.now().isoformat(),
            "settings": {
                "model_name": "debug_model",
                "input_shape": [640, 480, 3],
                "classes": ["person", "balloon", "weight", "object"],
                "confidence_threshold": 0.5
            },
            "visualization": {
                "show_bboxes": True,
                "show_labels": True,
                "show_confidence": True,
                "colors": {
                    "person": [255, 0, 0],
                    "balloon": [0, 255, 0],
                    "weight": [0, 0, 255],
                    "object": [255, 255, 0]
                }
            },
            "logic_chains": [
                {
                    "id": "chain_1",
                    "name": "Person Detection Chain",
                    "steps": ["detect", "classify", "track"]
                }
            ]
        }
        return config
    
    def load_image_as_base64(self, image_path):
        """Load an image and convert to base64, resize to [480, 640]"""
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize to fixed shape [480, 640]
                img = img.resize((640, 480), Image.LANCZOS)
                img_array = np.array(img)
                
                buffer = io.BytesIO()
                Image.fromarray(img_array).save(buffer, format='PNG')
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                return img_base64, (480, 640)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None, (480, 640)
    
    def generate_synthetic_image(self):
        """Generate a synthetic test image with fixed shape [480, 640]"""
        height, width = 480, 640
        img_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        center_x, center_y = width // 2, height // 2
        radius = 50
        y, x = np.ogrid[:height, :width]
        mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
        img_array[mask] = [255, 100, 100]
        
        img = Image.fromarray(img_array)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return img_base64, (height, width)
    
    def generate_frame(self):
        """Generate a sample frame message with 3 random images"""
        # Generate 3 random images with shape [480, 640]
        images_base64 = []
        for i in range(3):
            # Generate different patterns for each image
            height, width = 480, 640
            img_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            
            # Add different patterns to distinguish the images
            if i == 0:  # First image - horizontal gradient
                for y in range(height):
                    img_array[y, :, 0] = int(255 * y / height)
            elif i == 1:  # Second image - vertical gradient
                for x in range(width):
                    img_array[:, x, 1] = int(255 * x / width)
            else:  # Third image - diagonal pattern
                for y in range(height):
                    for x in range(width):
                        img_array[y, x, 2] = int(255 * (x + y) / (width + height))
            
            # Add some random circles
            num_circles = random.randint(2, 5)
            for _ in range(num_circles):
                center_x = random.randint(50, width - 50)
                center_y = random.randint(50, height - 50)
                radius = random.randint(20, 50)
                color = [random.randint(100, 255) for _ in range(3)]
                
                y, x = np.ogrid[:height, :width]
                mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
                img_array[mask] = color
            
            # Convert to base64 with JPEG compression for smaller size
            img = Image.fromarray(img_array)
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=50)  # Lower quality for smaller size
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            images_base64.append(img_base64)
        
        num_detections = random.randint(1, 5)
        results = []
        
        for _ in range(num_detections):
            x1 = random.randint(0, width - 100)
            y1 = random.randint(0, height - 100)
            x2 = min(x1 + random.randint(50, 150), width)
            y2 = min(y1 + random.randint(50, 150), height)
            
            class_name = random.choice(["person", "balloon", "weight", "object"])
            confidence = random.uniform(0.5, 0.99)
            
            # Generate random probabilities for attributes and affordances

            
            # Select random attributes and affordances
            selected_attrs = ["covered", "reversed", "filled", "plastic", "metal"]
            selected_affs = ["operate"]
            num_attrs = len(selected_attrs)
            num_affs = len(selected_affs)
            
            # Generate probabilities (normalized to sum to ~1)
            attr_probs = [random.random() for _ in range(num_attrs)]
            attr_sum = sum(attr_probs)
            attr_probs = [p / attr_sum for p in attr_probs]
            
            aff_probs = [random.random() for _ in range(num_affs)]
            aff_sum = sum(aff_probs)
            aff_probs = [p for p in aff_probs]
            
            results.append({
                "bbox": [x1, y1, x2, y2],
                "class": class_name,
                "confidence": confidence,
                "track_id": random.randint(1, 10),
                "top_attrs": selected_attrs,
                "attr_probs": attr_probs,
                "top_affs": selected_affs,
                "aff_probs": aff_probs
            })
        
        frame = {
            "type": "frame",
            "frame_index": self.frame_index,
            "timestamp": datetime.now().isoformat(),
            "images": images_base64,  # Changed to multiple images
            "results": results,
            "metadata": {
                "processing_time_ms": random.uniform(10, 50),
                "detection_count": len(results),
                "image_shape": [480, 640, 3],
                "num_images": 3
            }
        }
        
        return frame
    
    async def handle_client(self, websocket):
        """Handle a connected client"""
        logger.info(f"Client connected: {websocket.remote_address}")
        self.clients.add(websocket)
        
        try:
            config = self.generate_config()
            await websocket.send(json.dumps(config))
            logger.info(f"Sent config to client: {websocket.remote_address}")
            
            await websocket.wait_closed()
            
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {websocket.remote_address}")
        except Exception as e:
            logger.error(f"Error handling client: {e}")
        finally:
            self.clients.discard(websocket)
    
    async def broadcast_frames(self):
        """Broadcast frames to all connected clients"""
        while self.running:
            if self.clients:
                frame = self.generate_frame()
                message = json.dumps(frame)
                
                disconnected_clients = set()
                for client in self.clients:
                    try:
                        await client.send(message)
                        logger.info(f"Sent frame {self.frame_index} to {client.remote_address}")
                    except websockets.exceptions.ConnectionClosed:
                        disconnected_clients.add(client)
                    except Exception as e:
                        logger.error(f"Error sending frame to client: {e}")
                        disconnected_clients.add(client)
                
                self.clients -= disconnected_clients
                
                self.frame_index += 1
                
                if self.max_frames and self.frame_index >= self.max_frames:
                    if self.loop_frames:
                        self.frame_index = 0
                        logger.info("Looping back to frame 0")
                    else:
                        logger.info(f"Reached max frames ({self.max_frames}), stopping")
                        self.running = False
                        break
            
            await asyncio.sleep(self.frame_interval)
    
    async def start(self):
        """Start the debug server"""
        self.running = True
        logger.info(f"Starting debug server on {self.host}:{self.port}")
        print(f"🚀 Debug visualization server starting on ws://{self.host}:{self.port}")
        print(f"📊 Frame interval: {self.frame_interval}s")
        print(f"🔄 Loop frames: {self.loop_frames}")
        if self.max_frames:
            print(f"🎯 Max frames: {self.max_frames}")
        
        broadcast_task = asyncio.create_task(self.broadcast_frames())
        
        async with websockets.serve(self.handle_client, self.host, self.port):
            print(f"✅ Server is running. Waiting for clients...")
            print(f"📡 Connect clients to ws://{self.host}:{self.port}")
            
            try:
                await asyncio.Future()
            except KeyboardInterrupt:
                logger.info("Server shutdown requested")
                print("\n🛑 Shutting down server...")
            finally:
                self.running = False
                broadcast_task.cancel()
                try:
                    await broadcast_task
                except asyncio.CancelledError:
                    pass


def main():
    parser = argparse.ArgumentParser(description='Debug Visualization Server')
    parser.add_argument('--host', type=str, default='localhost',
                        help='Server host address')
    parser.add_argument('--port', type=int, default=8766,
                        help='Server port')
    parser.add_argument('--images', nargs='+', type=str,
                        help='Paths to images to use for frames')
    parser.add_argument('--interval', type=float, default=1.0,
                        help='Interval between frames in seconds')
    parser.add_argument('--no-loop', action='store_true',
                        help='Do not loop frames')
    parser.add_argument('--max-frames', type=int,
                        help='Maximum number of frames to send')
    
    args = parser.parse_args()
    
    image_paths = []
    if args.images:
        for img_path in args.images:
            if os.path.exists(img_path):
                image_paths.append(img_path)
                print(f"✅ Added image: {img_path}")
            else:
                print(f"⚠️  Image not found: {img_path}")
    
    if not image_paths:
        default_images = ['resources/balloon.png', 'resources/weight.png']
        for img_path in default_images:
            if os.path.exists(img_path):
                image_paths.append(img_path)
                print(f"✅ Using default image: {img_path}")
    
    if not image_paths:
        print("ℹ️  No images found, will generate synthetic images")
    
    server = DebugVisualizationServer(
        host=args.host,
        port=args.port,
        image_paths=image_paths,
        frame_interval=args.interval,
        loop_frames=not args.no_loop,
        max_frames=args.max_frames
    )
    
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        print("\n👋 Server stopped")


if __name__ == "__main__":
    main()