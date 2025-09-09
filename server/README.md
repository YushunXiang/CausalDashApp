# Debug Visualization Server

A WebSocket server that continuously publishes "config" and "frame" messages for debugging visualization clients.

## Features

- Publishes initial configuration message when client connects
- **Continuously broadcasts frame messages** with fixed shape [480, 640]
- Includes **attr_probs** and **aff_probs** for each detection result
- Simulates realistic detection results with attributes and affordances
- Supports custom images (auto-resized to 480x640) or generates synthetic test images
- Configurable frame interval and looping behavior

## Message Formats

### Config Message
```json
{
  "type": "config",
  "timestamp": "ISO-8601 timestamp",
  "settings": {
    "model_name": "debug_model",
    "input_shape": [640, 480, 3],
    "classes": ["person", "balloon", "weight", "object"],
    "confidence_threshold": 0.5
  },
  "visualization": {
    "show_bboxes": true,
    "show_labels": true,
    "show_confidence": true,
    "colors": {...}
  },
  "logic_chains": [...]
}
```

### Frame Message
```json
{
  "type": "frame",
  "frame_index": 0,
  "timestamp": "ISO-8601 timestamp",
  "image": "base64-encoded-png",
  "results": [
    {
      "bbox": [x1, y1, x2, y2],
      "class": "person",
      "confidence": 0.95,
      "track_id": 1,
      "top_attrs": ["red", "large", "round"],
      "attr_probs": [0.6, 0.3, 0.1],
      "top_affs": ["graspable", "throwable", "liftable"],
      "aff_probs": [0.5, 0.3, 0.2]
    }
  ],
  "metadata": {
    "processing_time_ms": 25.3,
    "detection_count": 3,
    "image_shape": [480, 640, 3]
  }
}
```

## Usage

### Quick Start
```bash
# Use the convenience script
./server.sh

# Or run directly
python3 server/debug_server.py
```

### Command Line Options
```bash
python3 server/debug_server.py [options]

Options:
  --host HOST           Server host address (default: localhost)
  --port PORT           Server port (default: 8766)
  --images IMG1 IMG2    Paths to images for frames
  --interval SECONDS    Interval between frames (default: 1.0)
  --no-loop            Don't loop frames
  --max-frames N       Maximum number of frames to send
```

### Examples

```bash
# Use default settings with included images
python3 server/debug_server.py

# Custom images and faster frame rate
python3 server/debug_server.py --images img1.png img2.png --interval 0.5

# Send only 100 frames without looping
python3 server/debug_server.py --max-frames 100 --no-loop

# Listen on all interfaces
python3 server/debug_server.py --host 0.0.0.0 --port 9000
```

## Testing with Client

Run the server and client in separate terminals:

```bash
# Terminal 1 - Start server
./server.sh

# Terminal 2 - Start client
./client.sh
```

The server will:
1. Wait for client connections
2. Send config message immediately upon connection
3. **Continuously send frame messages** at the specified interval
4. Generate random bounding boxes with:
   - Detection class and confidence
   - Attributes (color, size, shape) with probabilities
   - Affordances (graspable, throwable, etc.) with probabilities
5. All images are guaranteed to be shape [480, 640, 3] for consistent debugging