# Causal Visualization

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a debug visualization system for object detection and affordance analysis. It consists of a WebSocket server that simulates detection results and a client with multiple visualization backends (cv2, Dash, Plotly).

## Key Commands

### Running the System

```bash
# Start the debug server (simulates detection results)
python server/debug_server.py [--host HOST] [--port PORT] [--image IMAGE_PATH] [--size WIDTH HEIGHT]

# Start the visualization client
python client/visual_client.py [--config CONFIG_FILE]

# Run both with default settings
python server/debug_server.py  # Terminal 1
python client/visual_client.py  # Terminal 2
```

### Development Commands

```bash
# Install dependencies (if requirements.txt exists)
pip install websockets opencv-python-headless dash plotly pillow numpy

# Test visualization backends individually
python -m client.backends.dash_viz  # Test Dash backend
python -m client.backends.plotly_backend  # Test Plotly backend
```

## Architecture

### Core Components

1. **WebSocket Server** (`server/debug_server.py`)
   - Broadcasts detection results at 10 FPS
   - Generates synthetic or image-based frames
   - Sends bounding boxes, attributes, affordances, and causal chains

2. **Visualization Client** (`client/visual_client.py`) 
   - Connects to WebSocket server
   - Processes detection results
   - Dispatches to multiple visualization backends

3. **Visualization Backends** (`client/backends/`)
   - **cv2**: Real-time OpenCV display
   - **dash**: Web-based dashboard on port 8060
   - **plotly**: Interactive HTML output
   - **graph.py**: Causal graph visualization logic

### Data Flow

```
DebugServer → WebSocket → VisualClient → Backend(s) → Display
     ↓                          ↓                        ↓
Synthetic/Image Data    Detection Results      CV2/Dash/Plotly Views
```

### Message Protocol

Server sends JSON messages containing:
- `frame`: Base64-encoded image data
- `detections`: List of objects with bounding boxes, classes, attributes, affordances
- `mask` (optional): Segmentation masks
- `config`: Visualization settings (colors, labels, etc.)
- `logic_chains`: Attribute-affordance relationships

## Configuration

### Client Configuration (`visualization_config.json`)
```json
{
  "server_host": "127.0.0.1",
  "server_port": 8765,
  "visualization_backends": ["cv2", "dash", "plotly"],
  "show_causal_graph": true,
  "output_dir": "./output"
}
```

### Logic Chains (`client/logic_chains.json`)
Defines attribute-affordance relationships:
```json
[
  {
    "attribute": "plastic",
    "affordance": "operate", 
    "is_positive_affect": true
  }
]
```

## Key Implementation Details

### Graph Visualization (`client/backends/graph.py`)

The causal graph system visualizes attribute-affordance relationships:
- **Node size**: Proportional to probability values (20-60px range)
- **Edge brightness**: Based on node-to-node probability (opacity 0.1-1.0)
- **Edge styles**: 
  - Dashed gray: No causal relationship
  - Solid red with variable opacity: Causal relationship (strength shown by opacity)
- **Chinese translations**: Built-in dictionary for UI labels

Key functions:
- `build_causal_graph_figure()`: Main graph construction
- `prob2size()`, `prob2alpha()`: Probability-to-visual mappings
- `hsv_to_rgb()`: Color generation

### Backend Integration Pattern

All backends follow this interface:
```python
class Backend:
    def start(self) -> None
    def stop(self) -> None  
    def update(self, results: List[Dict[str, Any]]) -> None
```

## Recent Modifications

- Added edge brightness based on probability values between nodes
- Node sizes now proportional to their probability values
- All node pairs connected with lines (dashed gray for no causality)
- Chinese translations for UI elements

## VS Code Integration

The project includes `.vscode/settings.json` for proper Python environment configuration with the repository root as the Python path.

## Testing Approach

When testing visualizations:
1. Start debug server with test image or synthetic mode
2. Run client with desired backends
3. Verify real-time updates in cv2 window
4. Check Dash web interface at http://localhost:8060
5. Review generated HTML in output directory for Plotly

## Important Notes

- The system is designed for debugging and testing, not production use
- All network communication is unencrypted WebSocket
- Frame rate is fixed at 10 FPS in debug server
- Visualization backends can be run independently or together