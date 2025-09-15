# SkywaysTracker

A comprehensive Python toolkit for video frame annotation and validation, featuring AI-powered object detection and manual labeling capabilities.

## Overview

This project consists of two main tools:

1. **LabelTool.py** - Advanced video annotation tool with YOLO-World AI detection and manual labeling
2. **ValidationTool.py** - Annotation visualization and validation tool

## Features

### LabelTool.py - Advanced Annotation Tool

#### AI-Powered Detection
- **YOLO-World Integration**: Text-based object detection using natural language prompts
- **Automatic Object Detection**: AI automatically detects objects based on text descriptions
- **Confidence Thresholding**: Only accepts detections above 30% confidence
- **Fallback to Manual**: Seamlessly switches to manual labeling when AI detection fails

#### Manual Labeling
- **Center-Based Scaling**: Click and drag from center point for intuitive box creation
- **Real-Time Visual Feedback**: Smooth drawing updates at 60 FPS
- **Multiple Box Support**: Create multiple bounding boxes per frame
- **Smart Validation**: Prevents creation of invalid or too-small boxes

#### Object Tracking
- **CSRT Tracker Integration**: Advanced object tracking across frames
- **Prediction System**: AI predicts object location on subsequent frames
- **Manual Correction**: Fix tracker predictions with manual adjustments
- **Automatic Reinitialization**: Tracker adapts to corrections

#### Workflow Management
- **Three Action Types**: Label (L), Skip (S), Invisible (I)
- **Frame Navigation**: Automatic progression through video frames
- **Smart Mode Switching**: Context-aware interface based on available actions
- **Auto-Save**: Annotations automatically saved on exit

### ValidationTool.py - Annotation Viewer

#### Visualization
- **Color-Coded Display**: Green boxes for visible objects, status labels for skipped/invisible frames
- **Frame Navigation**: Navigate forward/backward with customizable skip amounts
- **Status Indicators**: Clear visual feedback for different annotation types
- **Responsive Interface**: Smooth navigation with keyboard shortcuts

#### Validation Features
- **Annotation Parsing**: Loads and validates annotation files
- **Error Detection**: Identifies malformed annotation entries
- **Frame-by-Frame Review**: Detailed inspection of annotation quality
- **Export Ready**: Validates annotations for downstream processing

## Requirements

- Python 3.6 or higher
- OpenCV (opencv-python)
- Ultralytics (for YOLO-World functionality)

## Installation

1. Clone or download this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Optional: YOLO-World Model

The LabelTool will automatically download the YOLO-World model (`yolov8s-world.pt`) on first use. Ensure you have an internet connection for the initial download.

## Usage

### LabelTool.py - Creating Annotations

#### Basic Usage (Manual Labeling)
```bash
python LabelTool.py video.mp4
```

#### With AI Detection
```bash
python LabelTool.py video.mp4 --prompt "person with red shirt"
```

#### With Custom Annotations File
```bash
python LabelTool.py video.mp4 custom_annotations.annotations --prompt "robot"
```

#### Examples
```bash
# Manual labeling mode
python LabelTool.py ./videos/sample.mp4

# AI detection for specific objects
python LabelTool.py video.mp4 --prompt "car"
python LabelTool.py video.mp4 --prompt "person walking"
python LabelTool.py video.mp4 --prompt "dog running"

# With custom annotations file
python LabelTool.py video.mp4 my_annotations.annotations --prompt "bicycle"
```

### ValidationTool.py - Viewing Annotations

#### Basic Usage
```bash
python ValidationTool.py video.mp4
```

#### With Custom Annotations File
```bash
python ValidationTool.py video.mp4 custom_annotations.annotations
```

#### Examples
```bash
# View annotations with auto-detected file
python ValidationTool.py ./videos/sample.mp4

# View with specific annotations file
python ValidationTool.py video.mp4 my_annotations.annotations
```

## Controls

### LabelTool.py Controls

#### When AI Detection is Available
- **L/l**: Run YOLO-World text detection
- **A/a**: Accept AI prediction
- **F/f**: Fix AI prediction manually
- **S/s**: Skip this frame
- **I/i**: Mark frame as invisible
- **Q/q**: Quit and save annotations

#### When in Manual Mode
- **L/l**: Enter manual labeling mode
- **F/f**: Enter manual labeling mode (fallback)
- **S/s**: Skip this frame
- **I/i**: Mark frame as invisible
- **Q/q**: Quit and save annotations

#### During Labeling/Fixing
- **Mouse**: Click and drag to create bounding boxes
- **Q/q**: Quit and save annotations

### ValidationTool.py Controls

- **n**: Next frame
- **N**: Skip 10 frames forward
- **p**: Previous frame
- **P**: Skip 10 frames back
- **Q/q**: Quit viewer

## Supported Video Formats

Both tools support all video formats that OpenCV can read, including:
- MP4 (.mp4)
- AVI (.avi)
- MOV (.mov)
- MKV (.mkv)
- WMV (.wmv)
- FLV (.flv)
- And many more...

## Annotation File Format

Annotations are saved in a custom format with one line per frame:

### Format
```
[STATUS] x_center y_center width height
```

### Status Types
- **V**: Visible object with bounding box
- **S**: Skipped frame
- **I**: Invisible frame (object not visible)

### Examples
```
V 250 150 50 75
S -1 -1 -1 -1
I -1 -1 -1 -1
V 400 300 100 80
```

### Coordinate System
- **x_center**: X coordinate of bounding box center
- **y_center**: Y coordinate of bounding box center
- **width**: Width of bounding box in pixels
- **height**: Height of bounding box in pixels
- **-1 -1 -1 -1**: Placeholder for skipped/invisible frames

## Workflow Examples

### Complete Annotation Workflow

1. **Start with AI Detection**:
   ```bash
   python LabelTool.py video.mp4 --prompt "person"
   ```

2. **Review and Validate**:
   ```bash
   python ValidationTool.py video.mp4
   ```


### Manual Annotation Workflow

1. **Manual Labeling**:
   ```bash
   python LabelTool.py video.mp4
   ```

2. **Review Results**:
   ```bash
   python ValidationTool.py video.mp4
   ```

## Technical Details

### YOLO-World Integration
- Uses `yolov8s-world.pt` model for text-based object detection
- Supports natural language prompts for object detection
- Automatic fallback to manual labeling when detection fails
- Confidence threshold of 30% for reliable detections

### Object Tracking
- CSRT (Channel and Spatial Reliability Tracker) implementation
- Automatic tracker initialization on first labeled frame
- Prediction correction and reinitialization
- Robust handling of tracking failures


## Error Handling

Both tools include comprehensive error handling for:
- Non-existent video files
- Invalid file paths
- Corrupted or unreadable video files
- Unsupported video formats
- Permission issues
- Malformed annotation files
- YOLO-World model download failures

## Video Information Display

When a video is successfully loaded, both tools display:
- Video resolution (width × height)
- Total number of frames
- Frames per second (FPS)
- Video duration in seconds

