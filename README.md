# SkywaysTracker

A Python application for video frame annotation that allows you to create bounding boxes while navigating through video frames.

## Features

- Navigate through all frames of any video file supported by OpenCV
- Show video metadata (resolution, frame count, FPS, duration)
- **Hold-to-Label Mode**: Hold 'L' key and click+drag to create red bounding boxes
- Automatic frame advancement when releasing the 'L' key
- Interactive box creation with center-based scaling
- Real-time frame counter display
- Automatic annotation saving in custom format
- Simple keyboard controls (Q to quit, hold L to label)
- Comprehensive error handling
- Cross-platform compatibility

## Requirements

- Python 3.6 or higher
- OpenCV (opencv-python)

## Installation

1. Clone or download this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the application from the command line with a video file path:

```bash
python main.py <path_to_video_file>
```

### Examples

```bash
# Display first frame of a local video file
python main.py ./videos/sample.mp4

# Display first frame using absolute path
python main.py C:\Users\username\Videos\movie.avi

# Display first frame of a video with spaces in the name
python main.py "C:\Users\username\Videos\my video.mov"
```

## Controls

### Basic Controls
- **Q** or **q**: Quit the application
- **Hold L**: Enable labeling mode (boxes appear while held)
- **Release L** (or press any other key): Clear boxes and advance to next frame
- **X** (window close button): Close the application

### Labeling Workflow
- **Hold L Key**: Enables labeling mode for the current frame
- **Click + Drag** (while holding L): Create red bounding boxes that scale from center
- **Release L Key**: Clears all boxes and automatically advances to the next frame
- **Frame Navigation**: Automatic - each L key release moves to the next frame

### Visual Indicators
- **Frame Counter**: Shows current frame number (e.g., "Frame: 5/120")
- **Red Boxes**: Visible only while L key is held down
- **Auto-Advance**: Seamless progression through video frames

## Supported Video Formats

The application supports all video formats that OpenCV can read, including:
- MP4 (.mp4)
- AVI (.avi)
- MOV (.mov)
- MKV (.mkv)
- WMV (.wmv)
- FLV (.flv)
- And many more...

## Error Handling

The application includes comprehensive error handling for:
- Non-existent video files
- Invalid file paths
- Corrupted or unreadable video files
- Unsupported video formats
- Permission issues

## Video Information Display

When a video is successfully loaded, the application displays:
- Video resolution (width × height)
- Total number of frames
- Frames per second (FPS)
- Video duration in seconds

## Labeling Workflow

1. **Load Video**: Run the application with a video file path
2. **Start on Frame 1**: Application begins with the first frame
3. **Hold L Key**: Hold down 'L' to enable labeling mode on current frame
4. **Create Boxes**: While holding L, click and drag to create red bounding boxes
5. **Multiple Boxes**: Create as many boxes as needed on the current frame
6. **Release L Key**: Release L (or press any other key) to clear boxes and advance to next frame
7. **Repeat**: Continue through all frames in the video
8. **Exit**: Press 'Q' to quit and auto-save annotations

**Note**: Boxes are temporary and only visible while the L key is held. Each frame starts fresh with no existing boxes.

## Annotation File Format

When you exit the application, your annotations are automatically saved to a file named `{videoName}.annotations` in the same directory as `main.py`.

**Format**: Each line represents one bounding box:
```
V x_center y_center width height
```

**Example**:
```
V 250 150 50 75
V 400 300 100 80
```

Where:
- `V` = Object identifier (fixed)
- `x_center` = X coordinate of box center
- `y_center` = Y coordinate of box center  
- `width` = Box width in pixels
- `height` = Box height in pixels

## License

This project is open source and available under the [MIT License](LICENSE).