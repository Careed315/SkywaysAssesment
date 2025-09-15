#!/usr/bin/env python3
# Video Frame Annotation CLI Tool
#
# Usage: python main.py videoPath.mp4 [--prompt "Custom prompt"] [--annotations pathToAnnotationsFile.annotations]
#
# Features:
# - Display video frames using OpenCV
# - Three actions per frame: Label (L/l), Skip (S/s), Invisible (I/i)
# - Quit with Q/q
# - Frame navigation with keyboard shortcuts
# - Optional annotations file support

import argparse
import cv2
import os
import sys
import time
from typing import Dict, List, Optional, Tuple
from enum import Enum

# Try to import YOLO-World dependencies
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: Ultralytics YOLO not available. Install with: pip install ultralytics")


class YOLOWorldDetector:
    # YOLO-World text-based object detection using Ultralytics
    
    def __init__(self, prompt: str):
        self.prompt = prompt
        self.model = None
        self.is_initialized = False
        self.is_yolo_world = False  # Track if we have true YOLO-World capability
        
        if not YOLO_AVAILABLE:
            print("YOLO-World not available - text detection disabled")
            return
            
        try:
            # Initialize YOLO-World model
            print(f"Initializing YOLO-World for prompt: '{prompt}'")
            print("Downloading YOLO-World model (this may take a moment on first run)...")
            
            # Initialize YOLO-World model using the official Ultralytics approach
            # YOLO-World models are available through Ultralytics Hub
            try:
                self.model = YOLO("yolov8s-world.pt")  # Official YOLO-World model
                self.is_yolo_world = True
                print("Successfully loaded yolov8s-world.pt")
            except Exception as e1:
                print(f"Failed to load yolov8s-world.pt: {e1}")
                try:
                    # Fallback to basic YOLO model (can still be used for object detection)
                    print("Trying fallback to basic YOLOv8...")
                    self.model = YOLO("yolov8s.pt")
                    self.is_yolo_world = False
                    print("Successfully loaded yolov8s.pt (basic YOLO - limited text capability)")
                except Exception as e2:
                    print(f"Failed to load any model: {e2}")
                    raise Exception("Could not load any YOLO model")
            
            # Set custom vocabulary based on the prompt (only for YOLO-World)
            if self.is_yolo_world:
                print(f"Setting custom vocabulary: [{prompt}]")
                self.model.set_classes([prompt])
            else:
                print("Note: Using basic YOLO - will detect general objects, not specific text prompts")
            
            self.is_initialized = True
            print("YOLO-World initialized successfully")
            
        except Exception as e:
            print(f"Failed to initialize YOLO-World: {e}")
            if "No such file or directory" in str(e):
                print("Model download failed - try running once with internet connection")
            self.is_initialized = False
    
    def detect_objects(self, frame):
        # Detect objects in frame using text prompt
        if not self.is_initialized or self.model is None:
            return None
            
        try:
            # Run detection
            results = self.model(frame, verbose=False)
            
            if len(results) == 0 or len(results[0].boxes) == 0:
                return None
                
            # Get the highest confidence detection
            boxes = results[0].boxes
            if len(boxes) == 0:
                return None
                
            # Find box with highest confidence
            best_idx = 0
            best_conf = 0
            for i, box in enumerate(boxes):
                conf = float(box.conf[0])
                if conf > best_conf:
                    best_conf = conf
                    best_idx = i
            
            # Only return detection if confidence is above threshold
            if best_conf < 0.3:  # 30% confidence threshold
                print(f"Low confidence detection ({best_conf:.2f}) for '{self.prompt}' - skipping")
                return None
                
            # Extract bounding box (x1, y1, x2, y2)
            best_box = boxes[best_idx]
            x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
            
            # Convert to integers and validate
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            if x2 <= x1 or y2 <= y1:
                return None
                
            print(f"YOLO-World detected '{self.prompt}' at ({x1}, {y1}, {x2}, {y2}) confidence: {best_conf:.2f}")
            return (x1, y1, x2, y2)
            
        except Exception as e:
            print(f"YOLO-World detection error: {e}")
            return None


class ActionType(Enum):
    # Define all possible user actions
    ENTER_LABEL_MODE = "enter_label_mode"
    DRAW_BOX = "draw_box"
    ACCEPT_PREDICTION = "accept_prediction"
    FIX_PREDICTION = "fix_prediction"
    SKIP_FRAME = "skip_frame"
    MARK_INVISIBLE = "mark_invisible"
    RUN_TEXT_DETECTION = "run_text_detection"
    QUIT = "quit"


class InputHandler:
    # Handle all user input (mouse, keyboard) and translate to high-level actions
    
    def __init__(self, action_callback, has_text_detector=False):
        self.action_callback = action_callback
        self.has_text_detector = has_text_detector
        
        # Mouse drawing state
        self.drawing = False
        self.start_point = None
        self.end_point = None
    
    def handle_mouse_events(self, event, x, y, flags, param):
        # Handle mouse events and translate to actions
        if event == cv2.EVENT_LBUTTONDOWN:
            self.handle_mouse_press(x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            self.handle_mouse_move(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.handle_mouse_release(x, y)
    
    def handle_mouse_press(self, x, y):
        # Handle mouse button press
        self.drawing = True
        self.start_point = (x, y)
        self.end_point = (x, y)
        # Notify that drawing started (for visual feedback)
        self.action_callback("DRAWING_STARTED", {"point": (x, y)})
    
    def handle_mouse_move(self, x, y):
        # Handle mouse movement while drawing
        if not self.drawing:
            return
        
        # Always update the endpoint for accurate final result
        self.end_point = (x, y)
        
        # Throttle drawing updates to ~60 FPS for smooth performance
        current_time = time.time() * 1000  # Convert to milliseconds
        if not hasattr(self, 'last_draw_update_time'):
            self.last_draw_update_time = 0
        
        time_since_last_update = current_time - self.last_draw_update_time
        if time_since_last_update >= 16.67:  # ~60 FPS (16.67ms between updates)
            self.last_draw_update_time = current_time
            
            # Emit drawing update for visual feedback
            self.action_callback("DRAWING_UPDATED", {
                "start": self.start_point,
                "end": self.end_point
            })
    
    def handle_mouse_release(self, x, y):
        # Handle mouse button release
        if not self.drawing:
            return
        
        self.drawing = False
        self.end_point = (x, y)
        
        # Always emit draw action - validation happens in the handler
        if self.start_point and self.end_point:
            self.action_callback(ActionType.DRAW_BOX.value, {
                "start": self.start_point,
                "end": self.end_point
            })
        
        # Reset drawing state
        self.start_point = None
        self.end_point = None
    
    def handle_key_input(self, key):
        # Handle keyboard input and translate to actions
        # L/l key behavior depends on whether text detection is available
        l_action = ActionType.RUN_TEXT_DETECTION if self.has_text_detector else ActionType.ENTER_LABEL_MODE
        
        # Map keys to actions using match/case with ASCII values
        match key:
            case 113 | 81:  # 'q' | 'Q'
                action_type = ActionType.QUIT
            case 108 | 76:  # 'l' | 'L'
                action_type = l_action
            case 97 | 65:   # 'a' | 'A'
                action_type = ActionType.ACCEPT_PREDICTION
            case 102 | 70:  # 'f' | 'F'
                action_type = ActionType.FIX_PREDICTION
            case 115 | 83:  # 's' | 'S'
                action_type = ActionType.SKIP_FRAME
            case 105 | 73:  # 'i' | 'I'
                action_type = ActionType.MARK_INVISIBLE
            case _:
                # Unknown key
                self.action_callback("UNKNOWN_KEY", {"key": key})
                return False
        
        self.action_callback(action_type.value, {})
        return True
    
    def process_input_events(self, window_name):
        # Process input events (keyboard) using non-blocking pollKey
        # Check window closure first
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                self.action_callback(ActionType.QUIT.value, {})
                return False
        except cv2.error:
            # Window was destroyed
            self.action_callback(ActionType.QUIT.value, {})
            return False
        
        # Poll for key press (non-blocking)
        key = cv2.pollKey() & 0xFF
        
        if key != 255:  # Key was pressed
            return self.handle_key_input(key)
        
        # No key pressed, continue processing
        return True


class TrackerManager:
    # Manages OpenCV tracker functionality.
    
    def __init__(self):
        self.tracker = None
        self.has_tracker = False
        self.predicted_box = None
    
    
    def _create_tracker(self):
        # Create a CSRT tracker instance.
        return cv2.TrackerCSRT.create()
    
    def initialize(self, frame, box):
        # Initialize tracker with a bounding box.
        x1, y1, x2, y2 = box
        x, y, w, h = x1, y1, x2 - x1, y2 - y1
        
        # Validate bounding box
        if w <= 0 or h <= 0:
            print(f"Invalid bounding box dimensions: w={w}, h={h}")
            return False
        
        if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
            # Clip the bounding box to frame bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)
            print(f"Clipped bbox to frame bounds: x={x}, y={y}, w={w}, h={h}")
        
        # Create CSRT tracker
        self.tracker = self._create_tracker()
        
        # Initialize tracker
        bbox = (int(x), int(y), int(w), int(h))
        success = self.tracker.init(frame, bbox)
        
        if success is True or success is None: #This is for safety because of differences in opencv versions
            self.has_tracker = True
            print(f"Tracker initialized successfully with bbox: {bbox}")
            return True
        else:
            print("Tracker initialization failed")
            self.tracker = None
            return False
    
    def update(self, frame):
        # Update tracker and get predicted box.
        if not self.has_tracker or self.tracker is None:
            self.predicted_box = None
            return
        
        try:
            success, bbox = self.tracker.update(frame)
            
            if success is True or (success is not False and bbox is not None):
                x, y, w, h = bbox
                if w > 0 and h > 0 and x >= 0 and y >= 0:
                    self.predicted_box = (int(x), int(y), int(w), int(h))
                    print(f"Tracker prediction: box at ({int(x)}, {int(y)}) size {int(w)}x{int(h)}")
                else:
                    self.predicted_box = None
            else:
                print("Tracker lost object")
                self.predicted_box = None
        except Exception as e:
            print(f"Tracker update error: {e}")
            self.predicted_box = None
    
    def reinitialize_with_correction(self, frame, box):
        # Reinitialize tracker with a corrected bounding box.
        x1, y1, x2, y2 = box
        x, y, w, h = x1, y1, x2 - x1, y2 - y1
        
        # Only reinitialize if correction is significant
        if self.predicted_box:
            px, py, pw, ph = self.predicted_box
            pred_center = (px + pw // 2, py + ph // 2)
            corr_center = (x + w // 2, y + h // 2)
            distance = ((pred_center[0] - corr_center[0]) ** 2 + (pred_center[1] - corr_center[1]) ** 2) ** 0.5
            

        
        # Reinitialize tracker
        self.tracker = self._create_tracker()
        if self.tracker and frame is not None:
            success = self.tracker.init(frame, (x, y, w, h))
            if success is True or success is None:
                print("Tracker reinitialized with correction")
            else:
                print("Failed to reinitialize tracker")


class AnnotationManager:
    # Manages annotation data and file operations.
    
    def __init__(self, annotations_file: str):
        self.annotations_file = annotations_file
        self.annotations = []
        self.load_existing_annotations()
    
    def load_existing_annotations(self):
        # Load existing annotations from file.
        if os.path.exists(self.annotations_file):
            try:
                with open(self.annotations_file, 'r') as f:
                    self.annotations = [line.strip() for line in f.readlines() if line.strip()]
                print(f"Loaded existing annotations from {self.annotations_file}")
            except Exception as e:
                print(f"Warning: Could not load annotations file: {e}")
    
    def save_annotations(self):
        # Save annotations to file.
        try:
            with open(self.annotations_file, 'w') as f:
                for annotation in self.annotations:
                    f.write(annotation + '\n')
            print(f"Annotations saved to {self.annotations_file}")
        except Exception as e:
            print(f"Error saving annotations: {e}")
    
    def add_box_annotation(self, box: Tuple[int, int, int, int]) -> str:
        # Convert box to annotation format and add to list.
        x1, y1, x2, y2 = box
        x_center = (x1 + x2) // 2
        y_center = (y1 + y2) // 2
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        annotation = f"V {x_center} {y_center} {width} {height}"
        self.annotations.append(annotation)
        return annotation
    
    def add_skip_annotation(self):
        # Add skip annotation.
        self.annotations.append("S -1 -1 -1 -1")
    
    def add_invisible_annotation(self):
        # Add invisible annotation.
        self.annotations.append("I -1 -1 -1 -1")


class DisplayManager:
    # Manages frame display and drawing operations.
    
    def __init__(self, window_name: str, prompt: str, max_display_width: int = 1200, max_display_height: int = 800):
        self.window_name = window_name
        self.prompt = prompt
        self.max_display_width = max_display_width
        self.max_display_height = max_display_height
        self.scale_factor = 1.0
        self.display_size = None
    
    def calculate_display_size(self, original_width: int, original_height: int):
        # Calculate optimal display size and scaling factor
        if self.display_size is not None:
            return  # Already calculated
        
        # Calculate scale factor to fit within max dimensions
        width_scale = self.max_display_width / original_width
        height_scale = self.max_display_height / original_height
        self.scale_factor = min(width_scale, height_scale, 1.0)  # Don't scale up, only down
        
        # Calculate display dimensions
        display_width = int(original_width * self.scale_factor)
        display_height = int(original_height * self.scale_factor)
        self.display_size = (display_width, display_height)
        
        print(f"Display scaling: {original_width}x{original_height} → {display_width}x{display_height} (scale: {self.scale_factor:.2f})")
    
    def scale_frame(self, frame):
        # Scale frame to display size if needed
        if self.scale_factor != 1.0:
            return cv2.resize(frame, self.display_size, interpolation=cv2.INTER_AREA)
        return frame
    
    def scale_coordinates_to_original(self, x, y):
        # Convert display coordinates back to original video coordinates
        if self.scale_factor != 1.0:
            return int(x / self.scale_factor), int(y / self.scale_factor)
        return x, y
    
    def scale_coordinates_to_display(self, x, y):
        # Convert original coordinates to display coordinates
        if self.scale_factor != 1.0:
            return int(x * self.scale_factor), int(y * self.scale_factor)
        return x, y
    
    def scale_box_to_display(self, box):
        # Scale a bounding box to display coordinates
        if box is None:
            return None
        if self.scale_factor != 1.0:
            x1, y1, x2, y2 = box
            return (int(x1 * self.scale_factor), int(y1 * self.scale_factor), 
                   int(x2 * self.scale_factor), int(y2 * self.scale_factor))
        return box
    
    def create_text_overlay(self, frame, current_frame: int, total_frames: int, 
                           label_mode: bool, fix_mode: bool, has_prediction: bool, has_text_detector: bool = False):
        # Add text overlay with instructions and frame info.
        frame_text = f"Frame: {current_frame + 1}/{total_frames}"
        
        # Build controls list dynamically based on available actions
        controls = []
        
        if label_mode:
            mode_text = " (LABEL MODE - Click and drag from center)"
            controls.extend([
                "Mouse - Click & drag (scales from center)",
                "Q/q - Quit and save"
            ])
        elif fix_mode:
            mode_text = " (FIX MODE - Adjust the predicted box)"
            controls.extend([
                "Mouse - Click & drag to adjust box",
                "Q/q - Quit and save"
            ])
        else:
            mode_text = ""
            
            # Add available shortcuts based on current context
            if has_prediction:
                mode_text = " (PREDICTION AVAILABLE)"
                controls.extend([
                    "A/a - Accept prediction",
                    "F/f - Fix prediction manually"
                ])
                # Show re-run option if text detector is available
                if has_text_detector:
                    controls.append("L/l - Re-run text detection")
            else:
                # Show appropriate L/l action based on whether text detector is available
                if has_text_detector:
                    controls.extend([
                        "L/l - Run text detection",
                        "F/f - Enter manual labeling mode"
                    ])
                else:
                    controls.append("L/l - Enter label mode")
            
            # These are always available when not in interactive modes
            controls.extend([
                "S/s - Skip this frame",
                "I/i - Mark as invisible",
                "Q/q - Quit and save"
            ])
        
        instructions = [
            self.prompt + mode_text,
            "",
            "Controls:"
        ] + controls + ["", frame_text]
        
        # Add semi-transparent background for text
        overlay = frame.copy()
        
        # Calculate font scale based on display size to keep text crisp
        # Use a base font scale that looks good on 1920x1080, then adjust
        base_font_scale = 0.6
        font_scale = max(0.4, base_font_scale * min(self.scale_factor, 1.0))  # Don't make text too small
        thickness = max(1, int(2 * self.scale_factor))
        
        text_height = int(25 * self.scale_factor)
        text_y_start = int(10 * self.scale_factor)
        
        for i, text in enumerate(instructions):
            y_pos = text_y_start + (i * text_height)
            
            # Add background rectangle for better readability
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            padding = int(5 * self.scale_factor)
            cv2.rectangle(overlay, (padding, y_pos - int(20 * self.scale_factor)), 
                         (text_size[0] + padding * 3, y_pos + padding), (0, 0, 0), -1)
            
            # Add text
            color = (0, 255, 255) if text == frame_text else (255, 255, 255)
            cv2.putText(overlay, text, (int(10 * self.scale_factor), y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        
        # Blend overlay with original frame
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        return frame
    
    def draw_boxes(self, frame, predicted_box=None, current_boxes=None, 
                   drawing_box=None, start_point=None, is_drawing=False):
        # Draw all boxes on the frame (coordinates should already be scaled for display).
        if current_boxes is None:
            current_boxes = []
        
        # Draw predicted box (blue) - convert from (x,y,w,h) to display coordinates
        if predicted_box and not is_drawing:
            x, y, w, h = predicted_box
            # Scale the prediction box to display coordinates
            x1, y1 = self.scale_coordinates_to_display(x, y)
            x2, y2 = self.scale_coordinates_to_display(x + w, y + h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, "PREDICTED", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Draw completed boxes (red) - already in display coordinates
        for box in current_boxes:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        
        # Draw current box being drawn (red) - already in display coordinates
        if is_drawing and drawing_box:
            cv2.rectangle(frame, (drawing_box[0], drawing_box[1]), (drawing_box[2], drawing_box[3]), (0, 0, 255), 2)
            if start_point:
                cv2.circle(frame, start_point, 3, (0, 255, 0), -1)
        
        return frame
    
    def render_and_show(self, raw_frame, current_frame: int, total_frames: int,
                       label_mode: bool, fix_mode: bool, has_prediction: bool,
                       predicted_box=None, current_boxes=None, 
                       drawing_box=None, start_point=None, is_drawing=False, has_text_detector=False):
        # Single method to render all overlays and display the frame.
        
        # Initialize display size if not done yet
        if self.display_size is None:
            height, width = raw_frame.shape[:2]
            self.calculate_display_size(width, height)
        
        # Scale frame to display size
        display_frame = self.scale_frame(raw_frame.copy())
        
        # Add all boxes first (on scaled frame)
        display_frame = self.draw_boxes(
            display_frame, predicted_box, current_boxes,
            drawing_box, start_point, is_drawing
        )
        
        # Add text overlay last at native display resolution for crisp text
        display_frame = self.create_text_overlay(
            display_frame, current_frame, total_frames,
            label_mode, fix_mode, has_prediction, has_text_detector
        )
        
        # Display the final frame
        cv2.imshow(self.window_name, display_frame)
    
    def fast_drawing_update(self, raw_frame, predicted_box=None, current_boxes=None, drawing_box=None):
        # Lightweight drawing update for smooth mouse feedback - skips expensive text overlays
        
        # Scale frame to display size
        display_frame = self.scale_frame(raw_frame.copy())
        
        # Draw only essential boxes for drawing feedback
        if predicted_box:
            x, y, w, h = predicted_box
            # Scale the prediction box to display coordinates
            x1, y1 = self.scale_coordinates_to_display(x, y)
            x2, y2 = self.scale_coordinates_to_display(x + w, y + h)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for prediction
        
        # Draw completed boxes (red) - already in display coordinates
        if current_boxes:
            for box in current_boxes:
                cv2.rectangle(display_frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        
        # Draw current drawing box (red for active drawing) - already in display coordinates
        if drawing_box:
            cv2.rectangle(display_frame, (drawing_box[0], drawing_box[1]), (drawing_box[2], drawing_box[3]), (0, 0, 255), 2)
            # Add center point for reference
            center_x, center_y = (drawing_box[0] + drawing_box[2]) // 2, (drawing_box[1] + drawing_box[3]) // 2
            cv2.circle(display_frame, (center_x, center_y), 3, (0, 0, 255), -1)
        
        # Quick display without text overlays
        cv2.imshow(self.window_name, display_frame)


class VideoAnnotator:
    # Core video annotation logic and application coordination.
    
    def __init__(self, video_path: str, prompt: Optional[str] = None, annotations_file: Optional[str] = None):
        self.video_path = video_path
        self.text_prompt = prompt  # Store the original prompt for YOLO-World
        self.display_prompt = "Choose action for this frame:"
        
        # Generate annotations file name if not provided
        if annotations_file is None:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            annotations_file = f"{video_name}.annotations"
        
        # Initialize text detector if prompt provided
        self.text_detector = None
        has_text_detector = False
        if self.text_prompt:
            self.text_detector = YOLOWorldDetector(self.text_prompt)
            has_text_detector = self.text_detector.is_initialized
            if has_text_detector:
                self.display_prompt = f"YOLO-World detection for: '{self.text_prompt}'"
        
        # Initialize managers
        self.tracker_manager = TrackerManager()
        self.annotation_manager = AnnotationManager(annotations_file)
        self.display_manager = DisplayManager("Video Frame Annotator", self.display_prompt)
        self.input_handler = InputHandler(self.handle_action, has_text_detector)
        
        # Video properties
        self.current_frame = 0
        self.total_frames = 0
        self.cap = None
        
        # State variables
        self.label_mode = False
        self.fix_mode = False
        self.should_quit = False  # Flag for auto-quit on last frame
        
        # Current annotation state
        self.current_boxes = []  # Display coordinates
        self.current_drawing_box = None  # Display coordinates  
        self.current_original_boxes = []  # Original video coordinates for saving
    
    def initialize_video(self) -> bool:
        # Initialize video capture and get properties.
        if not os.path.exists(self.video_path):
            print(f"Error: Video file '{self.video_path}' does not exist.")
            return False
        
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print(f"Error: Could not open video file '{self.video_path}'.")
            return False
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video loaded: {os.path.basename(self.video_path)}")
        print(f"Resolution: {width}x{height}")
        print(f"Total frames: {self.total_frames}")
        print(f"FPS: {fps:.2f}")
        print(f"Duration: {self.total_frames/fps:.2f} seconds")
        print()
        
        return True
    
    def get_current_frame(self):
        # Get the current frame from video.
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def next_frame(self):
        # Move to next frame. Returns True if successful, False if at end.
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            return True
        else:
            print("Reached end of video.")
            return False
    
    def calculate_center_based_box(self, center_point, end_point):
        # Calculate bounding box with center-based scaling.
        cx, cy = center_point
        ex, ey = end_point
        
        width = abs(ex - cx) * 2
        height = abs(ey - cy) * 2
        
        x1 = cx - width // 2
        y1 = cy - height // 2
        x2 = cx + width // 2
        y2 = cy + height // 2
        
        return (x1, y1, x2, y2)
    
    def handle_action(self, action_type: str, data: dict):
        # Central action handler - processes all user actions using match/case
        match action_type:
            # Core annotation actions (enum-based)
            case ActionType.ENTER_LABEL_MODE.value:
                self._handle_enter_label_mode()
            case ActionType.RUN_TEXT_DETECTION.value:
                self._handle_run_text_detection()
            case ActionType.ACCEPT_PREDICTION.value:
                self._handle_accept_prediction()
            case ActionType.FIX_PREDICTION.value:
                self._handle_fix_prediction()
            case ActionType.SKIP_FRAME.value:
                self._handle_skip_frame()
            case ActionType.MARK_INVISIBLE.value:
                self._handle_mark_invisible()
            case ActionType.QUIT.value:
                self._handle_quit()
            
            # Drawing actions (require data)
            case ActionType.DRAW_BOX.value:
                self._handle_draw_box(data)
            case "DRAWING_STARTED":
                self._handle_drawing_started(data)
            case "DRAWING_UPDATED":
                self._handle_drawing_updated(data)
            
            # Special cases
            case "UNKNOWN_KEY":
                self._show_help_message()
            case _:
                print(f"Unknown action: {action_type}")
    
    def _handle_enter_label_mode(self):
        # Handle entering label mode
        if not self.label_mode and not self.tracker_manager.predicted_box:
            self.label_mode = True
            self.current_boxes = []
            print(f"Frame {self.current_frame + 1}: ENTERING LABEL MODE")
            self.refresh_display()
        elif self.tracker_manager.predicted_box:
            print("Prediction available. Use A/a to accept or F/f to fix.")
        else:
            print("Already in label mode.")
    
    def _handle_run_text_detection(self):
        # Handle YOLO-World text detection
        if not self.text_detector or not self.text_detector.is_initialized:
            # Fallback to manual labeling if no text detector
            self._handle_enter_label_mode()
            return
            
        if self.label_mode or self.fix_mode:
            print("Exit current mode first")
            return
            
        print(f"Frame {self.current_frame + 1}: Running YOLO-World detection for '{self.text_prompt}'...")
        
        # Get current frame for detection
        frame = self.get_current_frame()
        if frame is None:
            print("Could not get current frame for detection")
            return
            
        # Run YOLO-World detection
        detection_box = self.text_detector.detect_objects(frame)
        
        if detection_box is None:
            print(f"No objects detected for '{self.text_prompt}'. Use F/f for manual labeling.")
            return
            
        # Convert detection to tracker format (x, y, w, h)
        x1, y1, x2, y2 = detection_box
        x, y, w, h = x1, y1, x2 - x1, y2 - y1
        
        # Set as tracker prediction (simulating tracker output)
        self.tracker_manager.predicted_box = (x, y, w, h)
        
        # Initialize tracker with detection for future frames
        if not self.tracker_manager.has_tracker:
            self.tracker_manager.initialize(frame, detection_box)
            
        print(f"Frame {self.current_frame + 1}: YOLO-World detected object")
        print("Use A/a to accept, L/l to re-run detection, or F/f to fix manually")
        self.refresh_display()
    
    def _handle_draw_box(self, data):
        # Handle completed box drawing
        if not self.label_mode and not self.fix_mode:
            return
        
        start_point = data["start"]
        end_point = data["end"]
        
        # Convert display coordinates to original video coordinates
        orig_start = self.display_manager.scale_coordinates_to_original(start_point[0], start_point[1])
        orig_end = self.display_manager.scale_coordinates_to_original(end_point[0], end_point[1])
        
        # Validate box size - prevent tiny/invalid boxes (use original coordinates)
        min_box_size = 5  # Minimum 5 pixels in each dimension in original scale
        distance_x = abs(orig_end[0] - orig_start[0])
        distance_y = abs(orig_end[1] - orig_start[1])
        
        if distance_x < min_box_size or distance_y < min_box_size:
            if self.fix_mode and self.tracker_manager.predicted_box:
                # In fix mode with tiny box: keep the original prediction
                print(f"Box too small ({distance_x}x{distance_y}), keeping original tracker prediction")
                # Don't add new box, just complete with existing prediction
                self._complete_fix_mode_with_original()
                return
            elif self.label_mode:
                # In label mode with tiny box: ignore the draw attempt
                print(f"Box too small ({distance_x}x{distance_y}), minimum size is {min_box_size}x{min_box_size} pixels")
                return
        
        # Box is valid, calculate in original coordinates and convert to display for storage
        orig_box = self.calculate_center_based_box(orig_start, orig_end)
        display_box = self.display_manager.scale_box_to_display(orig_box)
        self.current_boxes.append(display_box)
        
        # Store original coordinates for annotation (convert back to original)
        self.current_original_boxes = [orig_box]  # Store original for saving
        
        if self.fix_mode:
            self._complete_fix_mode()
        elif self.label_mode:
            self._complete_label_mode()
    
    def _handle_drawing_started(self, data):
        # Handle drawing started (for visual feedback)
        if self.label_mode or self.fix_mode:
            # Just refresh display to show drawing state
            self.refresh_display()
    
    def _handle_drawing_updated(self, data):
        # Handle drawing update (for real-time visual feedback)
        if self.label_mode or self.fix_mode:
            start_point = data["start"]
            end_point = data["end"]
            self.current_drawing_box = self.calculate_center_based_box(start_point, end_point)
            
            # Use lightweight drawing update for smooth performance
            raw_frame = self.get_current_frame()
            if raw_frame is not None:
                self.display_manager.fast_drawing_update(
                    raw_frame=raw_frame,
                    predicted_box=self.tracker_manager.predicted_box,
                    current_boxes=self.current_boxes,
                    drawing_box=self.current_drawing_box
                )
    
    def _handle_accept_prediction(self):
        # Handle accepting prediction
        if self.tracker_manager.predicted_box and not self.label_mode and not self.fix_mode:
            x, y, w, h = self.tracker_manager.predicted_box
            box = (x, y, x + w, y + h)
            self.annotation_manager.add_box_annotation(box)
            print(f"Frame {self.current_frame + 1}: ACCEPTED PREDICTION {box}")
            
            self.tracker_manager.predicted_box = None
            
            # Check if we should quit after moving to next frame
            if not self.next_frame():
                print("All frames completed. Quitting...")
                self.should_quit = True
            else:
                self.refresh_display()
        else:
            print("No prediction available to accept.")
    
    def _handle_fix_prediction(self):
        # Handle fixing prediction or manual labeling as fallback
        if not self.label_mode and not self.fix_mode:
            if self.tracker_manager.predicted_box:
                # Standard fix mode: adjust existing prediction
                self.fix_mode = True
                x, y, w, h = self.tracker_manager.predicted_box
                
                # Convert tracker coordinates (x, y, w, h) to display coordinates (x1, y1, x2, y2)
                orig_box = (x, y, x + w, y + h)  # Convert to (x1, y1, x2, y2) format
                display_box = self.display_manager.scale_box_to_display(orig_box)
                self.current_boxes = [display_box]
                
                print(f"Frame {self.current_frame + 1}: ENTERING FIX MODE")
                self.refresh_display()
            elif self.text_detector and self.text_detector.is_initialized:
                # Fallback: no prediction available, enter manual labeling mode
                self.label_mode = True
                self.current_boxes = []
                print(f"Frame {self.current_frame + 1}: NO PREDICTION - ENTERING MANUAL LABEL MODE")
                self.refresh_display()
            else:
                print("No prediction available to fix. Use L/l for manual labeling.")
        else:
            print("Exit current mode first.")
    
    def _handle_skip_frame(self):
        # Handle skipping frame
        if not self.label_mode and not self.fix_mode:
            self.annotation_manager.add_skip_annotation()
            print(f"Frame {self.current_frame + 1}: SKIPPED")
            
            # Check if we should quit after moving to next frame
            if not self.next_frame():
                print("All frames completed. Quitting...")
                self.should_quit = True
            else:
                self.refresh_display()
        else:
            print("Exit current mode first")
    
    def _handle_mark_invisible(self):
        # Handle marking frame as invisible
        if not self.label_mode and not self.fix_mode:
            self.annotation_manager.add_invisible_annotation()
            print(f"Frame {self.current_frame + 1}: MARKED AS INVISIBLE")
            
            # Check if we should quit after moving to next frame
            if not self.next_frame():
                print("All frames completed. Quitting...")
                self.should_quit = True
            else:
                self.refresh_display()
        else:
            print("Exit current mode first")
    
    def _handle_quit(self):
        # Handle quit action
        self.should_quit = True
        print("Quitting...")
    
    def _complete_fix_mode(self):
        # Complete fix mode and process results
        if hasattr(self, 'current_original_boxes') and self.current_original_boxes:
            for orig_box in self.current_original_boxes:
                self.annotation_manager.add_box_annotation(orig_box)
                
                # Update tracker with correction (use original coordinates)
                if self.tracker_manager.has_tracker:
                    frame = self.get_current_frame()
                    if frame is not None:
                        self.tracker_manager.reinitialize_with_correction(frame, orig_box)
            
            print(f"Frame {self.current_frame + 1}: FIXED and LABELED with {len(self.current_original_boxes)} box")
        
        self.fix_mode = False
        self.current_boxes = []
        self.current_drawing_box = None
        self.current_original_boxes = []
        self.tracker_manager.predicted_box = None
        
        # Check if we should quit after moving to next frame
        if not self.next_frame():
            print("All frames completed. Quitting...")
            self.should_quit = True
        else:
            self.refresh_display()
    
    def _complete_fix_mode_with_original(self):
        # Complete fix mode by keeping the original tracker prediction
        if self.tracker_manager.predicted_box:
            x, y, w, h = self.tracker_manager.predicted_box
            box = (x, y, x + w, y + h)
            self.annotation_manager.add_box_annotation(box)
            print(f"Frame {self.current_frame + 1}: KEPT ORIGINAL PREDICTION {box}")
        
        self.fix_mode = False
        self.current_boxes = []
        self.current_drawing_box = None
        self.tracker_manager.predicted_box = None
        
        # Check if we should quit after moving to next frame
        if not self.next_frame():
            print("All frames completed. Quitting...")
            self.should_quit = True
        else:
            self.refresh_display()
    
    def _complete_label_mode(self):
        # Complete label mode and process results
        if hasattr(self, 'current_original_boxes') and self.current_original_boxes:
            for orig_box in self.current_original_boxes:
                self.annotation_manager.add_box_annotation(orig_box)
                
                # Initialize tracker with first labeled box (use original coordinates)
                if not self.tracker_manager.has_tracker:
                    frame = self.get_current_frame()
                    if frame is not None:
                        self.tracker_manager.initialize(frame, orig_box)
            
            print(f"Frame {self.current_frame + 1}: LABELED with {len(self.current_original_boxes)} box")
        
        self.label_mode = False
        self.current_boxes = []
        self.current_drawing_box = None
        self.current_original_boxes = []
        
        # Check if we should quit after moving to next frame
        if not self.next_frame():
            print("All frames completed. Quitting...")
            self.should_quit = True
        else:
            self.refresh_display()
    
    def refresh_display(self):
        # Refresh the display with current frame and overlays.
        raw_frame = self.get_current_frame()
        if raw_frame is None:
            return
        
        # Update tracker if available (but not during fix mode to avoid lag)
        if self.tracker_manager.has_tracker and not self.fix_mode:
            self.tracker_manager.update(raw_frame)
        elif not self.tracker_manager.has_tracker:
            self.tracker_manager.predicted_box = None
        
        # Render and display everything in one call
        self.display_manager.render_and_show(
            raw_frame=raw_frame,
            current_frame=self.current_frame,
            total_frames=self.total_frames,
            label_mode=self.label_mode,
            fix_mode=self.fix_mode,
            has_prediction=self.tracker_manager.predicted_box is not None,
            predicted_box=self.tracker_manager.predicted_box,
            current_boxes=self.current_boxes,
            drawing_box=self.current_drawing_box,
            start_point=None,  # TODO:Not needed delete relevent code
            is_drawing=self.current_drawing_box is not None,
            has_text_detector=self.text_detector is not None and self.text_detector.is_initialized
        )
    
    def process_frame(self) -> bool:
        # Process current frame and handle user input. Returns False to quit.
        
        # Check if we should quit
        if self.should_quit:
            return False
        
        frame = self.get_current_frame()
        if frame is None:
            print("End of video reached.")
            return False
        
        # Reset boxes for new frame if not in interactive mode
        if not self.label_mode:
            self.current_boxes = []
            self.current_drawing_box = None
        
        # Process input events 
        return self.input_handler.process_input_events(self.display_manager.window_name)
    
    def _show_help_message(self):
        # Show help message for invalid keys based on current mode.
        if self.label_mode:
            print("In label mode: Draw boxes with mouse")
        elif self.fix_mode:
            print("In fix mode: Draw boxes with mouse")
        elif self.tracker_manager.predicted_box:
            print("Prediction available: Use A/a (Accept), F/f (Fix), S/s (Skip), I/i (Invisible), or Q/q (Quit)")
        else:
            print("Use L/l (Label), S/s (Skip), I/i (Invisible), or Q/q (Quit)")
    
    def run(self):
        # Main annotation loop.
        if not self.initialize_video():
            return False
        
        # Create window and set mouse callback
        cv2.namedWindow(self.display_manager.window_name, cv2.WINDOW_NORMAL)  # Resizable window
        cv2.setMouseCallback(self.display_manager.window_name, self.input_handler.handle_mouse_events)
        
        # Print startup information
        print("Starting annotation...")
        print(f"OpenCV version: {cv2.__version__}")
        print("Using CSRT tracker for object tracking.")
        
        # Show YOLO-World status
        if self.text_detector and self.text_detector.is_initialized:
            print(f"YOLO-World Text Detection: ENABLED for '{self.text_prompt}'")
            print("  Auto-detection will run on first frame, then use L/l to re-run, A/a to accept, F/f to fix")
        else:
            print("YOLO-World Text Detection: DISABLED")
            print("  Use L/l for manual labeling mode")
        
        print()
        print("Use keyboard shortcuts to annotate frames:")
        if self.text_detector and self.text_detector.is_initialized:
            print("L/l - Re-run YOLO-World text detection (auto-runs on first frame)")
        else:
            print("L/l - Enter manual label mode (click & drag from center)")
        print("A/a - Accept tracker prediction (when available)")
        print("F/f - Fix tracker prediction (when available)")
        print("S/s - Skip, I/i - Invisible, Q/q - Quit")
        print(f"Annotations will be saved to: {self.annotation_manager.annotations_file}")
        print("-" * 50)
        
        try:
            # Display initial frame
            self.refresh_display()
            
            # Auto-run text detection on first frame if prompt provided
            if self.text_detector and self.text_detector.is_initialized:
                print(f"Auto-running YOLO-World detection on first frame for '{self.text_prompt}'...")
                self._handle_run_text_detection()
            
            # Main annotation loop with pollKey
            while True:
                # Process input and check for quit
                if not self.process_frame():
                    break
                    
                # Small delay to prevent excessive CPU usage (60 FPS equivalent)
                time.sleep(1/60)  # ~16.67ms delay
            
            # Save annotations before exiting
            self.annotation_manager.save_annotations()
            
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            self.annotation_manager.save_annotations()
        
        finally:
            # Cleanup
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
        
        return True


def main():
    # Main entry point for the CLI tool.
    parser = argparse.ArgumentParser(
        description="Video Frame Annotation Tool",
        epilog="Example: python LabelTool.py video.mp4 annotations.annotations --prompt 'robot'"
    )
    
    parser.add_argument("video_path", help="Path to the video file to annotate")
    parser.add_argument("annotations_path", nargs="?", help="Optional path to annotations file (.annotations extension required)")
    parser.add_argument("--prompt", type=str, help="Text prompt for YOLO-World object detection (e.g., 'robot', 'person with red shirt')")
    
    args = parser.parse_args()
    
    # Validate annotations file extension if provided
    annotations_file = args.annotations_path
    if annotations_file and not annotations_file.endswith('.annotations'):
        print(f"Error: Annotations file must have .annotations extension, got: {annotations_file}")
        sys.exit(1)
    
    # Create and run annotator
    annotator = VideoAnnotator(
        video_path=args.video_path,
        prompt=args.prompt,
        annotations_file=annotations_file
    )
    
    success = annotator.run()
    
    if not success:
        sys.exit(1)
    
    print("Annotation completed successfully!")


if __name__ == "__main__":
    main()


# Fixed: Annotations file CLI argument and extension validation
# Fixed: Text overlay resolution scaling for crisp display
#TODO: Create Readme and install directions (make sure it works on linux)
#TODO: Understand and explain the logic of the object tracking
#TODO: explain why I used the tools and packages that I did (Text-based detection, Tracking function, versions of each of those)
#TODO: Find edge cases and test them
#TODO: Try and improve performance or at least understand why its so bad