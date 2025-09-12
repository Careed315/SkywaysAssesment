#!/usr/bin/env python3
"""
Video Frame Annotation CLI Tool

Usage: python main.py videoPath.mp4 [--prompt "Custom prompt"] [--annotations pathToAnnotationsFile.annotations]

Features:
- Display video frames using OpenCV
- Three actions per frame: Label (L/l), Skip (S/s), Invisible (I/i)
- Quit with Q/q
- Frame navigation with keyboard shortcuts
- Optional annotations file support
"""

import argparse
import cv2
import os
import sys
from typing import Dict, List, Optional, Tuple


class TrackerManager:
    """Manages OpenCV tracker functionality."""
    
    def __init__(self):
        self.tracker = None
        self.has_tracker = False
        self.predicted_box = None
    
    
    def _create_tracker(self):
        """Create a CSRT tracker instance."""
        return cv2.TrackerCSRT.create()
    
    def initialize(self, frame, box):
        """Initialize tracker with a bounding box."""
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
        """Update tracker and get predicted box."""
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
        """Reinitialize tracker with a corrected bounding box."""
        x1, y1, x2, y2 = box
        x, y, w, h = x1, y1, x2 - x1, y2 - y1
        
        # Only reinitialize if correction is significant
        if self.predicted_box:
            px, py, pw, ph = self.predicted_box
            pred_center = (px + pw // 2, py + ph // 2)
            corr_center = (x + w // 2, y + h // 2)
            distance = ((pred_center[0] - corr_center[0]) ** 2 + (pred_center[1] - corr_center[1]) ** 2) ** 0.5
            
            if distance < 30:
                print("Correction is minor, keeping current tracker state")
                return
        
        # Reinitialize tracker
        self.tracker = self._create_tracker()
        if self.tracker and frame is not None:
            success = self.tracker.init(frame, (x, y, w, h))
            if success is True or success is None:
                print("Tracker reinitialized with correction")
            else:
                print("Failed to reinitialize tracker")


class AnnotationManager:
    """Manages annotation data and file operations."""
    
    def __init__(self, annotations_file: str):
        self.annotations_file = annotations_file
        self.annotations = []
        self.load_existing_annotations()
    
    def load_existing_annotations(self):
        """Load existing annotations from file."""
        if os.path.exists(self.annotations_file):
            try:
                with open(self.annotations_file, 'r') as f:
                    self.annotations = [line.strip() for line in f.readlines() if line.strip()]
                print(f"Loaded existing annotations from {self.annotations_file}")
            except Exception as e:
                print(f"Warning: Could not load annotations file: {e}")
    
    def save_annotations(self):
        """Save annotations to file."""
        try:
            with open(self.annotations_file, 'w') as f:
                for annotation in self.annotations:
                    f.write(annotation + '\n')
            print(f"Annotations saved to {self.annotations_file}")
        except Exception as e:
            print(f"Error saving annotations: {e}")
    
    def add_box_annotation(self, box: Tuple[int, int, int, int]) -> str:
        """Convert box to annotation format and add to list."""
        x1, y1, x2, y2 = box
        x_center = (x1 + x2) // 2
        y_center = (y1 + y2) // 2
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        annotation = f"V {x_center} {y_center} {width} {height}"
        self.annotations.append(annotation)
        return annotation
    
    def add_skip_annotation(self):
        """Add skip annotation."""
        self.annotations.append("S -1 -1 -1 -1")
    
    def add_invisible_annotation(self):
        """Add invisible annotation."""
        self.annotations.append("I -1 -1 -1 -1")


class DisplayManager:
    """Manages frame display and drawing operations."""
    
    def __init__(self, window_name: str, prompt: str):
        self.window_name = window_name
        self.prompt = prompt
    
    def create_text_overlay(self, frame, current_frame: int, total_frames: int, 
                           label_mode: bool, fix_mode: bool, has_prediction: bool):
        """Add text overlay with instructions and frame info."""
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
            else:
                # Only show label mode if no prediction is available
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
        text_height = 25
        text_y_start = 10
        
        for i, text in enumerate(instructions):
            y_pos = text_y_start + (i * text_height)
            
            # Add background rectangle for better readability
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(overlay, (5, y_pos - 20), (text_size[0] + 15, y_pos + 5), (0, 0, 0), -1)
            
            # Add text
            color = (0, 255, 255) if text == frame_text else (255, 255, 255)
            cv2.putText(overlay, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Blend overlay with original frame
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        return frame
    
    def draw_boxes(self, frame, predicted_box=None, current_boxes=None, 
                   drawing_box=None, start_point=None, is_drawing=False):
        """Draw all boxes on the frame."""
        if current_boxes is None:
            current_boxes = []
        
        # Draw predicted box (blue)
        if predicted_box and not is_drawing:
            x, y, w, h = predicted_box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, "PREDICTED", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Draw completed boxes (red)
        for box in current_boxes:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        
        # Draw current box being drawn (red)
        if is_drawing and drawing_box:
            cv2.rectangle(frame, (drawing_box[0], drawing_box[1]), (drawing_box[2], drawing_box[3]), (0, 0, 255), 2)
            if start_point:
                cv2.circle(frame, start_point, 3, (0, 255, 0), -1)
        
        return frame
    
    def render_and_show(self, raw_frame, current_frame: int, total_frames: int,
                       label_mode: bool, fix_mode: bool, has_prediction: bool,
                       predicted_box=None, current_boxes=None, 
                       drawing_box=None, start_point=None, is_drawing=False):
        """Single method to render all overlays and display the frame."""
        # Start with a copy of the raw frame
        display_frame = raw_frame.copy()
        
        # Add text overlay
        display_frame = self.create_text_overlay(
            display_frame, current_frame, total_frames,
            label_mode, fix_mode, has_prediction
        )
        
        # Add all boxes
        display_frame = self.draw_boxes(
            display_frame, predicted_box, current_boxes,
            drawing_box, start_point, is_drawing
        )
        
        # Display the final frame
        cv2.imshow(self.window_name, display_frame)


class VideoAnnotator:
    """Main class for video frame annotation."""
    
    def __init__(self, video_path: str, prompt: Optional[str] = None, annotations_file: Optional[str] = None):
        self.video_path = video_path
        self.prompt = prompt or "Choose action for this frame:"
        
        # Generate annotations file name if not provided
        if annotations_file is None:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            annotations_file = f"{video_name}.annotations"
        
        # Initialize managers
        self.tracker_manager = TrackerManager()
        self.annotation_manager = AnnotationManager(annotations_file)
        self.display_manager = DisplayManager("Video Frame Annotator", self.prompt)
        
        # Video properties
        self.current_frame = 0
        self.total_frames = 0
        self.cap = None
        
        # State variables
        self.label_mode = False
        self.fix_mode = False
        self.should_quit = False  # Flag for auto-quit on last frame
        
        # Mouse drawing state
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.current_boxes = []
    
    def initialize_video(self) -> bool:
        """Initialize video capture and get properties."""
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
        """Get the current frame from video."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def next_frame(self):
        """Move to next frame. Returns True if successful, False if at end."""
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            return True
        else:
            print("Reached end of video.")
            return False
    
    def calculate_center_based_box(self, center_point, end_point):
        """Calculate bounding box with center-based scaling."""
        cx, cy = center_point
        ex, ey = end_point
        
        width = abs(ex - cx) * 2
        height = abs(ey - cy) * 2
        
        x1 = cx - width // 2
        y1 = cy - height // 2
        x2 = cx + width // 2
        y2 = cy + height // 2
        
        return (x1, y1, x2, y2)
    
    def refresh_display(self):
        """Refresh the display with current frame and overlays."""
        raw_frame = self.get_current_frame()
        if raw_frame is None:
            return
        
        # Update tracker if available (but not during fix mode to avoid lag)
        if self.tracker_manager.has_tracker and not self.fix_mode:
            self.tracker_manager.update(raw_frame)
        elif not self.tracker_manager.has_tracker:
            self.tracker_manager.predicted_box = None
        
        # Calculate drawing box if currently drawing
        drawing_box = None
        if self.drawing and self.start_point and self.end_point:
            drawing_box = self.calculate_center_based_box(self.start_point, self.end_point)
        
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
            drawing_box=drawing_box,
            start_point=self.start_point,
            is_drawing=self.drawing
        )
    
    def handle_mouse_press(self, x, y):
        """Handle mouse button press."""
        if not self.label_mode and not self.fix_mode:
            return
        
        self.drawing = True
        self.start_point = (x, y)
        self.end_point = (x, y)
    
    def handle_mouse_move(self, x, y):
        """Handle mouse movement while drawing."""
        if not self.drawing:
            return
        
        self.end_point = (x, y)
        # Simply refresh the display - the drawing box will be calculated automatically
        self.refresh_display()
    
    def handle_mouse_release(self, x, y):
        """Handle mouse button release."""
        if not self.drawing:
            return
        
        self.drawing = False
        self.end_point = (x, y)
        
        # Calculate and add the completed box
        if self.start_point and self.end_point:
            box = self.calculate_center_based_box(self.start_point, self.end_point)
            self.current_boxes.append(box)
        
        if self.fix_mode:
            result = self._handle_fix_mode_completion()
            if result is False:
                self.should_quit = True
        elif self.label_mode:
            result = self._handle_label_mode_completion()
            if result is False:
                self.should_quit = True
    
    def _handle_fix_mode_completion(self):
        """Handle completion of fix mode drawing. Returns False if should quit."""
        if self.current_boxes:
            for box in self.current_boxes:
                self.annotation_manager.add_box_annotation(box)
                
                # Update tracker with correction
                if self.tracker_manager.has_tracker:
                    frame = self.get_current_frame()
                    if frame is not None:
                        self.tracker_manager.reinitialize_with_correction(frame, box)
            
            print(f"Frame {self.current_frame + 1}: FIXED and LABELED with {len(self.current_boxes)} box(es)")
        
        self.fix_mode = False
        self.current_boxes = []
        self.tracker_manager.predicted_box = None
        
        # Check if we should quit after moving to next frame
        if not self.next_frame():
            print("All frames completed. Quitting...")
            return False
        
        self.refresh_display()
        return True
    
    def _handle_label_mode_completion(self):
        """Handle completion of label mode drawing. Returns False if should quit."""
        if self.current_boxes:
            for box in self.current_boxes:
                self.annotation_manager.add_box_annotation(box)
                
                # Initialize tracker with first labeled box
                if not self.tracker_manager.has_tracker:
                    frame = self.get_current_frame()
                    if frame is not None:
                        self.tracker_manager.initialize(frame, box)
            
            print(f"Frame {self.current_frame + 1}: LABELED with {len(self.current_boxes)} box(es)")
        
        self.label_mode = False
        self.current_boxes = []
        
        # Check if we should quit after moving to next frame
        if not self.next_frame():
            print("All frames completed. Quitting...")
            return False
        
        self.refresh_display()
        return True
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.handle_mouse_press(x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            self.handle_mouse_move(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.handle_mouse_release(x, y)
    
    def handle_key_quit(self) -> bool:
        """Handle quit key press. Returns False to quit."""
        print("Quitting...")
        return False
    
    def handle_key_label(self):
        """Handle label mode key press."""
        if not self.label_mode and not self.tracker_manager.predicted_box:
            self.label_mode = True
            self.current_boxes = []
            print(f"Frame {self.current_frame + 1}: ENTERING LABEL MODE")
            self.refresh_display()
        elif self.tracker_manager.predicted_box:
            print("Prediction available. Use A/a to accept or F/f to fix.")
        else:
            print("Already in label mode.")
    
    def handle_key_accept(self):
        """Handle accept prediction key press. Returns False if should quit."""
        if self.tracker_manager.predicted_box and not self.label_mode and not self.fix_mode:
            x, y, w, h = self.tracker_manager.predicted_box
            box = (x, y, x + w, y + h)
            self.annotation_manager.add_box_annotation(box)
            print(f"Frame {self.current_frame + 1}: ACCEPTED PREDICTION {box}")
            
            self.tracker_manager.predicted_box = None
            
            # Check if we should quit after moving to next frame
            if not self.next_frame():
                print("All frames completed. Quitting...")
                return False
            
            self.refresh_display()
        else:
            print("No prediction available to accept.")
        
        return True
    
    def handle_key_fix(self):
        """Handle fix prediction key press."""
        if self.tracker_manager.predicted_box and not self.label_mode and not self.fix_mode:
            self.fix_mode = True
            x, y, w, h = self.tracker_manager.predicted_box
            center_x, center_y = x + w // 2, y + h // 2
            self.start_point = (center_x, center_y)
            self.current_boxes = [(x, y, x + w, y + h)]
            print(f"Frame {self.current_frame + 1}: ENTERING FIX MODE")
            self.refresh_display()
        else:
            print("No prediction available to fix.")
    
    def handle_key_skip(self):
        """Handle skip frame key press. Returns False if should quit."""
        if not self.label_mode and not self.fix_mode:
            self.annotation_manager.add_skip_annotation()
            print(f"Frame {self.current_frame + 1}: SKIPPED")
            
            # Check if we should quit after moving to next frame
            if not self.next_frame():
                print("All frames completed. Quitting...")
                return False
            
            self.refresh_display()
        else:
            print("Exit current mode first")
        
        return True
    
    def handle_key_invisible(self):
        """Handle invisible key press. Returns False if should quit."""
        if not self.label_mode and not self.fix_mode:
            self.annotation_manager.add_invisible_annotation()
            print(f"Frame {self.current_frame + 1}: MARKED AS INVISIBLE")
            
            # Check if we should quit after moving to next frame
            if not self.next_frame():
                print("All frames completed. Quitting...")
                return False
            
            self.refresh_display()
        else:
            print("Exit current mode first")
        
        return True
    
    def process_frame(self) -> bool: # Process current frame and handle user input. Returns False to quit.
        
        # Check if we should quit due to mouse operations on last frame
        if self.should_quit:
            return False
        
        frame = self.get_current_frame()
        if frame is None:
            print("End of video reached.")
            return False
        
        # Reset boxes for new frame if not in interactive mode
        if not self.label_mode:
            self.current_boxes = []
        
        # Wait for key press (blocking - much simpler and more efficient!)
        while True:
            key = cv2.waitKey(50) & 0xFF  # 50ms = 20 checks/second
        
            # Check window closure
            try:
                if cv2.getWindowProperty(self.display_manager.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    return False
            except cv2.error:
                # Window was destroyed
                return False
            
            if key != 255:  # Key pressed
                break
        # Handle keyboard input with clean dictionary dispatch
        key_handlers = {
            ord('q'): self.handle_key_quit, ord('Q'): self.handle_key_quit,
            ord('l'): self.handle_key_label, ord('L'): self.handle_key_label,
            ord('a'): self.handle_key_accept, ord('A'): self.handle_key_accept,
            ord('f'): self.handle_key_fix, ord('F'): self.handle_key_fix,
            ord('s'): self.handle_key_skip, ord('S'): self.handle_key_skip,
            ord('i'): self.handle_key_invisible, ord('I'): self.handle_key_invisible,
        }
        
        if key in key_handlers:
            result = key_handlers[key]()
            # Quit handler returns False, others return None
            return False if result is False else True
        else:
            # Show help message for invalid keys
            self._show_help_message()
            return True
    
    def _show_help_message(self):
        """Show help message for invalid keys based on current mode."""
        if self.label_mode:
            print("In label mode: Draw boxes with mouse")
        elif self.fix_mode:
            print("In fix mode: Draw boxes with mouse")
        elif self.tracker_manager.predicted_box:
            print("Prediction available: Use A/a (Accept), F/f (Fix), S/s (Skip), I/i (Invisible), or Q/q (Quit)")
        else:
            print("Use L/l (Label), S/s (Skip), I/i (Invisible), or Q/q (Quit)")
    
    def run(self):
        """Main annotation loop."""
        if not self.initialize_video():
            return False
        
        # Create window and set mouse callback
        cv2.namedWindow(self.display_manager.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.display_manager.window_name, self.mouse_callback)
        
        # Print startup information
        print("Starting annotation...")
        print(f"OpenCV version: {cv2.__version__}")
        print("Using CSRT tracker for object tracking.")
        
        print()
        print("Use keyboard shortcuts to annotate frames:")
        print("L/l - Enter label mode (click & drag from center)")
        print("A/a - Accept tracker prediction (when available)")
        print("F/f - Fix tracker prediction (when available)")
        print("S/s - Skip, I/i - Invisible, Q/q - Quit")
        print(f"Annotations will be saved to: {self.annotation_manager.annotations_file}")
        print("-" * 50)
        
        try:
            # Display initial frame
            self.refresh_display()
            
            # Main annotation loop
            while True:
                if not self.process_frame():
                    break
            
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
    """Main entry point for the CLI tool."""
    parser = argparse.ArgumentParser(
        description="Video Frame Annotation Tool",
        epilog="Example: python main.py video.mp4 --prompt 'Label objects:' --annotations output.json"
    )
    
    parser.add_argument("video_path", help="Path to the video file to annotate")
    parser.add_argument("--prompt", type=str, help="Custom prompt text to display")
    parser.add_argument("--annotations", type=str, help="Path to annotations file")
    
    args = parser.parse_args()
    
    # Create and run annotator
    annotator = VideoAnnotator(
        video_path=args.video_path,
        prompt=args.prompt,
        annotations_file=args.annotations
    )
    
    success = annotator.run()
    
    if not success:
        sys.exit(1)
    
    print("Annotation completed successfully!")


if __name__ == "__main__":
    main()