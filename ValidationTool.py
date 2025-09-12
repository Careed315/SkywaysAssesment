#!/usr/bin/env python3
# Video Annotation Visualization Tool
#
# Usage: python main.py videoPath.mp4 [annotationsPath.annotations]
#
# Features:
# - Display video frames with annotation overlays
# - Color-coded bounding boxes for visible objects (green)
# - Status labels for skipped and invisible frames
# - Frame navigation with keyboard shortcuts

import argparse
import cv2
import os
import sys
from typing import Dict, List, Optional, Tuple


class AnnotationParser:
    # Parses and manages annotation data from files.
    
    def __init__(self, annotations_file: str):
        self.annotations_file = annotations_file
        self.annotations = []
        self.load_annotations()
    
    def load_annotations(self):
        # Load annotations from file.
        if not os.path.exists(self.annotations_file):
            print(f"Warning: Annotations file '{self.annotations_file}' does not exist.")
            return
        
        try:
            with open(self.annotations_file, 'r') as f:
                self.annotations = [line.strip() for line in f.readlines() if line.strip()]
            print(f"Loaded {len(self.annotations)} annotations from {self.annotations_file}")
        except Exception as e:
            print(f"Error loading annotations file: {e}")
    
    def get_frame_annotation(self, frame_index: int) -> Optional[Dict]:
        # Get annotation for a specific frame index.
        if frame_index >= len(self.annotations):
            return None
        
        annotation_line = self.annotations[frame_index]
        parts = annotation_line.split()
        
        if len(parts) != 5:
            print(f"Warning: Invalid annotation format at frame {frame_index}: {annotation_line}")
            return None
        
        status = parts[0]
        x_center = int(parts[1])
        y_center = int(parts[2])
        width = int(parts[3])
        height = int(parts[4])
        
        return {
            'status': status,
            'x_center': x_center,
            'y_center': y_center,
            'width': width,
            'height': height
        }


class DisplayManager:
    # Manages frame display and drawing operations.
    
    def __init__(self, window_name: str):
        self.window_name = window_name
    
    def create_text_overlay(self, frame, current_frame: int, total_frames: int, frame_status: str = None):
        # Add text overlay with frame info and navigation controls.
        frame_text = f"Frame: {current_frame + 1}/{total_frames}"
        
        controls = [
            "Video Annotation Viewer",
            "",
            "Navigation:",
            "n - Next frame",
            "N - Skip 10 frames forward",
            "p - Previous frame", 
            "P - Skip 10 frames back",
            "Q/q - Quit",
            "",
            frame_text
        ]
        
        # Add status if frame is skipped or invisible
        if frame_status in ['S', 'I']:
            status_text = "SKIPPED" if frame_status == 'S' else "INVISIBLE"
            controls.append(f"Status: {status_text}")
        
        # Add semi-transparent background for text
        overlay = frame.copy()
        text_height = 25
        text_y_start = 10
        
        for i, text in enumerate(controls):
            y_pos = text_y_start + (i * text_height)
            
            # Add background rectangle for better readability
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(overlay, (5, y_pos - 20), (text_size[0] + 15, y_pos + 5), (0, 0, 0), -1)
            
            # Add text
            color = (0, 255, 255) if "Frame:" in text else (255, 255, 255)
            if "Status:" in text:
                color = (0, 0, 255) if "SKIPPED" in text else (0, 165, 255)
            cv2.putText(overlay, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Blend overlay with original frame
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        return frame
    
    def draw_visible_box(self, frame, annotation: Dict):
        # Draw green bounding box for visible objects.
        if annotation['status'] != 'V':
            return frame
            
        x_center = annotation['x_center']
        y_center = annotation['y_center']
        width = annotation['width']
        height = annotation['height']
        
        # Convert center coordinates to corner coordinates
        x1 = x_center - width // 2
        y1 = y_center - height // 2
        x2 = x_center + width // 2
        y2 = y_center + height // 2
        
        # Draw green bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "VISIBLE", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame
    
    def render_and_show(self, raw_frame, current_frame: int, total_frames: int, annotation: Optional[Dict] = None):
        # Render frame with overlays and display.
        display_frame = raw_frame.copy()
        
        frame_status = annotation['status'] if annotation else None
        
        # Add text overlay
        display_frame = self.create_text_overlay(display_frame, current_frame, total_frames, frame_status)
        
        # Draw bounding box if annotation is visible
        if annotation and annotation['status'] == 'V':
            display_frame = self.draw_visible_box(display_frame, annotation)
        
        # Display the final frame
        cv2.imshow(self.window_name, display_frame)


class VideoAnnotationViewer:
    # Main class for viewing video annotations.
    
    def __init__(self, video_path: str, annotations_file: Optional[str] = None):
        self.video_path = video_path
        
        # Generate annotations file name if not provided
        if annotations_file is None:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            annotations_file = f"{video_name}.annotations"
        
        # Initialize managers
        self.annotation_parser = AnnotationParser(annotations_file)
        self.display_manager = DisplayManager("Video Annotation Viewer")
        
        # Video properties
        self.current_frame = 0
        self.total_frames = 0
        self.cap = None
    
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
    
    def next_frame(self, skip_count: int = 1):
        # Move to next frame(s). Returns True if successful, False if at end.
        new_frame = self.current_frame + skip_count
        if new_frame < self.total_frames:
            self.current_frame = new_frame
            return True
        else:
            print("Reached end of video.")
            return False
    
    def previous_frame(self, skip_count: int = 1):
        # Move to previous frame(s). Returns True if successful, False if at beginning.
        new_frame = self.current_frame - skip_count
        if new_frame >= 0:
            self.current_frame = new_frame
            return True
        else:
            print("At beginning of video.")
            return False
    
    def refresh_display(self):
        # Refresh the display with current frame and annotations.
        raw_frame = self.get_current_frame()
        if raw_frame is None:
            return
        
        # Get annotation for current frame
        annotation = self.annotation_parser.get_frame_annotation(self.current_frame)
        
        # Render and display
        self.display_manager.render_and_show(
            raw_frame=raw_frame,
            current_frame=self.current_frame,
            total_frames=self.total_frames,
            annotation=annotation
        )
    
    def handle_key_quit(self) -> bool:
        # Handle quit key press. Returns False to quit.
        print("Quitting...")
        return False
    
    def handle_key_next(self) -> bool:
        # Handle next frame key press.
        if self.next_frame():
            self.refresh_display()
        return True
    
    def handle_key_next_skip(self) -> bool:
        # Handle skip 10 frames forward key press.
        if self.next_frame(10):
            self.refresh_display()
        return True
    
    def handle_key_previous(self) -> bool:
        # Handle previous frame key press.
        if self.previous_frame():
            self.refresh_display()
        return True
    
    def handle_key_previous_skip(self) -> bool:
        # Handle skip 10 frames back key press.
        if self.previous_frame(10):
            self.refresh_display()
        return True
    
    def process_frame(self) -> bool:
        # Process current frame and handle user input. Returns False to quit.
        frame = self.get_current_frame()
        if frame is None:
            print("End of video reached.")
            return False
        
        # Wait for key press
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
        
        # Handle keyboard input
        key_handlers = {
            ord('q'): self.handle_key_quit, ord('Q'): self.handle_key_quit,
            ord('n'): self.handle_key_next,
            ord('N'): self.handle_key_next_skip,
            ord('p'): self.handle_key_previous,
            ord('P'): self.handle_key_previous_skip,
        }
        
        if key in key_handlers:
            result = key_handlers[key]()
            return False if result is False else True
        else:
            # Show help message for invalid keys
            print("Navigation: n (next), N (skip 10 forward), p (previous), P (skip 10 back), Q/q (quit)")
            return True
    
    def run(self):
        # Main viewing loop.
        if not self.initialize_video():
            return False
        
        # Create window
        cv2.namedWindow(self.display_manager.window_name, cv2.WINDOW_AUTOSIZE)
        
        # Print startup information
        print("Starting annotation viewer...")
        print(f"OpenCV version: {cv2.__version__}")
        print()
        print("Navigation controls:")
        print("n - Next frame")
        print("N - Skip 10 frames forward")
        print("p - Previous frame")
        print("P - Skip 10 frames back")
        print("Q/q - Quit")
        print("-" * 50)
        
        try:
            # Display initial frame
            self.refresh_display()
            
            # Main viewing loop
            while True:
                if not self.process_frame():
                    break
            
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        
        finally:
            # Cleanup
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
        
        return True


def main():
    # Main entry point for the CLI tool.
    parser = argparse.ArgumentParser(
        description="Video Annotation Visualization Tool",
        epilog="Example: python main.py video.mp4 [annotations.annotations]"
    )
    
    parser.add_argument("video_path", help="Path to the video file to view")
    parser.add_argument("annotations_file", nargs='?', help="Path to annotations file (optional)")
    
    args = parser.parse_args()
    
    # Create and run viewer
    viewer = VideoAnnotationViewer(
        video_path=args.video_path,
        annotations_file=args.annotations_file
    )
    
    success = viewer.run()
    
    if not success:
        sys.exit(1)
    
    print("Viewing completed!")


if __name__ == "__main__":
    main()