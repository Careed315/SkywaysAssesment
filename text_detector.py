#!/usr/bin/env python3
"""
Text-based Object Detection Module

This module provides state-of-the-art text-based object detection using GroundingDINO,
a powerful open-source model that can detect objects based on natural language prompts.

Features:
- Natural language object detection (e.g., "person with red shirt", "blue car")
- High accuracy bounding box predictions
- GPU acceleration when available
- Fallback to CPU processing
- Integration with tracking systems

Dependencies:
- torch
- torchvision
- transformers
- groundingdino
- supervision
- pillow
- numpy

Installation:
pip install torch torchvision transformers pillow numpy supervision
pip install groundingdino-py

Usage:
    detector = TextDetector("person with red shirt")
    boxes = detector.detect_objects(frame)
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Optional imports for text-based detection
try:
    import torch
    from PIL import Image
    import supervision as sv
    from groundingdino.util.inference import Model
    GROUNDING_DINO_AVAILABLE = True
    print("GroundingDINO dependencies available.")
except ImportError as e:
    GROUNDING_DINO_AVAILABLE = False
    print(f"GroundingDINO not available: {e}")
    print("Install with: pip install torch torchvision transformers groundingdino-py supervision")


class TextDetector:
    """
    Text-based object detector using GroundingDINO.
    
    This class provides natural language object detection capabilities,
    allowing users to detect objects using descriptive text prompts.
    """
    
    def __init__(self, text_prompt: str, confidence_threshold: float = 0.35):
        """
        Initialize the text detector.
        
        Args:
            text_prompt: Natural language description of objects to detect
            confidence_threshold: Minimum confidence for detections (0.0-1.0)
        """
        self.text_prompt = text_prompt.lower().strip()
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.device = None
        self.is_available = GROUNDING_DINO_AVAILABLE
        
        if self.is_available:
            self._initialize_model()
        else:
            print("Text detection not available. Manual labeling only.")
    
    def _initialize_model(self):
        """Initialize the GroundingDINO model with proper error handling."""
        try:
            print("Initializing GroundingDINO model...")
            
            # Set device (prefer CUDA if available)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {self.device}")
            
            # Initialize GroundingDINO model
            # The model will be downloaded automatically on first use
            self.model = Model(
                model_config_path="groundingdino/config/GroundingDINO_SwinT_OGC.py",
                model_checkpoint_path="groundingdino_swint_ogc.pth",
                device=self.device
            )
            
            print("✓ GroundingDINO model loaded successfully!")
            print(f"✓ Ready to detect: '{self.text_prompt}'")
            
        except Exception as e:
            print(f"✗ Failed to initialize GroundingDINO: {e}")
            print("✗ Falling back to manual detection mode.")
            self.model = None
            self.is_available = False
    
    def detect_objects(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect objects in frame using the text prompt.
        
        Args:
            frame: OpenCV frame (BGR format)
            
        Returns:
            List of bounding boxes in (x1, y1, x2, y2) format
        """
        if not self.is_available or self.model is None:
            return []
        
        try:
            # Convert BGR to RGB for the model
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run detection
            detections = self.model.predict_with_classes(
                image=frame_rgb,
                classes=[self.text_prompt],
                box_threshold=self.confidence_threshold,
                text_threshold=0.25
            )
            
            # Convert detections to our format
            boxes = []
            if len(detections.xyxy) > 0:
                for box in detections.xyxy:
                    x1, y1, x2, y2 = map(int, box)
                    boxes.append((x1, y1, x2, y2))
                
                print(f"✓ Detected {len(boxes)} object(s) matching '{self.text_prompt}'")
            else:
                print(f"✗ No objects found matching '{self.text_prompt}'")
            
            return boxes
            
        except Exception as e:
            print(f"✗ Detection error: {e}")
            return []
    
    def get_best_detection(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Get the highest confidence detection as a single bounding box.
        
        Args:
            frame: OpenCV frame (BGR format)
            
        Returns:
            Single bounding box (x1, y1, x2, y2) or None if no detection
        """
        boxes = self.detect_objects(frame)
        
        if boxes:
            # Return the first detection (highest confidence from the model)
            # You could add additional ranking logic here (size, position, etc.)
            best_box = boxes[0]
            print(f"✓ Best detection: {best_box}")
            return best_box
        
        return None
    
    def update_prompt(self, new_prompt: str):
        """
        Update the text prompt for detection.
        
        Args:
            new_prompt: New natural language description
        """
        self.text_prompt = new_prompt.lower().strip()
        print(f"✓ Updated detection prompt to: '{self.text_prompt}'")
    
    def set_confidence_threshold(self, threshold: float):
        """
        Update the confidence threshold for detections.
        
        Args:
            threshold: New confidence threshold (0.0-1.0)
        """
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        print(f"✓ Updated confidence threshold to: {self.confidence_threshold}")
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model and configuration.
        
        Returns:
            Dictionary with model information
        """
        return {
            "available": self.is_available,
            "device": str(self.device) if self.device else "N/A",
            "prompt": self.text_prompt,
            "confidence_threshold": self.confidence_threshold,
            "model_loaded": self.model is not None
        }


# Alternative lightweight detector for situations where GroundingDINO is not available
class FallbackDetector:
    """
    Fallback detector that provides manual detection interface.
    Used when GroundingDINO is not available.
    """
    
    def __init__(self, text_prompt: str):
        self.text_prompt = text_prompt
        self.is_available = False
    
    def detect_objects(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        return []
    
    def get_best_detection(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        return None
    
    def get_model_info(self) -> dict:
        return {
            "available": False,
            "device": "N/A",
            "prompt": self.text_prompt,
            "confidence_threshold": 0.0,
            "model_loaded": False
        }


def create_text_detector(prompt: str) -> TextDetector:
    """
    Factory function to create a text detector.
    
    Args:
        prompt: Natural language description for detection
        
    Returns:
        TextDetector instance (or FallbackDetector if dependencies missing)
    """
    if GROUNDING_DINO_AVAILABLE:
        return TextDetector(prompt)
    else:
        print("Using fallback detector (manual labeling only)")
        return FallbackDetector(prompt)


# Example usage and testing
if __name__ == "__main__":
    # Test the detector
    detector = create_text_detector("person")
    info = detector.get_model_info()
    
    print("\nText Detector Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    if info["available"]:
        print(f"\n✓ Ready for text-based detection!")
        print(f"✓ Try prompts like: 'person', 'car', 'person with red shirt', 'blue car'")
    else:
        print(f"\n✗ Text detection not available - manual labeling only")
