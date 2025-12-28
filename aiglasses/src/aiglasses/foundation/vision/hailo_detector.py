#!/usr/bin/env python3
"""
Hailo Object Detection using YOLO models optimized for Raspberry Pi 5 + AI HAT+.

Supports:
- YOLOv8s (80 COCO classes)
- YOLOv6n (80 COCO classes) 
- YOLOv5 person/face (2 classes)
- YOLOvX (80 COCO classes)
"""

import asyncio
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import structlog

log = structlog.get_logger()

# COCO class names for YOLO models
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

# Person/Face classes for specialized model
PERSONFACE_CLASSES = ["unlabeled", "person", "face"]


@dataclass
class Detection:
    """A single detected object."""
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 normalized 0-1
    
    def to_dict(self) -> dict:
        return {
            "class_id": int(self.class_id),
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 3),
            "bbox": [round(float(v), 4) for v in self.bbox],
        }


@dataclass 
class DetectionResult:
    """Result of object detection on a frame."""
    detections: list[Detection] = field(default_factory=list)
    inference_time_ms: float = 0.0
    model_name: str = ""
    
    def to_dict(self) -> dict:
        return {
            "detections": [d.to_dict() for d in self.detections],
            "count": len(self.detections),
            "inference_time_ms": round(self.inference_time_ms, 2),
            "model": self.model_name,
        }


class HailoObjectDetector:
    """
    YOLO object detection using Hailo AI accelerator.
    
    Optimized for Raspberry Pi 5 with AI HAT+ (Hailo-8/8L).
    """
    
    # Available models with their paths and configurations
    MODELS = {
        "yolov8s": {
            "hef_h8": "/usr/share/hailo-models/yolov8s_h8.hef",
            "hef_h8l": "/usr/share/hailo-models/yolov8s_h8l.hef",
            "classes": COCO_CLASSES,
            "input_size": (640, 640),
        },
        "yolov6n": {
            "hef_h8": "/usr/share/hailo-models/yolov6n_h8.hef",
            "hef_h8l": "/usr/share/hailo-models/yolov6n_h8l.hef",
            "classes": COCO_CLASSES,
            "input_size": (640, 640),
        },
        "yolov5_personface": {
            "hef_h8": "/usr/share/hailo-models/yolov5s_personface_h8l.hef",
            "hef_h8l": "/usr/share/hailo-models/yolov5s_personface_h8l.hef",
            "classes": PERSONFACE_CLASSES,
            "input_size": (640, 640),
        },
        "yolox_s": {
            "hef_h8": "/usr/share/hailo-models/yolox_s_leaky_h8l_rpi.hef",
            "hef_h8l": "/usr/share/hailo-models/yolox_s_leaky_h8l_rpi.hef",
            "classes": COCO_CLASSES,
            "input_size": (640, 640),
        },
    }
    
    def __init__(
        self,
        model_name: str = "yolov8s",
        confidence_threshold: float = 0.4,
        nms_threshold: float = 0.5,
        max_detections: int = 20,
    ):
        """
        Initialize the Hailo object detector.
        
        Args:
            model_name: One of 'yolov8s', 'yolov6n', 'yolov5_personface', 'yolox_s'
            confidence_threshold: Minimum confidence for detections
            nms_threshold: IoU threshold for non-max suppression
            max_detections: Maximum number of detections to return
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        
        self._vdevice = None
        self._network_group = None
        self._input_vstreams = None
        self._output_vstreams = None
        self._input_vstream_info = None
        self._output_vstream_info = None
        self._configured = False
        
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}. Choose from: {list(self.MODELS.keys())}")
        
        self._model_config = self.MODELS[model_name]
        self._classes = self._model_config["classes"]
        self._input_size = self._model_config["input_size"]
        
    async def setup(self) -> bool:
        """Initialize the Hailo device and load the model."""
        try:
            from hailo_platform import (
                VDevice, HEF, ConfigureParams,
                InputVStreamParams, OutputVStreamParams,
                FormatType, HailoStreamInterface
            )
            
            # Create virtual device
            self._vdevice = VDevice()
            
            # Try H8 first (full Hailo-8), then H8L (Hailo-8L)
            hef_path = None
            for key in ["hef_h8", "hef_h8l"]:
                candidate = self._model_config.get(key)
                if candidate and Path(candidate).exists():
                    hef_path = candidate
                    break
            
            if not hef_path:
                log.error("hailo_no_model_found", model=self.model_name)
                return False
            
            log.info("hailo_loading_model", model=self.model_name, hef=hef_path)
            
            # Load HEF
            self._hef = HEF(hef_path)
            
            # Configure network
            configure_params = ConfigureParams.create_from_hef(self._hef, interface=HailoStreamInterface.PCIe)
            self._network_group = self._vdevice.configure(self._hef, configure_params)[0]
            
            # Get stream info
            self._input_vstream_info = self._hef.get_input_vstream_infos()
            self._output_vstream_info = self._hef.get_output_vstream_infos()
            
            # Create vstream params - input as UINT8, output as FLOAT32
            self._input_params = InputVStreamParams.make_from_network_group(
                self._network_group, quantized=False, format_type=FormatType.UINT8
            )
            self._output_params = OutputVStreamParams.make_from_network_group(
                self._network_group, quantized=False, format_type=FormatType.FLOAT32
            )
            
            self._configured = True
            log.info("hailo_model_loaded", 
                    model=self.model_name,
                    input_shape=self._input_vstream_info[0].shape if self._input_vstream_info else None,
                    num_classes=len(self._classes))
            
            return True
            
        except Exception as e:
            log.error("hailo_setup_failed", error=str(e))
            return False
    
    async def teardown(self):
        """Release Hailo resources."""
        if self._vdevice:
            self._vdevice.release()
            self._vdevice = None
        self._configured = False
        log.info("hailo_detector_teardown")
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for YOLO inference.
        
        Args:
            image: BGR or RGB image as numpy array
            
        Returns:
            Preprocessed float32 array ready for inference
        """
        import cv2
        
        h, w = image.shape[:2]
        target_h, target_w = self._input_size
        
        # Resize with letterboxing to maintain aspect ratio
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create letterboxed image
        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        
        # Keep as uint8 - Hailo expects UINT8 input!
        preprocessed = canvas
        
        # Store scaling info for post-processing
        self._scale_info = {
            "scale": scale,
            "pad_x": pad_x,
            "pad_y": pad_y,
            "orig_w": w,
            "orig_h": h,
        }
        
        return preprocessed
    
    def _postprocess(self, outputs: dict, orig_shape: tuple) -> list[Detection]:
        """
        Post-process YOLO NMS outputs from Hailo.
        
        Hailo's YOLOv8 NMS postprocess output format:
        - outputs[name] = [batch][class_id] -> array of shape (num_detections, 5)
        - The 5 values are: [y1, x1, y2, x2, confidence] in normalized coords (0-1)
        
        Args:
            outputs: Raw model outputs from Hailo
            orig_shape: Original image shape (h, w)
            
        Returns:
            List of Detection objects
        """
        detections = []
        
        # Get the NMS output (it's a nested list structure)
        nms_output = None
        for name, data in outputs.items():
            if 'nms' in name.lower() or 'postprocess' in name.lower():
                nms_output = data
                break
        
        if nms_output is None:
            # Fallback to first output
            nms_output = list(outputs.values())[0] if outputs else None
        
        if nms_output is None:
            log.warning("hailo_no_output")
            return detections
        
        # Handle Hailo NMS output format: [batch][class_id] -> detections array
        # First level is batch (usually 1 element)
        # Second level is per-class detections (80 elements for COCO)
        try:
            # Get first batch
            if isinstance(nms_output, list) and len(nms_output) > 0:
                batch_output = nms_output[0]
            else:
                batch_output = nms_output
            
            # batch_output should be a list of 80 arrays (one per class)
            if not isinstance(batch_output, (list, tuple)):
                log.warning("hailo_unexpected_output_type", type=type(batch_output).__name__)
                return detections
            
            num_classes = len(batch_output)
            total_detections = 0
            
            # Iterate through each class
            for class_id, class_detections in enumerate(batch_output):
                if class_detections is None:
                    continue
                    
                # Convert to numpy array if needed
                if isinstance(class_detections, list):
                    if len(class_detections) == 0:
                        continue
                    class_detections = np.array(class_detections)
                
                if not isinstance(class_detections, np.ndarray):
                    continue
                
                if class_detections.size == 0:
                    continue
                
                # Ensure 2D: (num_detections, 5)
                if len(class_detections.shape) == 1:
                    class_detections = class_detections.reshape(1, -1)
                
                # Process each detection for this class
                for det in class_detections:
                    if len(det) < 5:
                        continue
                    
                    # Hailo NMS format: [y1, x1, y2, x2, confidence]
                    y1, x1, y2, x2, confidence = det[:5]
                    
                    if confidence < self.confidence_threshold:
                        continue
                    
                    total_detections += 1
                    
                    # Coordinates are already normalized 0-1, but we need to
                    # account for letterboxing we applied during preprocessing
                    scale = self._scale_info["scale"]
                    pad_x = self._scale_info["pad_x"] 
                    pad_y = self._scale_info["pad_y"]
                    orig_h, orig_w = orig_shape[:2]
                    input_h, input_w = self._input_size
                    
                    # Convert from letterboxed coords to original image coords
                    # The model outputs coords in the 640x640 input space
                    x1_px = x1 * input_w
                    y1_px = y1 * input_h
                    x2_px = x2 * input_w
                    y2_px = y2 * input_h
                    
                    # Remove padding and scale back
                    x1_orig = (x1_px - pad_x) / scale
                    y1_orig = (y1_px - pad_y) / scale
                    x2_orig = (x2_px - pad_x) / scale
                    y2_orig = (y2_px - pad_y) / scale
                    
                    # Normalize to 0-1
                    x1_norm = max(0, min(1, x1_orig / orig_w))
                    y1_norm = max(0, min(1, y1_orig / orig_h))
                    x2_norm = max(0, min(1, x2_orig / orig_w))
                    y2_norm = max(0, min(1, y2_orig / orig_h))
                    
                    if x2_norm > x1_norm and y2_norm > y1_norm:
                        class_name = self._classes[class_id] if class_id < len(self._classes) else f"class_{class_id}"
                        
                        detections.append(Detection(
                            class_id=class_id,
                            class_name=class_name,
                            confidence=float(confidence),
                            bbox=(x1_norm, y1_norm, x2_norm, y2_norm),
                        ))
            
            try:
                log.info("hailo_postprocess_complete", total_raw=total_detections, after_filter=len(detections))
            except BrokenPipeError:
                pass  # Ignore broken pipe in logging
            
        except Exception as e:
            try:
                log.error("hailo_postprocess_error", error=str(e))
            except BrokenPipeError:
                pass
        
        # Apply NMS
        detections = self._nms(detections)
        
        # Limit detections
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)[:self.max_detections]
        
        return detections
    
    def _nms(self, detections: list[Detection]) -> list[Detection]:
        """Apply Non-Maximum Suppression."""
        if not detections:
            return []
        
        # Group by class
        by_class: dict[int, list[Detection]] = {}
        for d in detections:
            by_class.setdefault(d.class_id, []).append(d)
        
        result = []
        for class_id, class_dets in by_class.items():
            # Sort by confidence
            class_dets = sorted(class_dets, key=lambda d: d.confidence, reverse=True)
            
            keep = []
            while class_dets:
                best = class_dets.pop(0)
                keep.append(best)
                
                # Remove overlapping boxes
                remaining = []
                for d in class_dets:
                    iou = self._compute_iou(best.bbox, d.bbox)
                    if iou < self.nms_threshold:
                        remaining.append(d)
                class_dets = remaining
            
            result.extend(keep)
        
        return result
    
    def _compute_iou(self, box1: tuple, box2: tuple) -> float:
        """Compute IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    async def detect(self, image: np.ndarray) -> DetectionResult:
        """
        Run object detection on an image.
        
        Args:
            image: BGR or RGB image as numpy array
            
        Returns:
            DetectionResult with list of detections
        """
        import time
        
        try:
            log.debug("hailo_detect_called", configured=self._configured, image_shape=image.shape if image is not None else None)
        except BrokenPipeError:
            pass
        
        if not self._configured:
            try:
                log.warning("hailo_detect_not_configured")
            except BrokenPipeError:
                pass
            return DetectionResult(model_name=self.model_name)
        
        start_time = time.perf_counter()
        
        try:
            from hailo_platform import InferVStreams
            
            # Preprocess
            preprocessed = self._preprocess(image)
            
            # Add batch dimension if needed
            if len(preprocessed.shape) == 3:
                preprocessed = np.expand_dims(preprocessed, 0)
            
            # Create input dict
            input_name = self._input_vstream_info[0].name
            input_data = {input_name: preprocessed}
            
            # Run inference
            with InferVStreams(self._network_group, self._input_params, self._output_params) as pipeline:
                with self._network_group.activate():
                    outputs = pipeline.infer(input_data)
            
            # Post-process
            detections = self._postprocess(outputs, image.shape)
            
            inference_time = (time.perf_counter() - start_time) * 1000
            
            try:
                log.info("hailo_detection_complete", num_detections=len(detections), inference_ms=inference_time)
            except BrokenPipeError:
                pass
            
            return DetectionResult(
                detections=detections,
                inference_time_ms=inference_time,
                model_name=self.model_name,
            )
            
        except Exception as e:
            import traceback
            try:
                log.error("hailo_inference_error", error=str(e), traceback=traceback.format_exc())
            except BrokenPipeError:
                pass
            return DetectionResult(model_name=self.model_name)
    
    @property
    def classes(self) -> list[str]:
        """Get list of class names this model can detect."""
        return self._classes.copy()
    
    @property
    def is_ready(self) -> bool:
        """Check if detector is ready for inference."""
        return self._configured


# Alternative: Use rpicam-apps with Hailo postprocessing (simpler approach)
class RpicamHailoDetector:
    """
    Object detection using rpicam-apps with Hailo postprocessing.
    
    This is a simpler alternative that uses the system's rpicam-apps
    with built-in Hailo integration.
    """
    
    CONFIGS = {
        "yolov8": "/usr/share/rpi-camera-assets/hailo_yolov8_inference.json",
        "yolov6": "/usr/share/rpi-camera-assets/hailo_yolov6_inference.json",
        "yolox": "/usr/share/rpi-camera-assets/hailo_yolox_inference.json",
        "personface": "/usr/share/rpi-camera-assets/hailo_yolov5_personface.json",
    }
    
    def __init__(self, model: str = "yolov8"):
        self.model = model
        self.config_path = self.CONFIGS.get(model)
        self._process = None
        
    async def start_detection_stream(
        self,
        output_callback,
        width: int = 1280,
        height: int = 720,
        framerate: int = 30,
    ):
        """
        Start detection stream using rpicam-apps.
        
        Args:
            output_callback: Async callback for detection results
            width: Video width
            height: Video height  
            framerate: Frames per second
        """
        import subprocess
        import json
        
        if not self.config_path or not Path(self.config_path).exists():
            log.error("rpicam_config_not_found", config=self.config_path)
            return
        
        cmd = [
            "rpicam-hello",
            "--post-process-file", self.config_path,
            "--width", str(width),
            "--height", str(height),
            "--framerate", str(framerate),
            "-t", "0",  # Run indefinitely
            "--metadata", "-",  # Output metadata to stdout
        ]
        
        log.info("starting_rpicam_detection", cmd=" ".join(cmd))
        
        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        
        # Read detection output
        while True:
            line = await self._process.stdout.readline()
            if not line:
                break
            
            try:
                data = json.loads(line.decode())
                if "detections" in data:
                    await output_callback(data["detections"])
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
    
    async def stop(self):
        """Stop the detection stream."""
        if self._process:
            self._process.terminate()
            await self._process.wait()
            self._process = None

