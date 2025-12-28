#!/usr/bin/env python3
"""
AI Glasses Camera Server

Provides MJPEG streaming and API endpoints for the camera viewer frontend.
Run with: python server.py
"""

import asyncio
import io
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

# Add parent to path for aiglasses imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aiohttp import web
from aiohttp.web import Response, StreamResponse

# Camera configuration
CAPTURES_DIR = Path(__file__).parent.parent / "captures"
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_QUALITY = 80
DEFAULT_FPS = 30


class CameraServer:
    """MJPEG streaming server with camera controls."""

    def __init__(self):
        self.camera = None
        self.detector = None
        self.width = DEFAULT_WIDTH
        self.height = DEFAULT_HEIGHT
        self.quality = DEFAULT_QUALITY
        self.target_fps = DEFAULT_FPS
        self.frame_interval = 1.0 / DEFAULT_FPS
        self.running = False
        self.detection_enabled = False
        self.current_frame: bytes | None = None
        self.current_frame_rgb: "np.ndarray | None" = None
        self.last_detections: list = []
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.fps_frame_count = 0

    async def setup(self):
        """Initialize the camera."""
        try:
            from aiglasses.foundation.camera.service import CameraService
            from aiglasses.config import Config
            
            config = Config()
            self.camera = CameraService(config, mock_mode=False)
            await self.camera.setup()
            self.running = True
            print(f"✓ Camera initialized: {self.width}x{self.height}")
            
            # Initialize Hailo object detector
            await self.setup_detector()
            
            return True
        except Exception as e:
            print(f"✗ Camera init failed: {e}")
            return False
    
    async def setup_detector(self, model: str = "yolov8s"):
        """Initialize Hailo object detector."""
        try:
            from aiglasses.foundation.vision.hailo_detector import HailoObjectDetector
            
            self.detector = HailoObjectDetector(
                model_name=model,
                confidence_threshold=0.4,
                max_detections=20,
            )
            if await self.detector.setup():
                self.detection_enabled = True
                print(f"✓ Object detector initialized: {model}")
            else:
                print("⚠ Object detector setup failed, continuing without detection")
        except Exception as e:
            print(f"⚠ Object detector not available: {e}")

    async def teardown(self):
        """Cleanup camera resources."""
        self.running = False
        self.detection_enabled = False
        if self.detector:
            await self.detector.teardown()
            self.detector = None
        if self.camera:
            await self.camera.teardown()
            self.camera = None

    async def capture_frame(self) -> bytes | None:
        """Capture a single JPEG frame."""
        if not self.camera:
            return None
        
        try:
            frame = await self.camera.snapshot(
                format="jpeg",
                quality=self.quality,
            )
            
            # Update FPS stats
            self.frame_count += 1
            self.fps_frame_count += 1
            now = time.time()
            elapsed = now - self.last_fps_time
            if elapsed >= 1.0:
                self.fps = int(self.fps_frame_count / elapsed)
                self.fps_frame_count = 0
                self.last_fps_time = now
            
            self.current_frame = frame.data
            return frame.data
        except Exception as e:
            print(f"Frame capture error: {e}")
            return None

    async def set_resolution(self, width: int, height: int):
        """Change camera resolution."""
        if self.camera:
            try:
                success = await self.camera.set_resolution(width, height)
                if success:
                    self.width = width
                    self.height = height
                    print(f"✓ Resolution changed to {width}x{height}")
                    return True
                else:
                    print(f"✗ Failed to change resolution to {width}x{height}")
                    return False
            except Exception as e:
                print(f"✗ Resolution change error: {e}")
                return False
        else:
            self.width = width
            self.height = height
            print(f"Resolution set to {width}x{height} (camera not ready)")
            return True


# Global server instance
server = CameraServer()


# --- API Handlers ---

async def handle_status(request: web.Request) -> Response:
    """Return camera status."""
    return web.json_response({
        "connected": server.running,
        "resolution": f"{server.width}x{server.height}",
        "fps": server.fps,
        "backend": "IMX500 / libcamera",
        "frame_count": server.frame_count,
    })


async def handle_resolution(request: web.Request) -> Response:
    """Change camera resolution."""
    width = int(request.query.get("width", DEFAULT_WIDTH))
    height = int(request.query.get("height", DEFAULT_HEIGHT))
    await server.set_resolution(width, height)
    return web.json_response({"success": True, "width": width, "height": height})


async def handle_capture(request: web.Request) -> Response:
    """Capture and save a snapshot."""
    frame = await server.capture_frame()
    if not frame:
        return web.json_response({"error": "Failed to capture"}, status=500)
    
    # Save to captures folder
    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"snapshot_{timestamp}.jpg"
    filepath = CAPTURES_DIR / filename
    
    with open(filepath, "wb") as f:
        f.write(frame)
    
    return web.json_response({
        "success": True,
        "path": str(filepath),
        "size": len(frame),
    })


async def handle_run_app(request: web.Request) -> Response:
    """Run a test app."""
    app_id = request.match_info["app_id"]
    
    result = ""
    
    if app_id == "what_am_i_wearing":
        result = await run_what_am_i_wearing()
    elif app_id == "what_am_i_seeing":
        result = await run_what_am_i_seeing()
    elif app_id == "object_detection":
        result = await run_object_detection()
    else:
        return web.json_response({"error": f"Unknown app: {app_id}"}, status=404)
    
    return web.json_response({"result": result})


async def run_what_am_i_wearing() -> str:
    """Run the 'what am I wearing' test."""
    try:
        # Capture frame
        frame = await server.capture_frame()
        if not frame:
            return "Error: Failed to capture frame"
        
        # Save to temp file for Claude to read
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = CAPTURES_DIR / timestamp
        session_dir.mkdir(parents=True, exist_ok=True)
        img_path = session_dir / "capture.jpg"
        
        with open(img_path, "wb") as f:
            f.write(frame)
        
        # Use Claude to analyze
        from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, AssistantMessage, TextBlock, ResultMessage
        
        options = ClaudeAgentOptions(
            model="haiku",
            allowed_tools=["Read"],
            permission_mode="acceptEdits",
        )
        
        prompt = f"""Read the image at {img_path} and tell me in 1-2 short sentences:
What is the person wearing? What colors do you see?
Be specific about clothing items and their colors."""

        response = ""
        cost = 0
        
        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)
            async for msg in client.receive_response():
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            response += block.text
                elif isinstance(msg, ResultMessage):
                    cost = msg.total_cost_usd or 0
        
        # Clean up response
        lines = [l.strip() for l in response.split('\n') if l.strip() and not l.startswith("I'll")]
        clean = ' '.join(lines) if lines else response
        
        # Save transcript
        with open(session_dir / "transcript.md", "w") as f:
            f.write(f"# What Am I Wearing - {timestamp}\n\n")
            f.write(f"## Response\n{clean}\n\n")
            f.write(f"## Cost\n${cost:.4f}\n")
        
        return f"📷 {img_path}\n\n{clean}\n\n💰 ${cost:.4f}"
        
    except Exception as e:
        return f"Error: {e}"


async def run_what_am_i_seeing() -> str:
    """Run the 'what am I seeing' test."""
    try:
        frame = await server.capture_frame()
        if not frame:
            return "Error: Failed to capture frame"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = CAPTURES_DIR / timestamp
        session_dir.mkdir(parents=True, exist_ok=True)
        img_path = session_dir / "capture.jpg"
        
        with open(img_path, "wb") as f:
            f.write(frame)
        
        from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, AssistantMessage, TextBlock, ResultMessage
        
        options = ClaudeAgentOptions(
            model="haiku",
            allowed_tools=["Read"],
            permission_mode="acceptEdits",
        )
        
        prompt = f"""Read the image at {img_path} and describe what you see in 2-3 sentences.
Focus on the main objects, people, and scene. Be concise."""

        response = ""
        cost = 0
        
        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)
            async for msg in client.receive_response():
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            response += block.text
                elif isinstance(msg, ResultMessage):
                    cost = msg.total_cost_usd or 0
        
        lines = [l.strip() for l in response.split('\n') if l.strip() and not l.startswith("I'll")]
        clean = ' '.join(lines) if lines else response
        
        with open(session_dir / "transcript.md", "w") as f:
            f.write(f"# What Am I Seeing - {timestamp}\n\n")
            f.write(f"## Response\n{clean}\n\n")
            f.write(f"## Cost\n${cost:.4f}\n")
        
        return f"📷 {img_path}\n\n{clean}\n\n💰 ${cost:.4f}"
        
    except Exception as e:
        return f"Error: {e}"


async def run_object_detection() -> str:
    """Run YOLO object detection on current frame."""
    try:
        import numpy as np
        import cv2
        
        if not server.detection_enabled or not server.detector:
            return "⚠ Object detection not available. Hailo AI HAT+ may not be connected."
        
        # Capture frame as JPEG
        frame_data = await server.capture_frame()
        if not frame_data:
            return "Error: Failed to capture frame"
        
        # Decode JPEG to numpy array
        nparr = np.frombuffer(frame_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return "Error: Failed to decode image"
        
        # Run detection
        result = await server.detector.detect(image)
        
        # Save annotated image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = CAPTURES_DIR / f"detection_{timestamp}"
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Draw detections on image
        annotated = image.copy()
        colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
        ]
        
        h, w = image.shape[:2]
        for i, det in enumerate(result.detections):
            x1 = int(det.bbox[0] * w)
            y1 = int(det.bbox[1] * h)
            x2 = int(det.bbox[2] * w)
            y2 = int(det.bbox[3] * h)
            
            color = colors[det.class_id % len(colors)]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            label = f"{det.class_name}: {det.confidence:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Save images
        cv2.imwrite(str(session_dir / "original.jpg"), image)
        cv2.imwrite(str(session_dir / "annotated.jpg"), annotated)
        
        # Save transcript
        with open(session_dir / "detections.md", "w") as f:
            f.write(f"# Object Detection - {timestamp}\n\n")
            f.write(f"**Model:** {result.model_name}\n")
            f.write(f"**Inference Time:** {result.inference_time_ms:.1f}ms\n")
            f.write(f"**Objects Detected:** {len(result.detections)}\n\n")
            
            if result.detections:
                f.write("## Detections\n\n")
                f.write("| Class | Confidence | Bounding Box |\n")
                f.write("|-------|------------|-------------|\n")
                for det in result.detections:
                    bbox_str = f"({det.bbox[0]:.2f}, {det.bbox[1]:.2f}, {det.bbox[2]:.2f}, {det.bbox[3]:.2f})"
                    f.write(f"| {det.class_name} | {det.confidence:.1%} | {bbox_str} |\n")
        
        # Build response
        if result.detections:
            det_summary = ", ".join([f"{d.class_name} ({d.confidence:.0%})" for d in result.detections[:5]])
            if len(result.detections) > 5:
                det_summary += f" +{len(result.detections) - 5} more"
        else:
            det_summary = "No objects detected"
        
        return f"""🔍 **Object Detection Results**

📷 {session_dir / 'annotated.jpg'}

**Found {len(result.detections)} objects** in {result.inference_time_ms:.1f}ms

{det_summary}

Model: {result.model_name}"""
        
    except Exception as e:
        import traceback
        return f"Error: {e}\n{traceback.format_exc()}"


async def handle_detect(request: web.Request) -> Response:
    """Run object detection and return JSON results."""
    try:
        import numpy as np
        import cv2
        
        if not server.detection_enabled or not server.detector:
            return web.json_response({
                "error": "Object detection not available",
                "detections": [],
            })
        
        # Capture frame
        frame_data = await server.capture_frame()
        if not frame_data:
            return web.json_response({"error": "Failed to capture", "detections": []})
        
        # Decode
        nparr = np.frombuffer(frame_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return web.json_response({"error": "Failed to decode", "detections": []})
        
        # Run detection
        result = await server.detector.detect(image)
        
        # Store for overlay
        server.last_detections = [d.to_dict() for d in result.detections]
        
        return web.json_response({
            "detections": server.last_detections,
            "count": len(result.detections),
            "inference_ms": round(result.inference_time_ms, 1),
            "model": result.model_name,
        })
        
    except Exception as e:
        return web.json_response({"error": str(e), "detections": [], "inference_ms": 0, "count": 0})


async def handle_detector_status(request: web.Request) -> Response:
    """Get object detector status."""
    return web.json_response({
        "enabled": server.detection_enabled,
        "model": server.detector.model_name if server.detector else None,
        "classes": server.detector.classes[:10] if server.detector else [],  # First 10 classes
        "total_classes": len(server.detector.classes) if server.detector else 0,
    })


async def handle_stream(request: web.Request) -> StreamResponse:
    """MJPEG stream endpoint."""
    # Get resolution from query params
    width = int(request.query.get("width", server.width))
    height = int(request.query.get("height", server.height))
    
    if width != server.width or height != server.height:
        await server.set_resolution(width, height)
    
    response = StreamResponse(
        status=200,
        reason="OK",
        headers={
            "Content-Type": "multipart/x-mixed-replace; boundary=frame",
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Access-Control-Allow-Origin": "*",
        },
    )
    await response.prepare(request)
    
    print(f"Stream started: {width}x{height}")
    
    try:
        while True:
            frame = await server.capture_frame()
            if frame:
                # MJPEG frame format
                await response.write(
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(frame)).encode() + b"\r\n"
                    b"\r\n" + frame + b"\r\n"
                )
            
            # Rate limiting
            await asyncio.sleep(server.frame_interval)
            
    except (ConnectionResetError, asyncio.CancelledError):
        print("Stream ended")
    
    return response


# --- App Setup ---

async def on_startup(app: web.Application):
    """Initialize camera on server start."""
    print("\n🎥 AI Glasses Camera Server")
    print("=" * 40)
    await server.setup()


async def on_cleanup(app: web.Application):
    """Cleanup on server shutdown."""
    await server.teardown()
    print("Server stopped")


def create_app() -> web.Application:
    """Create the aiohttp application."""
    app = web.Application()
    
    # Lifecycle
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    
    # Routes
    app.router.add_get("/api/status", handle_status)
    app.router.add_post("/api/resolution", handle_resolution)
    app.router.add_post("/api/capture", handle_capture)
    app.router.add_post("/api/run/{app_id}", handle_run_app)
    app.router.add_get("/api/detect", handle_detect)
    app.router.add_get("/api/detector", handle_detector_status)
    app.router.add_get("/stream", handle_stream)
    
    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8765))
    print(f"\n🚀 Starting server on http://0.0.0.0:{port}")
    print(f"   Stream: http://rpi5kavecany.local:{port}/stream")
    print(f"   API: http://rpi5kavecany.local:{port}/api/status\n")
    
    app = create_app()
    web.run_app(app, host="0.0.0.0", port=port, print=None)


