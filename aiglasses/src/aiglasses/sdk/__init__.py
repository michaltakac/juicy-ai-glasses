"""AI Glasses SDK - Build apps for the AI Glasses Platform.

This SDK provides a high-level API for building AI-powered applications
that run on the AI Glasses Platform.

Example usage:

    from aiglasses.sdk import App, AudioAPI, VisionAPI, LLMAPI

    async def main():
        app = App("my-app")
        await app.start()

        # Listen for wake word
        async for wake in app.audio.on_wake_word():
            # Capture what user sees
            frame = await app.vision.snapshot()
            detections = await app.vision.detect_objects(frame)

            # Ask LLM to describe
            answer = ""
            async for chunk in app.llm.chat([
                {"role": "user", "content": f"What do you see? Objects: {detections}"}
            ]):
                answer += chunk.content

            # Speak the answer
            await app.audio.speak(answer)

        await app.stop()

"""

from aiglasses.sdk.app import App
from aiglasses.sdk.audio import AudioAPI
from aiglasses.sdk.vision import VisionAPI
from aiglasses.sdk.llm import LLMAPI
from aiglasses.sdk.events import EventBus, subscribe, publish
from aiglasses.sdk.manifest import AppManifest, Permission

__all__ = [
    "App",
    "AudioAPI",
    "VisionAPI",
    "LLMAPI",
    "EventBus",
    "subscribe",
    "publish",
    "AppManifest",
    "Permission",
]


