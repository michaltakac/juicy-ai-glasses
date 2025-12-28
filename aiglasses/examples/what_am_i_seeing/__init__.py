"""What Am I Seeing? - Example AI Glasses App.

This example app demonstrates the core AI Glasses functionality:
1. User speaks to AirPods (voice input)
2. User asks: "What am I seeing?" (or similar)
3. System captures a frame from the IMX500 camera
4. System performs object recognition
5. System uses an LLM to generate a natural language answer
6. Answer is spoken back via TTS and displayed in logs
"""

from examples.what_am_i_seeing.app import main, WhatAmISeeingApp

__all__ = ["main", "WhatAmISeeingApp"]


