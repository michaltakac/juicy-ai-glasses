"""What Am I Seeing? - Main application.

This app listens for voice commands and describes what the camera sees.

Usage:
    # With mock services (development)
    python -m examples.what_am_i_seeing --mock

    # With real hardware
    python -m examples.what_am_i_seeing

    # Push-to-talk mode (no wake word detection)
    python -m examples.what_am_i_seeing --ptt
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import AsyncIterator

from aiglasses.common.logging import get_logger, setup_logging
from aiglasses.sdk import App
from aiglasses.sdk.vision import Detection


# Latency budgets (milliseconds)
LATENCY_BUDGET = {
    "wake_to_stt": 700,  # Wake word to STT result
    "snapshot": 120,  # Frame capture
    "detection": 50,  # Object detection
    "llm_first_token": 1500,  # LLM first token
    "tts_start": 300,  # TTS playback start
}


@dataclass
class QueryResult:
    """Result of a user query."""

    question: str
    answer: str
    detections: list[Detection]
    latency_ms: dict


class WhatAmISeeingApp:
    """Main application class for 'What Am I Seeing?' functionality."""

    def __init__(self, mock_mode: bool = False, ptt_mode: bool = False) -> None:
        """Initialize the app.

        Args:
            mock_mode: Run with mock services.
            ptt_mode: Push-to-talk mode (no wake word).
        """
        self.mock_mode = mock_mode
        self.ptt_mode = ptt_mode
        self.app = App("what-am-i-seeing", mock_mode=mock_mode)
        self.logger = get_logger("what-am-i-seeing")

        # State
        self._running = False
        self._last_query: QueryResult | None = None

    async def start(self) -> None:
        """Start the application."""
        self.logger.info(
            "app_starting",
            mock_mode=self.mock_mode,
            ptt_mode=self.ptt_mode,
        )

        await self.app.start()
        self._running = True

        self.logger.info("app_started")

        # Announce startup
        if not self.ptt_mode:
            await self.app.audio.speak(
                "Hello! Say 'Hey Glasses' followed by your question."
            )

    async def stop(self) -> None:
        """Stop the application."""
        self.logger.info("app_stopping")
        self._running = False
        await self.app.stop()
        self.logger.info("app_stopped")

    async def run(self) -> None:
        """Run the main application loop."""
        await self.start()

        try:
            if self.ptt_mode:
                await self._run_ptt_loop()
            else:
                await self._run_wake_word_loop()
        finally:
            await self.stop()

    async def _run_wake_word_loop(self) -> None:
        """Run loop with wake word detection."""
        self.logger.info("waiting_for_wake_word")

        async for wake in self.app.audio.on_wake_word():
            if not self._running:
                break

            self.logger.info(
                "wake_detected",
                wake_word=wake.wake_word,
                confidence=wake.confidence,
            )

            # Process the query
            await self._handle_query()

    async def _run_ptt_loop(self) -> None:
        """Run loop in push-to-talk mode."""
        self.logger.info("ptt_mode_active")
        print("\n[Push-to-Talk Mode]")
        print("Press Enter to ask a question, or 'q' to quit.\n")

        while self._running:
            try:
                # Wait for user input
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, input, "Press Enter to ask (q to quit): "
                )

                if user_input.lower() == "q":
                    break

                await self._handle_query()

            except EOFError:
                break

    async def _handle_query(self) -> None:
        """Handle a single user query."""
        timings: dict[str, int] = {}
        start_time = time.time()

        try:
            # Step 1: Listen and transcribe
            self.logger.info("listening_for_question")

            if self.mock_mode:
                # In mock mode, use default question
                question = "What am I seeing?"
            else:
                # Record and transcribe audio
                audio_chunks = []
                async for chunk in self.app.audio.listen():
                    audio_chunks.append(chunk.data)
                    # Stop after ~2 seconds of audio
                    if len(audio_chunks) >= 20:
                        break

                audio_data = b"".join(audio_chunks)
                result = await self.app.audio.transcribe(audio_data)
                question = result.text

            timings["stt"] = int((time.time() - start_time) * 1000)
            self.logger.info("question_transcribed", question=question)

            # Step 2: Capture frame and detect objects (parallel)
            snapshot_start = time.time()

            frame = await self.app.vision.snapshot()
            timings["snapshot"] = int((time.time() - snapshot_start) * 1000)

            detect_start = time.time()
            detection_result = await self.app.vision.detect_objects(frame)
            timings["detection"] = int((time.time() - detect_start) * 1000)

            self.logger.info(
                "objects_detected",
                count=len(detection_result.detections),
                labels=[d.label for d in detection_result.detections],
            )

            # Step 3: Generate answer with LLM
            llm_start = time.time()

            # Build context-aware prompt
            prompt = self._build_prompt(question, detection_result.detections)

            # Stream LLM response
            answer = ""
            first_token_time: float | None = None
            async for chunk in self.app.llm.chat([
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt},
            ]):
                if first_token_time is None and chunk.content:
                    first_token_time = time.time()
                    timings["llm_first_token"] = int((first_token_time - llm_start) * 1000)

                answer += chunk.content

            timings["llm_total"] = int((time.time() - llm_start) * 1000)
            self.logger.info("answer_generated", answer_length=len(answer))

            # Step 4: Speak the answer
            tts_start = time.time()
            await self.app.audio.speak(answer)
            timings["tts"] = int((time.time() - tts_start) * 1000)

            # Record result
            timings["total"] = int((time.time() - start_time) * 1000)
            self._last_query = QueryResult(
                question=question,
                answer=answer,
                detections=detection_result.detections,
                latency_ms=timings,
            )

            self.logger.info(
                "query_complete",
                question=question,
                answer_preview=answer[:100] + "..." if len(answer) > 100 else answer,
                timings=timings,
            )

            # Print to console
            self._print_result(self._last_query)

        except Exception as e:
            self.logger.exception("query_failed", error=str(e))
            await self.app.audio.speak(
                "I'm sorry, I encountered an error. Please try again."
            )

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the LLM."""
        return """You are an AI assistant built into smart glasses. Your job is to help 
the user understand what they're seeing. Be concise, helpful, and conversational.

Guidelines:
- Keep responses brief (2-3 sentences unless more detail is requested)
- Describe the most important elements first
- If you notice anything safety-relevant, mention it
- Be natural and friendly in tone
- Don't say "I can see" - the user knows you're using their camera"""

    def _build_prompt(self, question: str, detections: list[Detection]) -> str:
        """Build the prompt for the LLM.

        Args:
            question: User's question.
            detections: Detected objects.

        Returns:
            Formatted prompt.
        """
        # Format detections
        if detections:
            detection_str = ", ".join(
                f"{d.label} ({d.confidence:.0%})"
                for d in sorted(detections, key=lambda x: -x.confidence)
            )
            context = f"Objects detected in the scene: {detection_str}."
        else:
            context = "No specific objects were detected in the scene."

        return f"""{context}

User's question: {question}

Please answer the user's question based on what's in the scene."""

    def _print_result(self, result: QueryResult) -> None:
        """Print query result to console."""
        print("\n" + "=" * 60)
        print(f"Question: {result.question}")
        print("-" * 60)
        print(f"Objects: {', '.join(d.label for d in result.detections)}")
        print("-" * 60)
        print(f"Answer: {result.answer}")
        print("-" * 60)
        print("Latency breakdown:")
        for key, value in result.latency_ms.items():
            budget = LATENCY_BUDGET.get(key)
            status = ""
            if budget:
                status = " ✓" if value <= budget else f" ⚠ (budget: {budget}ms)"
            print(f"  {key}: {value}ms{status}")
        print("=" * 60 + "\n")


async def main(mock: bool = False, ptt: bool = False) -> None:
    """Run the What Am I Seeing app.

    Args:
        mock: Run in mock mode.
        ptt: Push-to-talk mode.
    """
    setup_logging(level="INFO")

    app = WhatAmISeeingApp(mock_mode=mock, ptt_mode=ptt)
    await app.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="What Am I Seeing? AI Glasses App")
    parser.add_argument("--mock", action="store_true", help="Run with mock services")
    parser.add_argument("--ptt", action="store_true", help="Push-to-talk mode")
    args = parser.parse_args()

    asyncio.run(main(mock=args.mock, ptt=args.ptt))


