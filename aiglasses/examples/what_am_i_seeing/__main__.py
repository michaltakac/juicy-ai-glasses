"""Entry point for running as python -m examples.what_am_i_seeing."""

import asyncio
import argparse

from examples.what_am_i_seeing.app import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="What Am I Seeing? AI Glasses App")
    parser.add_argument("--mock", action="store_true", help="Run with mock services")
    parser.add_argument("--ptt", action="store_true", help="Push-to-talk mode")
    args = parser.parse_args()

    asyncio.run(main(mock=args.mock, ptt=args.ptt))


