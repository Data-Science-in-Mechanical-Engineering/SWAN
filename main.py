#!/usr/bin/env python3
"""SWAN drone show pipeline - entry point."""

import asyncio
import sys

# Import pipeline first so that the matplotlib backend is configured before
# Gradio is loaded by `ui.ui`.
from swan.pipeline import main_async
from ui.ui import create_interface


def main():
    if "--cli" in sys.argv:
        asyncio.run(main_async())
    else:
        demo = create_interface()
        demo.launch(server_port=7860, share=False, server_name="0.0.0.0")


if __name__ == "__main__":
    main()
