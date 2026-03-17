"""Simple chat loop with an interrupt hotkey."""

import asyncio
import select
import sys
import termios
import threading
import tty

from fury import Agent

MODEL = "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_X"
INTERRUPT_KEY = "i"
INTERRUPTED_STATUS = "[interrupted]"
EXIT_COMMANDS = {"q", "quit", "exit"}

agent = Agent(
    model=MODEL,
    system_prompt="You are a helpful assistant.",
)


def wait_for_interrupt_key(stop_event: threading.Event, runner) -> None:
    if not sys.stdin.isatty():
        return

    fd = sys.stdin.fileno()
    original_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while not stop_event.is_set():
            if (
                select.select([sys.stdin], [], [], 0.05)[0]
                and sys.stdin.read(1).lower() == INTERRUPT_KEY
            ):
                runner.interrupt()
                return
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, original_settings)


async def main() -> None:
    print(f"Press {INTERRUPT_KEY!r} while streaming to interrupt the current reply.")
    print("Type 'quit' to exit.")
    history = []

    while True:
        print()
        user_input = input("> ")
        if user_input.strip().lower() in EXIT_COMMANDS:
            break

        history.append({"role": "user", "content": user_input})

        buffer = ""
        runner = agent.runner()
        stop_event = threading.Event()
        hotkey_task = asyncio.create_task(
            asyncio.to_thread(wait_for_interrupt_key, stop_event, runner)
        )

        try:
            async for event in runner.chat(history):
                if event.content:
                    buffer += event.content
                    print(event.content, end="", flush=True)
        finally:
            stop_event.set()
            await hotkey_task

        if runner.interrupted:
            print(f"\n{INTERRUPTED_STATUS}")
            continue

        history.append({"role": "assistant", "content": buffer})
        print()


if __name__ == "__main__":
    asyncio.run(main())
