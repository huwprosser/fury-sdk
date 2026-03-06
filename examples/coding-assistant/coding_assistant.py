"""
A coding assistant that uses the Fury agent library to help with coding tasks.

This example is based on the Pi.dev coding assistant example.

This example is a semi-vibe-coded clone of the Pi.dev coding assistant example to use the Fury agent library.

It strips away most of Pi's features but does include:
- Auto-compaction of history.
- AgentSkills system.
- SOUL.md and MEMORY.md injection.
- A handful of useful tools for the agent to use.

It is a simple example of how to use the Fury agent library to create a coding assistant.
"""

import asyncio
import base64
import mimetypes
import os
import tempfile
import subprocess
from dataclasses import dataclass
from termcolor import cprint
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel

from fury import Agent, HistoryManager, create_tool

MAX_LINES = 2000
MAX_BYTES = 100 * 1024
IMAGE_MIME_PREFIXES = {"image/png", "image/jpeg", "image/gif", "image/webp"}
SKILLS_DOC_PATH = "docs/skills.md"


class ReadInput(BaseModel):
    path: str
    offset: Optional[int] = None
    limit: Optional[int] = None


class BashInput(BaseModel):
    command: str
    timeout: Optional[int] = None


class WriteInput(BaseModel):
    path: str
    content: str


class EditInput(BaseModel):
    path: str
    old_text: str
    new_text: str


READ_OUTPUT_SCHEMA = {"oneOf": [{"type": "string"}, {"type": "object"}]}
BASH_OUTPUT_SCHEMA = {"type": "string"}
WRITE_OUTPUT_SCHEMA = {"type": "string"}
EDIT_OUTPUT_SCHEMA = {"type": "string"}


def get_model_schema(model: type[BaseModel]) -> Dict[str, Any]:
    if hasattr(model, "model_json_schema"):
        return model.model_json_schema()
    return model.schema()


@dataclass
class Skill:
    name: str
    description: str
    file_path: str
    base_dir: str
    disable_model_invocation: bool = False


# ------------------------------------------------------------------------------------------------ #
# TOOLS
# ------------------------------------------------------------------------------------------------ #


def resolve_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def truncate_text(text: str) -> tuple[str, bool, int, int]:
    lines = text.splitlines()
    total_lines = len(lines)
    total_bytes = len(text.encode("utf-8"))
    truncated = False

    if total_lines > MAX_LINES:
        lines = lines[-MAX_LINES:]
        truncated = True

    truncated_text = "\n".join(lines)
    if len(truncated_text.encode("utf-8")) > MAX_BYTES:
        truncated = True
        truncated_text = truncated_text.encode("utf-8")[-MAX_BYTES:].decode(
            "utf-8", errors="replace"
        )

    return truncated_text, truncated, total_lines, total_bytes


def read_text_file(path: str, offset: Optional[int], limit: Optional[int]) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        if offset is not None or limit is not None:
            start_line = max(offset or 1, 1)
            end_line = start_line + (limit - 1) if limit else None
            selected = []
            for index, line in enumerate(handle, start=1):
                if index < start_line:
                    continue
                if end_line is not None and index > end_line:
                    break
                selected.append(line)
            return "".join(selected) or "(no content)"

        tail_lines: deque[str] = deque()
        total_lines = 0
        kept_bytes = 0
        for line in handle:
            total_lines += 1
            line_bytes = len(line.encode("utf-8"))
            tail_lines.append(line)
            kept_bytes += line_bytes
            while len(tail_lines) > MAX_LINES or kept_bytes > MAX_BYTES:
                removed = tail_lines.popleft()
                kept_bytes -= len(removed.encode("utf-8"))

        content = "".join(tail_lines) or "(empty file)"
        if total_lines > len(tail_lines):
            start_line = total_lines - len(tail_lines) + 1
            content += (
                f"\n\n[Showing lines {start_line}-{total_lines} of {total_lines}. "
                "Use offset/limit to view earlier lines.]"
            )
        return content


def read_tool(path: str, offset: Optional[int] = None, limit: Optional[int] = None):
    print()
    cprint(f"Reading {path}...", "green")
    resolved_path = resolve_path(path)
    if not os.path.exists(resolved_path):
        return f"Error: path not found: {resolved_path}"

    mime_type, _ = mimetypes.guess_type(resolved_path)
    if mime_type in IMAGE_MIME_PREFIXES:
        with open(resolved_path, "rb") as handle:
            encoded = base64.b64encode(handle.read()).decode("utf-8")
        return {
            "description": f"Image read from {resolved_path}.",
            "image_base64": encoded,
        }

    return read_text_file(resolved_path, offset, limit)


def bash_tool(command: str, timeout: Optional[int] = None):
    try:
        print()
        cprint(f"Running command {command}...", "cyan")
        result = subprocess.run(
            command,
            shell=True,
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = (result.stdout or "") + (result.stderr or "")
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + (exc.stderr or "")
        output += f"\n\nCommand timed out after {timeout} seconds"
        return output

    output = output or "(no output)"
    truncated_text, truncated, total_lines, _ = truncate_text(output)
    if truncated:
        with tempfile.NamedTemporaryFile(
            delete=False, prefix="pi-bash-", suffix=".log", mode="w", encoding="utf-8"
        ) as handle:
            handle.write(output)
            full_path = handle.name
        truncated_text += (
            f"\n\n[Showing last {MAX_LINES} lines. Full output: {full_path}]"
        )

    if result.returncode != 0:
        truncated_text += f"\n\nCommand exited with code {result.returncode}"

    return truncated_text


def write_tool(path: str, content: str):
    print()
    cprint(f"Writing to {path}...", "magenta")
    resolved_path = resolve_path(path)
    os.makedirs(os.path.dirname(resolved_path) or ".", exist_ok=True)
    with open(resolved_path, "w", encoding="utf-8") as handle:
        handle.write(content)
    return f"Wrote {len(content)} characters to {resolved_path}"


def edit_tool(path: str, old_text: str, new_text: str):
    print()
    cprint(f"Editing {path}...", "red")
    resolved_path = resolve_path(path)
    if not os.path.exists(resolved_path):
        return f"Error: path not found: {resolved_path}"

    with open(resolved_path, "r", encoding="utf-8") as handle:
        content = handle.read()

    if old_text not in content:
        return "Error: old_text not found in file."

    updated = content.replace(old_text, new_text)
    with open(resolved_path, "w", encoding="utf-8") as handle:
        handle.write(updated)

    return f"Replaced {content.count(old_text)} occurrence(s) in {resolved_path}"


# ------------------------------------------------------------------------------------------------ #
# CORE FUNCTIONS
# ------------------------------------------------------------------------------------------------ #


def parse_skill_frontmatter(content: str) -> Dict[str, Any]:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}

    frontmatter: Dict[str, Any] = {}
    for index in range(1, len(lines)):
        line = lines[index]
        if line.strip() == "---":
            break
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value.lower() in {"true", "false"}:
            frontmatter[key] = value.lower() == "true"
        else:
            frontmatter[key] = value

    return frontmatter


def discover_skill_files(root_dir: str) -> List[str]:
    files: List[str] = []
    if not os.path.exists(root_dir):
        return files

    def _walk(path: str, root_level: bool) -> None:
        try:
            entries = list(os.scandir(path))
        except OSError:
            return

        for entry in entries:
            if entry.name.startswith(".") or entry.name == "node_modules":
                continue
            entry_path = entry.path
            try:
                if entry.is_dir(follow_symlinks=False):
                    _walk(entry_path, root_level=False)
                elif entry.is_file(follow_symlinks=False):
                    if root_level and entry.name.endswith(".md"):
                        files.append(entry_path)
                    elif not root_level and entry.name == "SKILL.md":
                        files.append(entry_path)
            except OSError:
                continue

    _walk(root_dir, root_level=True)

    return files


def load_skill_from_file(file_path: str) -> Optional[Skill]:
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as handle:
            content = handle.read()
    except OSError:
        return None

    frontmatter = parse_skill_frontmatter(content)
    description = str(frontmatter.get("description", "")).strip()
    if not description:
        return None

    base_dir = os.path.dirname(file_path)
    name = str(frontmatter.get("name") or os.path.basename(base_dir)).strip()
    disable_model_invocation = frontmatter.get("disable-model-invocation") is True

    return Skill(
        name=name,
        description=description,
        file_path=file_path,
        base_dir=base_dir,
        disable_model_invocation=disable_model_invocation,
    )


def load_skills() -> List[Skill]:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    search_dirs = [os.path.join(base_dir, "skills")]

    skills_by_name: Dict[str, Skill] = {}
    seen_paths: set[str] = set()

    for search_dir in search_dirs:
        for file_path in discover_skill_files(search_dir):
            real_path = os.path.realpath(file_path)
            if real_path in seen_paths:
                continue
            skill = load_skill_from_file(file_path)
            if not skill:
                continue
            if skill.name in skills_by_name:
                continue
            skills_by_name[skill.name] = skill
            seen_paths.add(real_path)

    return list(skills_by_name.values())


def escape_xml(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def format_skills_for_prompt(skills: List[Skill]) -> str:
    visible_skills = [skill for skill in skills if not skill.disable_model_invocation]
    if not visible_skills:
        return ""

    lines = [
        "The following skills provide specialized instructions for specific tasks.",
        "Use the read tool to load a skill's file when the task matches its description.",
        "",
        "<available_skills>",
    ]

    for skill in visible_skills:
        lines.append("  <skill>")
        lines.append(f"    <name>{escape_xml(skill.name)}</name>")
        lines.append(f"    <description>{escape_xml(skill.description)}</description>")
        lines.append(f"    <location>{escape_xml(skill.file_path)}</location>")
        lines.append("  </skill>")

    lines.append("</available_skills>")

    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ #
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------------------------ #


def format_token_count(count: int) -> str:
    if count < 1000:
        return str(count)
    if count < 10000:
        return f"{count / 1000:.1f}k"
    if count < 1000000:
        return f"{round(count / 1000)}k"
    if count < 10000000:
        return f"{count / 1000000:.1f}M"
    return f"{round(count / 1000000)}M"


def load_context_files() -> List[tuple[str, str]]:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = ["SOUL.md", "MEMORY.md"]
    context_files: List[tuple[str, str]] = []

    for filename in candidates:
        path = os.path.join(base_dir, filename)
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                content = handle.read().strip()
        except OSError:
            continue
        if content:
            context_files.append((path, content))

    return context_files


def build_prompt() -> str:
    now = datetime.now().astimezone()
    date_time = (
        f"{now.strftime('%A, %B ')}"
        f"{now.day}"
        f"{now.strftime(', %Y at %I:%M:%S %p %Z')}"
    )
    cwd = os.getcwd()
    context_files = load_context_files()
    skills_section = format_skills_for_prompt(load_skills())

    prompt = f"""You are an expert coding assistant operating inside pi, a coding agent harness. You help users by reading files, executing commands, editing code, and writing new files.

Available tools:
- read: Read file contents
- bash: Execute bash commands (ls, grep, find, etc.)
- edit: Make surgical edits to files (find exact text and replace)
- write: Create or overwrite files
- multi_tool_use.parallel: Run multiple tool calls in parallel (use when independent tools can run together).  

Guidelines:
- Use bash for file operations like ls, rg, find
- Use read to examine files before editing. You must use this tool instead of cat or sed.
- Use edit for precise changes (old text must match exactly)
- Use write only for new files or complete rewrites
- When summarizing your actions, output plain text directly - do NOT use cat or bash to display what you did
- Be concise in your responses. Also sarcastic and witty.
- Show file paths clearly when working with files
- When a skill applies, read its SKILL.md and follow any linked docs before acting.
- Use multi_tool_use.parallel to batch independent tool calls instead of sequential calls.            
- Before creating or modifying a skill, ALWAYS read docs/skills.md and follow it exactly.

MEMORY.md: Access long-term facts, user preferences, and historical context from MEMORY.md when relevant to the conversation."""

    if context_files:
        prompt += "\n\n# Project Context\n\n"
        prompt += "Project-specific instructions and guidelines:\n\n"
        for path, content in context_files:
            prompt += f"## {path}\n\n{content}\n\n"

    if skills_section:
        prompt += skills_section

    prompt += (
        f"\n\nCurrent date and time: {date_time}\nCurrent working directory: {cwd}"
    )
    return prompt


async def main():
    load_dotenv()

    agent = Agent(
        model="unsloth/GLM-4.6V-Flash-GGUF:Q8_0",
        system_prompt=build_prompt(),
        tools=[
            create_tool(
                "read",
                "Read file contents (text or image). Supports offset/limit for text files.",
                read_tool,
                "Reading [path]...",
                get_model_schema(ReadInput),
                READ_OUTPUT_SCHEMA,
            ),
            create_tool(
                "bash",
                "Execute a bash command in the current working directory.",
                bash_tool,
                "Running command...",
                get_model_schema(BashInput),
                BASH_OUTPUT_SCHEMA,
            ),
            create_tool(
                "write",
                "Write content to a file, creating parent directories if needed.",
                write_tool,
                "Writing to [path]...",
                get_model_schema(WriteInput),
                WRITE_OUTPUT_SCHEMA,
            ),
            create_tool(
                "edit",
                "Replace exact text in a file with new text.",
                edit_tool,
                "Editing [path]...",
                get_model_schema(EditInput),
                EDIT_OUTPUT_SCHEMA,
            ),
        ],
    )

    history_manager = HistoryManager(agent=agent)

    while True:
        user_input = input("> ").strip()
        if not user_input:
            continue
        await history_manager.add({"role": "user", "content": user_input})

        buffer = []
        last_stream_kind = None
        async for event in agent.chat(history_manager.history, True):
            if event.content:
                if last_stream_kind == "reasoning":
                    print()
                last_stream_kind = "chunk"
                buffer.append(event.content)
                print(event.content, end="", flush=True)

            if event.reasoning:
                if last_stream_kind == "chunk":
                    print()
                last_stream_kind = "reasoning"
                cprint(event.reasoning, "grey", end="", flush=True)

        print()
        await history_manager.add({"role": "assistant", "content": "".join(buffer)})

        context_tokens, context_percent = history_manager.get_context_usage()
        cprint(
            f"Context: {format_token_count(context_tokens)}/{format_token_count(history_manager.context_window)} "
            f"({context_percent:.1f}%)",
            "blue",
        )


if __name__ == "__main__":
    asyncio.run(main())
