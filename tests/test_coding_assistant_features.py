from contextlib import contextmanager
from types import SimpleNamespace


def test_coding_assistant_discovers_root_docs_and_nested_skills(tmp_path, coding_assistant_module):
    root_doc = tmp_path / "docs.md"
    root_doc.write_text("root", encoding="utf-8")
    nested_skill_dir = tmp_path / "skills" / "demo"
    nested_skill_dir.mkdir(parents=True)
    nested_skill = nested_skill_dir / "SKILL.md"
    nested_skill.write_text("skill", encoding="utf-8")

    files = coding_assistant_module.discover_skill_files(str(tmp_path))

    assert str(root_doc) in files
    assert str(nested_skill) in files


def test_coding_assistant_read_tool_returns_tail_for_large_file(tmp_path, coding_assistant_module):
    target = tmp_path / "large.txt"
    lines = [f"line {index}" for index in range(coding_assistant_module.MAX_LINES + 25)]
    target.write_text("\n".join(lines), encoding="utf-8")

    content = coding_assistant_module.read_tool(str(target))

    assert "line 0" not in content
    assert f"line {coding_assistant_module.MAX_LINES + 24}" in content
    assert "Use offset/limit to view earlier lines." in content


def test_coding_assistant_read_tool_returns_image_payload_for_images(tmp_path, coding_assistant_module):
    target = tmp_path / "image.png"
    target.write_bytes(b"fake-image")

    result = coding_assistant_module.read_tool(str(target))

    assert result["description"].startswith("Image read from ")
    assert result["image_base64"] == "ZmFrZS1pbWFnZQ=="


def test_coding_assistant_bash_tool_truncates_large_output_and_reports_exit_code(
    monkeypatch, coding_assistant_module
):
    huge_output = "\n".join(f"line {index}" for index in range(coding_assistant_module.MAX_LINES + 10))
    written = {}

    @contextmanager
    def fake_named_temporary_file(**kwargs):
        class Handle:
            name = "/tmp/fake.log"

            def write(self, value):
                written["value"] = value

        yield Handle()

    monkeypatch.setattr(
        coding_assistant_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=huge_output, stderr="", returncode=3),
    )
    monkeypatch.setattr(
        coding_assistant_module.tempfile,
        "NamedTemporaryFile",
        fake_named_temporary_file,
    )

    result = coding_assistant_module.bash_tool("fake command")

    assert "Showing last" in result
    assert "Full output: /tmp/fake.log" in result
    assert "Command exited with code 3" in result
    assert written["value"] == huge_output
