import asyncio
from fury import HistoryManager
from fury.multimodal import IMAGE_HISTORY_PLACEHOLDER, materialize_history_message


def test_history_manager_accepts_initial_history():
    manager = HistoryManager(
        history=[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ],
        target_context_length=100,
    )

    assert manager.history == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]


def test_history_manager_reduces_to_target_context_length_on_init():
    manager = HistoryManager(
        history=[
            {"role": "user", "content": "a" * 80},
            {"role": "assistant", "content": "b" * 80},
            {"role": "user", "content": "latest"},
        ],
        target_context_length=25,
    )

    assert manager.history == [
        {"role": "assistant", "content": "b" * 80},
        {"role": "user", "content": "latest"},
    ]


def test_history_manager_add_and_extend_reduce_history():
    manager = HistoryManager(target_context_length=10)

    asyncio.run(manager.add({"role": "user", "content": "a" * 40}))
    asyncio.run(manager.extend([
        {"role": "assistant", "content": "b" * 40},
        {"role": "user", "content": "c"},
    ]))

    assert manager.history == [
        {"role": "user", "content": "c"},
    ]


def test_history_manager_keeps_single_oversized_latest_message():
    manager = HistoryManager(target_context_length=1)

    asyncio.run(manager.extend([
        {"role": "user", "content": "short"},
        {"role": "assistant", "content": "x" * 100},
    ]))

    assert manager.history == [{"role": "assistant", "content": "x" * 100}]


def test_history_manager_clear():
    manager = HistoryManager(
        history=[{"role": "user", "content": "hello"}],
        target_context_length=100,
    )

    assert manager.clear() == []
    assert manager.history == []



def test_history_manager_reports_context_usage():
    manager = HistoryManager(
        history=[{"role": "user", "content": "a" * 40}],
        target_context_length=20,
    )

    tokens, percent = manager.get_context_usage()

    assert tokens == 10
    assert percent == 50


def test_history_manager_adds_image_placeholder_by_default(tmp_path):
    image_path = tmp_path / "tiny.png"
    image_path.write_bytes(b"fake image")
    manager = HistoryManager(target_context_length=1000)

    asyncio.run(manager.add_image(str(image_path), text="look"))

    message = manager.history[0]
    assert message["role"] == "user"
    assert IMAGE_HISTORY_PLACEHOLDER in str(message["content"])
    assert "base64," not in str(message["content"])

    materialized = materialize_history_message(message)
    assert materialized["role"] == "user"


def test_history_manager_can_store_full_image_payload(tmp_path):
    image_path = tmp_path / "tiny.png"
    image_path.write_bytes(b"fake image")
    manager = HistoryManager(target_context_length=1000, save_images_to_history=True)

    asyncio.run(manager.add_image(str(image_path), text="look"))

    assert "base64," in str(manager.history[0]["content"])
