import os

from fury.utils.console import silence_console_output


def test_silence_console_output_suppresses_python_and_fd_writes(capfd):
    print("before")

    with silence_console_output():
        print("hidden stdout")
        os.write(1, b"hidden fd stdout\n")
        os.write(2, b"hidden fd stderr\n")

    print("after")
    captured = capfd.readouterr()

    assert "hidden stdout" not in captured.out
    assert "hidden fd stdout" not in captured.out
    assert "hidden fd stderr" not in captured.err
    assert "before" in captured.out
    assert "after" in captured.out
