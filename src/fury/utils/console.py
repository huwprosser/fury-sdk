from __future__ import annotations

import contextlib
import logging
import os
import sys
import warnings
from typing import Iterator, List, Tuple

logger = logging.getLogger(__name__)


def _restore_file_descriptors(saved_fds: List[Tuple[int, int]]) -> None:
    for fd, saved_fd in reversed(saved_fds):
        try:
            os.dup2(saved_fd, fd)
        finally:
            os.close(saved_fd)


@contextlib.contextmanager
def silence_console_output() -> Iterator[None]:
    """Temporarily silence Python and native library stdout/stderr output."""

    ctranslate2_module = None
    previous_log_level = None
    saved_fds: List[Tuple[int, int]] = []

    try:
        try:
            import ctranslate2

            ctranslate2_module = ctranslate2
            previous_log_level = ctranslate2.get_log_level()
            ctranslate2.set_log_level(logging.ERROR)
        except Exception:
            ctranslate2_module = None
            previous_log_level = None

        with open(os.devnull, "w") as devnull_stream:
            try:
                for stream in (sys.stdout, sys.stderr):
                    flush = getattr(stream, "flush", None)
                    if callable(flush):
                        flush()

                devnull_stream_fd = devnull_stream.fileno()
                for fd in (1, 2):
                    saved_fds.append((fd, os.dup(fd)))
                    os.dup2(devnull_stream_fd, fd)
            except OSError:
                _restore_file_descriptors(saved_fds)
                saved_fds = []

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with contextlib.redirect_stdout(devnull_stream):
                    with contextlib.redirect_stderr(devnull_stream):
                        yield
    finally:
        if saved_fds:
            _restore_file_descriptors(saved_fds)
        if ctranslate2_module is not None and previous_log_level is not None:
            try:
                ctranslate2_module.set_log_level(previous_log_level)
            except Exception:
                logger.debug("Failed to restore ctranslate2 log level.", exc_info=True)
