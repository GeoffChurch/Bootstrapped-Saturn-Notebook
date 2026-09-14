"""The contract a harness must satisfy to plug into `mo.ui.chat`.

    python3 harness_test.py my_harness.py        # prints PASS or a failed assertion

A harness is a module defining `agent_chat(messages, config)`: a generator that
yields progress as strings, drives a model with tools until the task is done,
and may `return` the OpenAI-shaped message list it built. It passes when its
tools really ran, which it proves by creating a file with the right contents.
"""

import importlib.util
import itertools
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Message:
    """What marimo's chat widget actually hands over. Use `role` and `content`."""

    role: str
    content: str
    id: str = "m1"
    parts: list = field(default_factory=list)
    attachments: list | None = None


TASK = "Create a file called hello.py containing exactly this line: print('generated harness works')"


def test_harness(agent_chat):
    stream = agent_chat([Message("user", TASK)], config={})
    yields = []
    try:
        for chunk in itertools.islice(stream, 400):
            yields.append(chunk)
    except StopIteration:
        pass
    assert yields, "must yield at least one progress string"
    assert all(isinstance(y, str) for y in yields), f"every yield must be a str, got {type(yields[-1]).__name__}"
    assert len(yields) < 400, "did not stop within 400 yields"
    hello = Path("hello.py")
    assert hello.exists(), "hello.py was never created: the tools did not run"
    assert "generated harness works" in hello.read_text(), f"hello.py says {hello.read_text()[:60]!r}"


def load(path):
    spec = importlib.util.spec_from_file_location("candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    candidate = Path(sys.argv[1]).resolve()
    workdir = tempfile.mkdtemp(prefix="harness-test-")
    os.chdir(workdir)  # the harness writes to the current directory
    test_harness(load(candidate).agent_chat)
    print(f"PASS: {candidate.name}")
