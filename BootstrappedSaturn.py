# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.24",
#     "nbformat",   # so the editor can export this session to the gallery itself
#     "numpy",
#     "openai",
#     "pillow",
# ]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import base64
    import contextlib
    import importlib.util
    import json
    import os
    import re
    import subprocess
    import sys
    import tempfile
    import time
    from pathlib import Path
    from typing import NamedTuple

    import marimo as mo
    from openai import OpenAI

    HERE = Path(__file__).parent
    WORK = HERE / "work"
    # One folder per sitting, so its log and everything it made travel together.
    SESSION = WORK / time.strftime("%Y-%m-%d-%H%M%S")
    SESSION.mkdir(parents=True, exist_ok=True)
    # Code the model writes uses relative paths, so the working directory decides
    # where its files land. Make that work/ (gitignored), not the notebook directory.
    os.chdir(WORK)

    # Keys live in a gitignored .env next to the notebook, one KEY=value per line.
    # Never in a cell, and never typed into a prompt: marimo records what you type
    # at a prompt in the cell's console output, which an export then carries.
    for _line in (HERE / ".env").read_text().splitlines() if (HERE / ".env").exists() else []:
        if "=" in _line and not _line.startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

    def api_key(var: str | None) -> str:
        """The key for an endpoint, from .env. Empty means it is not set."""
        if var is None:
            return "local"  # Ollama ignores it; the OpenAI client insists on a non-empty string
        return os.environ.get(var, "")

    return (
        HERE,
        NamedTuple,
        OpenAI,
        Path,
        SESSION,
        WORK,
        api_key,
        base64,
        contextlib,
        importlib,
        json,
        mo,
        os,
        re,
        subprocess,
        sys,
        tempfile,
        time,
    )


@app.cell
def _(NamedTuple):
    class Model(NamedTuple):
        base_url: str
        key_var: str | None  # the environment variable holding the key, if one is needed
        id: str
        thinking_off: dict  # what to send to switch reasoning off; endpoints differ

    OLLAMA = "http://localhost:11434/v1"  # local, or a remote box forwarded to this port
    OPENROUTER = "https://openrouter.ai/api/v1"

    # Every entry does vision and tool calls, which rungs 2 and 4 need.
    # OpenRouter prices are $ per million tokens in / out, checked 2026-09-13.
    MODELS = {
        "z-ai/glm-5.3-flash on OpenRouter ($0.15 / $0.50)": Model(OPENROUTER, "OPENROUTER_API_KEY", "z-ai/glm-5.3-flash", {"reasoning": {"effort": "minimal"}}),
        "qwen3.8:27b on Ollama (free, local)": Model(OLLAMA, None, "qwen3.8:27b", {"reasoning_effort": "none"}),
        "gemma4:31b-it-qat on Ollama (free, local)": Model(OLLAMA, None, "gemma4:31b-it-qat", {"reasoning_effort": "none"}),
        "qwen/qwen3.8-27b on OpenRouter ($0.21 / $2.55)": Model(OPENROUTER, "OPENROUTER_API_KEY", "qwen/qwen3.8-27b", {"reasoning_effort": "none"}),
        "deepseek/deepseek-v4.1-flash on OpenRouter ($0.15 / $0.60)": Model(OPENROUTER, "OPENROUTER_API_KEY", "deepseek/deepseek-v4.1-flash", {"reasoning": {"enabled": False}}),
        "google/gemini-3.8-flash on OpenRouter ($0.75 / $3.75)": Model(OPENROUTER, "OPENROUTER_API_KEY", "google/gemini-3.8-flash", {"reasoning": {"enabled": False}}),
    }
    return (MODELS,)


@app.cell(hide_code=True)
def _(MODELS, mo, os):
    # `MODEL=...` picks the entry, for running the notebook from a script. Otherwise
    # the first entry, which is the one the measurements in the README recommend.
    _default = next((k for k, m in MODELS.items() if m.id == os.environ.get("MODEL")), next(iter(MODELS)))
    model_ui = mo.ui.dropdown(options=MODELS, value=_default, label="model")
    thinking_ui = mo.ui.checkbox(value=False, label="thinking")
    # Widgets only render in the editor; an export shows the status line below instead.
    mo.hstack([model_ui, thinking_ui], justify="start", gap=2) if mo.app_meta().mode == "edit" else None
    return model_ui, thinking_ui


@app.cell(hide_code=True)
def _(OpenAI, api_key, json, mo, model_ui, os, thinking_ui):
    # Changing the model re-runs every rung below. Pick it first.
    MODEL = model_ui.value.id
    EXTRA = {} if thinking_ui.value else model_ui.value.thinking_off
    # Private, so the key reaches neither the dataflow graph nor any cell output.
    _key = api_key(model_ui.value.key_var)
    mo.stop(
        not _key,
        mo.md(f"🔑 Put `{model_ui.value.key_var}=...` in a `.env` beside the notebook.").callout("danger"),
    )
    # The environment is the interface to the code the model writes: its harness
    # calls OpenAI() with no arguments, here and in the subprocesses we spawn.
    os.environ["OPENAI_BASE_URL"] = model_ui.value.base_url
    os.environ["OPENAI_API_KEY"] = _key
    os.environ["MODEL"] = MODEL
    os.environ["EXTRA_BODY"] = json.dumps(EXTRA)
    client = OpenAI(timeout=240)
    mo.md(f"✅ `{MODEL}` · thinking {'on' if thinking_ui.value else 'off'} · sends `extra_body={json.dumps(EXTRA)}`")
    return EXTRA, MODEL, client


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Raytracing Saturn, bootstrapped

    The same model, asked for the same picture, four times. Each time it gets
    more machinery. The first rung's is ours; every piece after that is written
    by the model itself, one rung at a time, in front of you.

    | rung | who decides what happens next | written by |
    |---|---|---|
    | 1. one shot | nobody: one request | us, in the cells below |
    | 2. the loop | the notebook: generate, run, review, repeat | the model, from a description |
    | 3. the harness | the model: it has tools and picks the steps | the model, driven by its own loop |
    | 4. the final render | the harness | -- |

    This notebook *is* the conversation. A turn is one line, `turn("...")`, and
    its reply renders underneath. A follow-up is `turn("...", after=t1)`, and
    because that dependency is explicit, editing an early prompt re-runs every
    turn that came after it. Code the model writes becomes importable with
    `t.module("name")`, so nothing is pasted by hand.

    Thinking is off by default so the rungs go fast. Tick the box and compare.
    Changing the model or the box re-runs every rung below, so choose first.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## What we hand-roll

    Four small things do the work with the model:

    * `turn` -- one prompt to the model, streamed, with the conversation carried
      along explicitly. `permit` hands out the same thing without the display,
      for code the model writes to call.
    * `run_render` and `run_test` -- how to *run* what the model wrote, each in a
      subprocess with a timeout and a memory cap. The first executes a raytracer
      and captures the array it returns; the second runs `harness_test.py`
      against a candidate harness. They are the *environment*. The model writes
      the *agent*.
    * `run_harness` -- drives a model-written harness on a task and shows its
      progress.

    One folder per sitting, a folder per model inside it, and a directory per
    run inside that: `work/2026-09-14-124906/z-ai-glm-5.3-flash/render-4cn7veq8/`.
    One thing per path segment. Re-running a cell never picks up the last run's
    files, two models never mix, every round's output stays around to look at,
    and a sitting is one folder you can hand over whole. A `Result` carries the
    directory it used.

    Everything slow or file-writing reports through `log`, which prints a line
    and appends it to the `log.md` in that sitting's folder. Markdown, so a render
    is linked next to the line about it and a sitting reads back as one page once
    the kernel is gone. Those are the seams: every directory through `run_dir`, every report
    through `log`, and every model call through `permit`.

    `permit` is also the brake. It asks before anything is sent and then hands
    back the function that sends it, so you can run cells at random and be
    stopped at exactly the ones that cost time and money. Nothing else can reach
    the model: the function it returns is private to that cell, so no other cell
    can name it. An export has nobody to ask, so there it goes straight through.
    """)
    return


@app.cell
def _(
    EXTRA,
    MODEL,
    NamedTuple,
    Path,
    SESSION,
    WORK,
    base64,
    client,
    contextlib,
    importlib,
    mo,
    re,
    subprocess,
    sys,
    tempfile,
    time,
):
    FENCE = re.compile(r"^```(?:python)?[ \t]*\n(.*?)^```[ \t]*$", re.S | re.M)
    LOG = SESSION / "log.md"


    def log(message: str, image: Path | None = None) -> None:
        """Every slow or file-writing step reports here: the cell console, and a log.md
        in this session's folder, which outlives the kernel. Markdown, so `image` can be
        linked where it was made and a sitting reads as one page of text and pictures."""
        line = f"{time.strftime('%H:%M:%S')}  {message}"
        print(line)
        with LOG.open("a") as handle:
            handle.write(f"- {line}\n")
            if image is not None:
                handle.write(f"\n  ![]({image.relative_to(SESSION)})\n\n")


    def _ask(prompt: str, image: bytes | None = None, *, history=()) -> str:
        """One request, no display. `image` is a PNG for the model to look at.

        Private on purpose: `permit` is the only way to get hold of this."""
        content = [{"type": "text", "text": prompt}]
        if image is not None:
            data = base64.standard_b64encode(image).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}})
        started = time.monotonic()
        reply = client.chat.completions.create(
            model=MODEL, messages=[*history, {"role": "user", "content": content}],
            max_tokens=16000, extra_body=EXTRA,
        )
        out = reply.usage.completion_tokens if reply.usage else "?"
        log(f"ask: {time.monotonic() - started:.0f}s, {out} tokens out{', with an image' if image else ''}")
        return reply.choices[0].message.content or ""


    def permit(what: str = "this cell"):
        """Say yes before anything is sent, then get back the function that sends it.

        Every path to the model runs through here, so running cells at random stops
        at exactly the ones that cost time and money, and nowhere else. Handing the
        result to `loop` means the loop cannot be given a connection nobody agreed
        to. Outside the editor -- an export -- there is nobody to ask, so it goes."""
        if mo.app_meta().mode == "edit":
            if input(f"{what} will call {MODEL}. Type y to allow: ").strip().lower() != "y":
                raise RuntimeError(f"{what}: declined, nothing was sent")
        log(f"{what}: allowed to call {MODEL}")
        return _ask


    def as_module(code: str, name: str):
        """Write `code` to work/<name>.py and import it, so the next cell can call into it."""
        path = WORK / f"{name}.py"
        path.write_text(code)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


    class Turn(NamedTuple):
        """One exchange. `history` is the whole conversation up to and including it."""

        prompt: str
        reply: str
        history: tuple

        @property
        def code(self) -> str:
            """The first python block in the reply."""
            return FENCE.search(self.reply).group(1)

        def module(self, name: str):
            return as_module(self.code, name)


    def turn(prompt: str, after: Turn | None = None) -> Turn:
        """Send one prompt, streaming the reply into this cell's output."""
        permit("this turn")
        history = after.history if after else ()
        started = time.monotonic()
        stream = client.chat.completions.create(
            model=MODEL, messages=[*history, {"role": "user", "content": prompt}],
            max_tokens=16000, stream=True, extra_body=EXTRA,
        )
        reply, painted = "", 0.0
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                reply += chunk.choices[0].delta.content
            if time.monotonic() - painted > 0.5:
                mo.output.replace(mo.md(reply))
                painted = time.monotonic()
        mo.output.replace(mo.md(reply))
        log(f"turn: {time.monotonic() - started:.0f}s, {len(reply)} characters back")
        return Turn(prompt, reply, (*history, {"role": "user", "content": prompt}, {"role": "assistant", "content": reply}))


    class Png(NamedTuple):
        """A picture the notebook hands back. `_repr_png_` makes marimo emit a real
        image/png, which an export keeps."""

        data: bytes

        def _repr_png_(self) -> bytes:
            return self.data


    class Result(NamedTuple):
        """What running the model's code produced. `image` is a PNG."""

        stdout: str
        stderr: str
        image: bytes | None
        dir: Path  # where it ran, so you can go and look


    # The whole id: two endpoints can offer the same model name.
    RUN_ROOT = SESSION / re.sub(r"[^A-Za-z0-9._-]+", "-", MODEL)[:40]


    def run_dir(label: str) -> Path:
        """A new directory per run: work/<sitting>/<model>/<what made it>-<unique>.

        One thing per path segment. Re-running a cell never sees the last run's
        files, two models never mix, and a sitting is one folder you can hand over
        whole. Every runner goes through here, so saying where it is once covers
        all of them."""
        RUN_ROOT.mkdir(parents=True, exist_ok=True)
        here = Path(tempfile.mkdtemp(dir=RUN_ROOT, prefix=f"{label}-"))
        log(f"{label}: {here.relative_to(SESSION)}/")
        return here


    MEMORY_CAP_GB = 2  # a legitimate 800x600 render needs a few megabytes
    # POSIX only; on Windows there is no cap. An over-allocation then fails as a
    # numpy error naming the shape, which the reviewer reads and the model can act on.
    CAP = "" if sys.platform == "win32" else (
        "import resource\n"
        f"_bytes = {MEMORY_CAP_GB} * 1024**3\n"
        "resource.setrlimit(resource.RLIMIT_AS, (_bytes, _bytes))\n"
    )

    RENDER_DRIVER = CAP + (
        "import numpy as np, saturn; from PIL import Image\n"
        "Image.fromarray(np.asarray(saturn.render()).astype('uint8')).save('out.png')"
    )

    TEST_DRIVER = CAP + "import runpy; runpy.run_path('harness_test.py', run_name='__main__')"


    def run_render(code: str, timeout: int = 180) -> Result:
        """Run a raytracer that defines `render() -> HxWx3 uint8 array`, in a subprocess."""
        here = run_dir("render")
        (here / "saturn.py").write_text(code)
        try:
            done = subprocess.run(
                [sys.executable, "-c", RENDER_DRIVER], cwd=here, capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            return Result("", f"killed after {timeout}s", None, here)
        png = here / "out.png"
        log(f"render: {'a picture' if png.exists() else 'no picture'}, {len(done.stderr)} characters on stderr",
            image=png if png.exists() else None)
        return Result(done.stdout[-4000:], done.stderr[-4000:], png.read_bytes() if png.exists() else None, here)


    def run_test(code: str, timeout: int = 900) -> Result:
        """Run harness_test.py against a candidate harness. PASS or a failed assert is in the output."""
        here = run_dir("harness")
        (here / "my_harness.py").write_text(code)
        done = subprocess.run(
            [sys.executable, "-c", TEST_DRIVER, str(here / "my_harness.py")],
            cwd=WORK.parent, capture_output=True, text=True, timeout=timeout,
        )
        log(f"harness: {'PASS' if 'PASS' in done.stdout else 'no PASS'}")
        return Result(done.stdout[-4000:], done.stderr[-4000:], None, here)


    def show(result: Result):
        """The image if there is one, else the output."""
        if result.image:
            return Png(result.image)
        return mo.md(f"```\n{result.stdout}\n{result.stderr}\n```").callout("warn")


    def run_harness(agent_chat, task: str, where: Path):
        """Drive a model-written harness on `task` in `where`, showing its progress as it goes."""
        permit("this harness")  # it builds its own client, so it cannot go through `permit`
        from harness_test import Message

        lines = []
        with contextlib.chdir(where):  # the harness writes to the working directory
            gen = agent_chat([Message("user", task)], config={})
            while True:
                try:
                    lines.append(next(gen))
                except StopIteration as done:
                    mo.output.replace(mo.md("\n\n".join(lines)))
                    return done.value  # the message list, if the harness returns one
                mo.output.replace(mo.md("\n\n".join(lines)))

    return (
        Png,
        as_module,
        permit,
        run_dir,
        run_harness,
        run_render,
        run_test,
        show,
        turn,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Rung 1 — one shot

    One request. The contract is a function, not a file: `render()` returns the
    picture as an array, and *we* decide what to do with it -- in a subprocess,
    so an array the wrong shape costs us a subprocess and not the notebook.

    If it fails, that is the exhibit. To try again, add a cell:
    `t1b = turn("It failed with: ...", after=t1)` and point the next cell at it.
    """)
    return


@app.cell
def _():
    SATURN_SPEC = (
        "Write a Python module that raytraces Saturn -- an oblate planet with a ring "
        "system -- using only numpy. Define `render()` returning an 800x600x3 uint8 "
        "array. The rings must cast their shadows on the planet and the planet must "
        "cast its shadow on the rings. Vectorise with numpy; it must finish in under "
        "60 seconds. Reply with the complete module in one ```python block."
    )
    return (SATURN_SPEC,)


@app.cell
def _(SATURN_SPEC, turn):
    t1 = turn(SATURN_SPEC)
    return (t1,)


@app.cell
def _(run_render, show, t1):
    result1 = run_render(t1.code) # This might fail (which is okay) as the model had to write the script with no feedback
    show(result1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Rung 2 — the model writes the loop

    Now describe a loop and let the model write it. The loop never gets tools.
    The *notebook* decides what happens: generate, run, show the model what
    happened, ask whether it is done, repeat. The model only ever answers
    questions.
    """)
    return


@app.cell
def _():
    LOOP_SPEC = """Write a Python function `loop(task, run, ask, max_rounds=8)` and nothing else:
    no other top-level definitions, classes, or imports (import inside the function
    if you must).

    It is an iterative code-writing loop with no tools. The pieces it is given:
    * `task`: a string describing a program to write.
    * `run(code) -> Result`: executes the code and returns a NamedTuple with
      `.stdout`, `.stderr`, and `.image` (PNG bytes, or None).
    * `ask(prompt, image=None) -> str`: sends one prompt to a language model,
      optionally with a PNG for it to look at, and returns the reply text.

    Each round:
    1. Ask the model for the code (the first round: the task; later rounds: the
       task, the previous code, and the reviewer's feedback). Extract the code
       from the first ```python block in the reply.
    2. `run` it.
    3. Ask the model to REVIEW: show it the code, the stdout and stderr, and the
       image if there is one -- and if there is none, say so explicitly. Tell it
       to reply with exactly the word DONE if the result fully satisfies the task,
       or otherwise a list of what must change; and tell it that a run which
       crashed with a traceback, or produced no image when the task asks for one,
       is never DONE. Warnings on stderr are not crashes.
    4. If the reviewer's reply, ignoring surrounding whitespace, is DONE, stop.
       Otherwise carry the feedback into the next round. The loop itself never
       judges the result; only the reviewer's verdict decides.

    Print one line per round saying what happened. Return `(code, result)` from
    the last round. Reply with the function in one ```python block."""
    return (LOOP_SPEC,)


@app.cell
def _(LOOP_SPEC, turn):
    t2 = turn(LOOP_SPEC)
    return (t2,)


@app.cell
def _(SATURN_SPEC, permit, run_render, show, t2):
    loop = t2.module("loop").loop
    code2, result2 = loop(SATURN_SPEC, run_render, permit("rung 2"))
    show(result2)
    return (loop,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Rung 3 — the loop writes the harness

    Same loop, different task: write an agent *with* tools. The run step is now
    `harness_test.py`, which is both the acceptance test and the interface: a
    candidate passes when its tools really execute, which it proves by creating
    a file with the right contents. The reviewer reads PASS or a failed assert.

    This is the first rung where the model decides its own steps. Read what it
    wrote: that is an agentic loop.
    """)
    return


@app.cell
def _(HERE):
    HARNESS_SPEC = f"""Write a Python module that implements a coding agent with tools, and nothing else.
    It must pass this test, which is also the interface -- read it:

    ```python
    {(HERE / "harness_test.py").read_text()}
    ```

    Yield one short progress line per tool call -- the tool's name and its `path`
    argument only, never file contents -- each ending in a newline, then
    the model's final answer, and `return` the message list you built. Everything
    the agent writes goes in the current working directory.

    Use the `openai` package's chat completions API with tool calling. The client
    is `OpenAI()` with no arguments -- OPENAI_BASE_URL and OPENAI_API_KEY are set in
    the environment -- and the model name is in the MODEL environment variable.
    Pass `extra_body=json.loads(os.environ.get("EXTRA_BODY", "{{}}"))` on every request.

    Give the model three tools: `write_file(path, content)`, `run_python(path)`
    (returns exit code, stdout, stderr), and `look_at(path)` for a PNG, whose
    text result must state the image's width and height in pixels. A tool
    message can only carry text, so for `look_at` return a short note as the tool
    result and then append a user message with the image as a base64 data URL.
    Loop: request with tools, execute every tool call, append the results, repeat
    until the model replies without tool calls or 25 turns pass.

    Reply with the module in one ```python block."""
    return (HARNESS_SPEC,)


@app.cell
def _(HARNESS_SPEC, loop, permit, run_test):
    code3, result3 = loop(HARNESS_SPEC, run_test, permit("rung 3"))
    print(result3.stdout)
    return (code3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Rung 4 — Saturn, one last time

    The harness the model wrote, on the original task. Watch for the moment it
    calls `look_at` and then changes something.
    """)
    return


@app.cell
def _(SATURN_SPEC, as_module, code3, run_dir, run_harness):
    agent_chat = as_module(code3, "harness").agent_chat
    SATURN_PNG_SPEC = SATURN_SPEC.replace("using only numpy", "using only numpy and Pillow").replace(
        "Define `render()` returning an 800x600x3 uint8 array", "Save it as saturn.png")
    SATURN_DIR = run_dir("saturn")  # empty, so the picture below can only be this run's
    transcript = run_harness(agent_chat, SATURN_PNG_SPEC, SATURN_DIR)
    return SATURN_DIR, transcript


@app.cell
def _(Png, SATURN_DIR, mo, transcript):
    _png = SATURN_DIR / "saturn.png"
    mo.vstack([
        Png(_png.read_bytes()) if _png.exists() else mo.md("**no saturn.png**").callout("warn"),
        mo.md(f"_{len(transcript or [])} messages in the harness transcript_"),
    ])
    return


if __name__ == "__main__":
    app.run()
