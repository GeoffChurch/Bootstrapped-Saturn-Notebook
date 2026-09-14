# Raytracing Saturn, bootstrapped

One marimo notebook, `BootstrappedSaturn.py`. The same model is asked for the
same picture four times, with more machinery each time. The first rung's
machinery is ours; every piece after that is written by the model itself, one
rung at a time. The notebook explains itself as you go. This file is only how
to start it.

```bash
uvx marimo edit --sandbox BootstrappedSaturn.py
```

That is the whole install. The dependency list is a PEP 723 block at the top of
the notebook, `--sandbox` builds an isolated environment from it with uv, and
nothing lands in your own Python. Without uv, install the packages that block
names and drop the flag.

## The key

The dropdown at the top defaults to GLM-5.3-Flash on OpenRouter, which needs
`OPENROUTER_API_KEY` in a `.env` beside the notebook, one `KEY=value` per line.
That file is gitignored, and the notebook stops with a message naming the
variable if it is missing. The Ollama entries need no key, but they do expect a
server on `localhost:11434` with that model pulled.

A key belongs in that file and nowhere else. Not in a cell, because the
notebook is tracked. And not typed at a prompt, because marimo copies what you
type at a prompt into the cell's console output, which an exported session then
carries.

## Gallery

`gallery/glm-5.3-flash.ipynb` is one sitting exported with its outputs, so you
can read a whole run, failures included, without making one.

## Safety note

Every rung executes model-written code on the machine running the notebook, and
rung 4 runs a model-written *agent*. Fine for a scratch directory on your own
laptop; not fine on a shared host.

What model-written code computes runs in a subprocess with a timeout and a two
gigabyte cap on its address space, which is what stops a bad array shape taking
the machine down. That cap is POSIX only, and rung 4's agent drives the model
from inside the kernel, so for a hard ceiling on everything, start the notebook
under `systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0`.
