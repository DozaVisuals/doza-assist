"""Single source of truth for the Ollama base URL.

The Electron wrapper runs a *bundled* Ollama on a dynamically chosen port
(to avoid colliding with a user's own system Ollama) and tells the core
about it via the OLLAMA_HOST env var (e.g. "127.0.0.1:11436"). The core
must honor that — hardcoding localhost:11434 only works on machines that
also happen to run a system Ollama on the default port (e.g. a dev box),
and fails with "connection refused" on a clean install.

Use ollama_base_url() everywhere instead of a literal http://localhost:11434.
"""

import os

_DEFAULT = "http://localhost:11434"


def ollama_base_url() -> str:
    host = (os.environ.get("OLLAMA_HOST") or "").strip()
    if not host:
        return _DEFAULT
    if host.startswith(("http://", "https://")):
        return host.rstrip("/")
    return "http://" + host.rstrip("/")
