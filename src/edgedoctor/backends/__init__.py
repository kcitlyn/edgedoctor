"""Per-backend modules.

Each backend (tensorrt, polygraphy, onnxruntime, coreml, …) implements the
`Backend` interface from `base.py`. See docs/DESIGN.md.

`get_parser()` below is the registry the CLI uses, so adding a backend means
adding one entry here plus its module and rule file — never editing the CLI.
That's the expansion seam working as designed.

Imports are done lazily inside the function: backends may eventually pull in
heavy vendor SDKs, and `import edgedoctor` must stay fast and GPU-free.
"""

from __future__ import annotations

from .base import Backend

#: Backend name → "module:class" for every backend with a working parser.
#: Names here are exactly the values accepted by the CLI's --backend flag.
PARSER_REGISTRY: dict[str, str] = {
    "tensorrt": "tensorrt:TensorRTBackend",
    "polygraphy": "polygraphy:PolygraphyBackend",
    "onnxruntime": "onnxruntime:OnnxRuntimeBackend",
    "raspberrypi": "raspberrypi:RaspberryPiBackend",
}


def get_parser(name: str) -> Backend:
    """Instantiate the backend whose parser handles `name`.

    Raises KeyError if the backend has no parser yet — callers turn that into a
    clean "not implemented, see ROADMAP" message rather than a traceback.
    """
    if name not in PARSER_REGISTRY:
        raise KeyError(name)
    module_name, class_name = PARSER_REGISTRY[name].split(":")
    from importlib import import_module

    module = import_module(f".{module_name}", package=__name__)
    backend_cls = getattr(module, class_name)
    return backend_cls()
