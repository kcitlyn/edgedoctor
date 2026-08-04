"""Tests for the Backend interface and its implementations.

Pins two things: (1) the abstract interface can't be instantiated bare, and
(2) stubs stay honest — they raise NotImplementedError rather than returning
fake results.
"""

from pathlib import Path

import pytest

from edgedoctor.backends.base import Backend
from edgedoctor.backends.tensorrt import TensorRTBackend


class TestBackendInterface:
    def test_base_is_abstract(self):
        with pytest.raises(TypeError):
            Backend()  # abstract methods convert/parse are unimplemented

    def test_tensorrt_is_a_backend(self):
        assert issubclass(TensorRTBackend, Backend)
        assert TensorRTBackend.name == "tensorrt"


class TestTensorRTStub:
    def test_convert_is_honest(self):
        # convert() still needs the NVIDIA machine — must stay an honest stub.
        with pytest.raises(NotImplementedError):
            TensorRTBackend().convert(Path("model.onnx"))
    # parse() is now real — covered by tests/test_tensorrt_parser.py.


class TestBackendAbcContract:
    """The ABC must enforce everything the rest of the tool depends on.

    A method that every parser happens to implement but the ABC doesn't require
    is a latent gap: a new backend could omit it, and the failure would surface
    as an AttributeError deep in a test rather than as a clear contract error at
    construction.
    """

    def test_abstract_methods_are_the_expected_set(self):
        from edgedoctor.backends.base import Backend

        abstract = {
            name for name, value in vars(Backend).items()
            if getattr(value, "__isabstractmethod__", False)
        }
        assert abstract == {"convert", "parse", "parse_text"}

    def test_parse_text_is_required(self):
        # The robustness suite drives every parser through parse_text, so the
        # ABC must guarantee it exists.
        from pathlib import Path

        from edgedoctor.backends.base import Backend, Facts

        class MissingParseText(Backend):
            name = "missing"

            def convert(self, model_path: Path, **options):
                raise NotImplementedError

            def parse(self, artifact_path: Path) -> Facts:
                return Facts(backend=self.name, artifact_path="x")

        with pytest.raises(TypeError, match="abstract"):
            MissingParseText()

    def test_parse_is_required(self):
        from pathlib import Path

        from edgedoctor.backends.base import Backend, Facts

        class MissingParse(Backend):
            name = "missing"

            def convert(self, model_path: Path, **options):
                raise NotImplementedError

            def parse_text(self, text: str, artifact_name: str = "<string>") -> Facts:
                return Facts(backend=self.name, artifact_path=artifact_name)

        with pytest.raises(TypeError, match="abstract"):
            MissingParse()

    def test_every_registered_backend_satisfies_the_abc(self):
        from edgedoctor.backends import PARSER_REGISTRY, get_parser
        from edgedoctor.backends.base import Backend

        for name in PARSER_REGISTRY:
            parser = get_parser(name)
            assert isinstance(parser, Backend)
            assert parser.name == name, "registry key must match Backend.name"

    def test_parse_text_signature_is_consistent(self):
        # Callers pass artifact_name positionally in places, so the order is
        # part of the contract.
        import inspect

        from edgedoctor.backends import PARSER_REGISTRY, get_parser

        for name in PARSER_REGISTRY:
            params = list(inspect.signature(get_parser(name).parse_text).parameters)
            assert params[:2] == ["text", "artifact_name"], f"{name}: {params}"


class TestParsersHoldNoState:
    """A parser instance must be reusable and safe to share.

    get_parser() returns a fresh instance today, but the parsers are documented
    as pure, and a library consumer may reasonably hold one. Block-structured
    parsers track state DURING a parse (the current tensor or provider), so
    leaking it between parses is a realistic bug.
    """

    def test_reuse_after_a_different_artifact_is_identical(self):
        from pathlib import Path

        from edgedoctor.backends import get_parser

        cases = [
            ("onnxruntime", "corpus/onnxruntime/ort_partial_fallback.log",
             "corpus/onnxruntime/ort_cpu_only.log"),
            ("polygraphy", "corpus/onnxruntime/int8_vs_fp32_fail.log",
             "corpus/onnxruntime/fp32_baseline_run.log"),
        ]
        for backend, first, second in cases:
            parser = get_parser(backend)
            baseline = parser.parse(Path(first)).model_dump()
            parser.parse(Path(second))  # a different artifact in between
            assert parser.parse(Path(first)).model_dump() == baseline, (
                f"{backend} leaked state between parses"
            )

    def test_concurrent_parsing_is_consistent(self):
        # Block-state kept on `self` instead of in a local would corrupt here.
        import queue
        import threading
        from pathlib import Path

        from edgedoctor.backends import get_parser
        from edgedoctor.diagnoser import diagnose

        log = Path("corpus/onnxruntime/ort_partial_fallback.log")
        results: queue.Queue = queue.Queue()
        errors: queue.Queue = queue.Queue()

        def worker():
            try:
                for _ in range(10):
                    facts = get_parser("onnxruntime").parse(log)
                    results.put(tuple(sorted(d.code for d in diagnose(facts))))
            except Exception as exc:  # pragma: no cover - failure path
                errors.put(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors.empty(), f"concurrent parsing raised: {list(errors.queue)}"
        distinct = set()
        while not results.empty():
            distinct.add(results.get())
        assert len(distinct) == 1, f"inconsistent results under concurrency: {distinct}"
