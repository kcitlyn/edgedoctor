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
