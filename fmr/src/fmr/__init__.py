"""FMR — Faithful Medical Reasoning.

A training-free faithfulness verification & abstention layer for medical VLMs,
with an optional learned faithfulness verifier and faithfulness-LoRA.

The package is model-agnostic. The core pipeline runs fully offline with the
built-in ``MockVLM`` (no GPU, no downloads) so the whole system can be exercised
end-to-end; real Hugging Face VLMs plug in behind the same ``BaseVLM`` interface.
"""

__version__ = "0.1.0"

from .types import Sample, Step, VLMOutput  # noqa: E402
from .data.regions import Region  # noqa: E402

__all__ = ["Sample", "Step", "VLMOutput", "Region", "__version__"]
