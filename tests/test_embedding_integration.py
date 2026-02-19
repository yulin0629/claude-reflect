"""Integration tests for ONNX embedding classification with real model.

Validates multilingual correction detection accuracy using the actual
multilingual-e5-small INT8 model. Skipped automatically when model files
or dependencies are unavailable.

These tests guard against regressions in:
- Anchor quality (anchors.json)
- Classification logic (max-similarity, thresholds, margin)
- Cross-language detection (zh, ja, ko, fr, etc.)
"""
import sys
import os
from pathlib import Path

import pytest

# Ensure scripts/lib is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "lib"))

# --- Skip conditions ---
try:
    import onnxruntime  # noqa: F401
    import tokenizers  # noqa: F401
    import numpy  # noqa: F401

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

MODEL_DIR = Path.home() / ".claude" / "models" / "multilingual-e5-small"
HAS_MODEL = (MODEL_DIR / "model_int8.onnx").exists() and (MODEL_DIR / "tokenizer.json").exists()

pytestmark = pytest.mark.skipif(
    not (HAS_DEPS and HAS_MODEL),
    reason="ONNX model or dependencies not available",
)


@pytest.fixture(scope="module")
def classifier():
    """Load real model + compute anchor embeddings once per module."""
    from lib.embedding_classifier import EmbeddingModel, AnchorStore

    model = EmbeddingModel()
    model.load()
    anchor_store = AnchorStore(model)
    anchor_store.compute()
    return model, anchor_store


# =============================================================================
# Corrections: must be detected (type is not None)
# =============================================================================

CORRECTION_CASES = [
    # English
    ("no, use pytest not unittest", "en"),
    ("that's wrong, it should be async", "en"),
    ("wrong, change it back to the original", "en"),
    # Chinese (zh-TW)
    ("不對，用 fastapi 不要用 flask", "zh"),
    ("錯了，應該用 async 不是 sync", "zh"),
    ("不要用 var，改用 const", "zh"),
    # Japanese
    ("それは違う、pytestを使って", "ja"),
    ("違います、flaskじゃなくてfastapiを使ってください", "ja"),
    # Korean
    ("아니요, pytest를 사용하세요, unittest 말고", "ko"),
    # French
    ("non, utilise pytest pas unittest", "fr"),
]


class TestCorrectionsDetected:
    """Corrections across languages must be captured (type is not None)."""

    @pytest.mark.parametrize(
        "text, lang",
        CORRECTION_CASES,
        ids=[f"{lang}:{text[:30]}" for text, lang in CORRECTION_CASES],
    )
    def test_correction_captured(self, classifier, text, lang):
        from lib.embedding_classifier import classify_message

        model, anchor_store = classifier
        result = classify_message(text, model, anchor_store)
        assert result[0] is not None, (
            f"[{lang}] Expected detection but got None for: {text}"
        )


# =============================================================================
# Guardrails: must be detected as guardrail specifically
# =============================================================================

GUARDRAIL_CASES = [
    ("don't add features unless I ask for them", "en"),
    ("不要擅自加功能", "zh"),
    ("only change what I specifically requested", "en"),
]


class TestGuardrailsDetected:
    @pytest.mark.parametrize(
        "text, lang",
        GUARDRAIL_CASES,
        ids=[f"{lang}:{text[:30]}" for text, lang in GUARDRAIL_CASES],
    )
    def test_guardrail_captured(self, classifier, text, lang):
        from lib.embedding_classifier import classify_message

        model, anchor_store = classifier
        result = classify_message(text, model, anchor_store)
        assert result[0] == "guardrail", (
            f"[{lang}] Expected guardrail but got {result[0]} for: {text}"
        )


# =============================================================================
# Not-learning: must be rejected (type is None)
# =============================================================================

NOT_LEARNING_CASES = [
    ("help me write a sort function", "en"),
    ("can you review this code", "en"),
    ("幫我寫一個排序函式", "zh"),
    ("幫我跑測試", "zh"),
    ("次のステップに進みましょう", "ja"),
    ("テストを実行してください", "ja"),
    ("이 버그를 수정해 주세요", "ko"),
    ("ok let's continue with the next step", "en"),
]


class TestNotLearningRejected:
    """General requests and task messages must NOT be captured."""

    @pytest.mark.parametrize(
        "text, lang",
        NOT_LEARNING_CASES,
        ids=[f"{lang}:{text[:30]}" for text, lang in NOT_LEARNING_CASES],
    )
    def test_not_learning_rejected(self, classifier, text, lang):
        from lib.embedding_classifier import classify_message

        model, anchor_store = classifier
        result = classify_message(text, model, anchor_store)
        assert result[0] is None, (
            f"[{lang}] Expected None (reject) but got {result[0]} for: {text}"
        )


# =============================================================================
# Result format: always a 5-tuple with correct types
# =============================================================================

class TestResultFormat:
    def test_detected_result_format(self, classifier):
        from lib.embedding_classifier import classify_message

        model, anchor_store = classifier
        result = classify_message("no, use this instead", model, anchor_store)
        assert len(result) == 5
        assert isinstance(result[0], (str, type(None)))
        assert isinstance(result[1], str)
        assert isinstance(result[2], float)
        assert isinstance(result[3], str)
        assert isinstance(result[4], int)

    def test_rejected_result_format(self, classifier):
        from lib.embedding_classifier import classify_message

        model, anchor_store = classifier
        result = classify_message("help me fix this bug", model, anchor_store)
        assert result == (None, "", 0.0, "correction", 90)
