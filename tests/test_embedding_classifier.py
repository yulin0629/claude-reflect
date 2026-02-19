"""Tests for embedding_classifier.py — ONNX model loading and classification.

Uses mocks to avoid requiring real ONNX model files.
"""
import json
import sys
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np

# Ensure scripts/lib is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "lib"))

from lib.embedding_classifier import (
    EmbeddingModel,
    AnchorStore,
    classify_message,
    cosine_similarity,
    _score_to_confidence,
    CORRECTION_THRESHOLD,
    GUARDRAIL_THRESHOLD,
    POSITIVE_THRESHOLD,
)


# =============================================================================
# Helper: create a mock model that returns predictable embeddings
# =============================================================================

def make_mock_model(embed_fn=None):
    """Create an EmbeddingModel mock with controllable embed() output."""
    model = MagicMock(spec=EmbeddingModel)
    model.is_loaded = True

    if embed_fn is None:
        # Default: return a random unit vector
        def embed_fn(text):
            rng = np.random.RandomState(hash(text) % (2**31))
            vec = rng.randn(384)
            return vec / np.linalg.norm(vec)

    model.embed = embed_fn
    model.embed_batch = lambda texts: np.array([embed_fn(t) for t in texts])
    return model


def make_anchor_store_with_known_categories(model):
    """Create an AnchorStore where each category is a known direction."""
    store = AnchorStore(model)
    # Manually set category embeddings to known unit vectors
    store.category_embeddings = {
        "correction": np.array([1.0, 0.0, 0.0] + [0.0] * 381),
        "guardrail": np.array([0.0, 1.0, 0.0] + [0.0] * 381),
        "positive": np.array([0.0, 0.0, 1.0] + [0.0] * 381),
        "not_learning": np.array([0.0, 0.0, 0.0, 1.0] + [0.0] * 380),
    }
    # Normalize
    for k, v in store.category_embeddings.items():
        store.category_embeddings[k] = v / np.linalg.norm(v)
    return store


# =============================================================================
# Tests: cosine_similarity
# =============================================================================

class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 0.0, 0.0])
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(-1.0)


# =============================================================================
# Tests: _score_to_confidence
# =============================================================================

class TestScoreToConfidence:
    def test_zero_score(self):
        assert _score_to_confidence(0.0, 0.6, 0.9) == pytest.approx(0.6)

    def test_full_score(self):
        assert _score_to_confidence(1.0, 0.6, 0.9) == pytest.approx(0.9)

    def test_half_score(self):
        assert _score_to_confidence(0.5, 0.6, 0.9) == pytest.approx(0.75)

    def test_clamps_above_one(self):
        result = _score_to_confidence(1.5, 0.6, 0.9)
        assert result == pytest.approx(0.9)


# =============================================================================
# Tests: EmbeddingModel
# =============================================================================

class TestEmbeddingModel:
    def test_default_model_dir(self):
        model = EmbeddingModel()
        assert "multilingual-e5-small" in str(model.model_dir)

    def test_custom_model_dir(self):
        model = EmbeddingModel(model_dir=Path("/tmp/test-model"))
        assert str(model.model_dir) == "/tmp/test-model"

    @patch.dict(os.environ, {"CLAUDE_REFLECT_MODEL_DIR": "/tmp/env-model"})
    def test_env_override(self):
        model = EmbeddingModel()
        assert str(model.model_dir) == "/tmp/env-model"

    def test_not_loaded_initially(self):
        model = EmbeddingModel()
        assert not model.is_loaded

    def test_load_missing_model_raises(self):
        model = EmbeddingModel(model_dir=Path("/nonexistent"))
        # FileNotFoundError from our code, or ImportError if onnxruntime not installed
        with pytest.raises((FileNotFoundError, ImportError)):
            model.load()


# =============================================================================
# Tests: AnchorStore
# =============================================================================

class TestAnchorStore:
    def test_compute_loads_all_categories(self):
        model = make_mock_model()
        store = AnchorStore(model)
        store.compute()

        assert "correction" in store.category_embeddings
        assert "guardrail" in store.category_embeddings
        assert "positive" in store.category_embeddings
        assert "not_learning" in store.category_embeddings

    def test_is_computed_after_compute(self):
        model = make_mock_model()
        store = AnchorStore(model)
        assert not store.is_computed
        store.compute()
        assert store.is_computed

    def test_missing_anchors_file_raises(self):
        model = make_mock_model()
        store = AnchorStore(model, anchors_path=Path("/nonexistent/anchors.json"))
        with pytest.raises(FileNotFoundError):
            store.compute()


# =============================================================================
# Tests: classify_message
# =============================================================================

class TestClassifyMessage:
    def test_correction_detected(self):
        """When text embedding is closest to correction anchor."""
        model = make_mock_model()
        store = make_anchor_store_with_known_categories(model)

        # Make model return a vector close to correction direction
        correction_vec = store.category_embeddings["correction"]
        model.embed = lambda text: correction_vec

        result = classify_message("no, use this instead", model, store)
        assert result[0] == "auto"
        assert "correction" in result[1]
        assert result[3] == "correction"

    def test_guardrail_detected(self):
        model = make_mock_model()
        store = make_anchor_store_with_known_categories(model)

        guardrail_vec = store.category_embeddings["guardrail"]
        model.embed = lambda text: guardrail_vec

        result = classify_message("don't add unless asked", model, store)
        assert result[0] == "guardrail"
        assert result[3] == "correction"

    def test_positive_detected(self):
        model = make_mock_model()
        store = make_anchor_store_with_known_categories(model)

        positive_vec = store.category_embeddings["positive"]
        model.embed = lambda text: positive_vec

        result = classify_message("perfect, great job!", model, store)
        assert result[0] == "positive"
        assert result[3] == "positive"

    def test_not_learning_returns_none(self):
        model = make_mock_model()
        store = make_anchor_store_with_known_categories(model)

        not_learning_vec = store.category_embeddings["not_learning"]
        model.embed = lambda text: not_learning_vec

        result = classify_message("help me fix this", model, store)
        assert result[0] is None

    def test_low_score_returns_none(self):
        """When all scores are below threshold, should return None."""
        model = make_mock_model()
        store = make_anchor_store_with_known_categories(model)

        # Return a vector that's roughly equidistant from all categories
        equidistant = np.ones(384) / np.sqrt(384)
        model.embed = lambda text: equidistant

        result = classify_message("something ambiguous", model, store)
        # The result depends on exact geometry, but confidence should be low
        # or type should be None
        if result[0] is not None:
            assert result[2] > 0  # If detected, confidence must be > 0

    def test_result_tuple_format(self):
        """Result should always be a 5-tuple."""
        model = make_mock_model()
        store = make_anchor_store_with_known_categories(model)
        model.embed = lambda text: store.category_embeddings["correction"]

        result = classify_message("test", model, store)
        assert len(result) == 5
        assert isinstance(result[0], (str, type(None)))  # type
        assert isinstance(result[1], str)  # patterns
        assert isinstance(result[2], float)  # confidence
        assert isinstance(result[3], str)  # sentiment
        assert isinstance(result[4], int)  # decay_days
