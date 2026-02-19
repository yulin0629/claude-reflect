#!/usr/bin/env python3
"""Local ONNX embedding classifier for multilingual correction detection.

Uses intfloat/multilingual-e5-small (INT8 quantized) to classify user messages
by cosine similarity against anchor embeddings.

Designed to run inside a persistent daemon (embedding_server.py) to avoid
repeated model loading (~500ms). When called directly, loads model on each call.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# Default model directory
DEFAULT_MODEL_DIR = Path.home() / ".claude" / "models" / "multilingual-e5-small"

# E5 models require "query: " prefix for queries
E5_QUERY_PREFIX = "query: "

# Classification thresholds
CORRECTION_THRESHOLD = 0.45
GUARDRAIL_THRESHOLD = 0.50
POSITIVE_THRESHOLD = 0.50
NOT_LEARNING_THRESHOLD = 0.40

# Anchor definitions file
ANCHORS_FILE = Path(__file__).parent / "anchors.json"


class EmbeddingModel:
    """ONNX-based embedding model with tokenizer."""

    def __init__(self, model_dir: Optional[Path] = None):
        custom = os.environ.get("CLAUDE_REFLECT_MODEL_DIR")
        if custom:
            self.model_dir = Path(custom)
        elif model_dir:
            self.model_dir = model_dir
        else:
            self.model_dir = DEFAULT_MODEL_DIR

        self._session = None
        self._tokenizer = None

    def load(self) -> None:
        """Load ONNX model and tokenizer. Call once at startup."""
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_path = self.model_dir / "model_int8.onnx"
        tokenizer_path = self.model_dir / "tokenizer.json"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                f"Run: python scripts/download_model.py"
            )

        # Use CPU-only for consistent behavior
        self._session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )

        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        # Truncate at max length but do NOT pad to fixed length.
        # We handle dynamic padding per-input for efficiency.
        self._tokenizer.enable_truncation(max_length=512)
        self._tokenizer.no_padding()

    @property
    def is_loaded(self) -> bool:
        return self._session is not None and self._tokenizer is not None

    def embed(self, text: str) -> np.ndarray:
        """Compute embedding for a single text.

        E5 models require "query: " prefix for input text.
        Returns L2-normalized embedding vector.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        prefixed = E5_QUERY_PREFIX + text
        encoded = self._tokenizer.encode(prefixed)

        # Dynamic length — no padding to 512. Only use actual token count.
        seq_len = len(encoded.ids)
        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.ones((1, seq_len), dtype=np.int64)
        token_type_ids = np.zeros((1, seq_len), dtype=np.int64)

        outputs = self._session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )

        # outputs[0] shape: (1, seq_len, hidden_dim)
        # Mean pooling over non-padding tokens
        token_embeddings = outputs[0]  # (1, seq_len, hidden_dim)
        mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
        summed = np.sum(token_embeddings * mask_expanded, axis=1)
        counts = np.sum(mask_expanded, axis=1)
        mean_pooled = summed / np.maximum(counts, 1e-9)

        # L2 normalize
        norm = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
        normalized = mean_pooled / np.maximum(norm, 1e-9)

        return normalized[0]  # (hidden_dim,)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Compute embeddings for multiple texts. Returns (N, hidden_dim) array."""
        if not texts:
            return np.array([])
        return np.array([self.embed(t) for t in texts])


class AnchorStore:
    """Pre-computed anchor embeddings for classification."""

    def __init__(self, model: EmbeddingModel, anchors_path: Optional[Path] = None):
        self.model = model
        self.anchors_path = anchors_path or ANCHORS_FILE
        self.category_embeddings: Dict[str, np.ndarray] = {}

    def compute(self) -> None:
        """Compute anchor embeddings for all categories."""
        with open(self.anchors_path, "r", encoding="utf-8") as f:
            anchors = json.load(f)

        for category, sentences in anchors.items():
            embeddings = self.model.embed_batch(sentences)
            # Store mean of all anchor embeddings per category
            self.category_embeddings[category] = np.mean(embeddings, axis=0)

    @property
    def is_computed(self) -> bool:
        return len(self.category_embeddings) > 0


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors. Both should be L2-normalized."""
    return float(np.dot(a, b))


def classify_message(
    text: str,
    model: EmbeddingModel,
    anchor_store: AnchorStore,
) -> Tuple[Optional[str], str, float, str, int]:
    """Classify a message using embedding similarity.

    Returns:
        Tuple matching detect_patterns() signature:
        (type, matched_patterns, confidence, sentiment, decay_days)
    """
    text_embedding = model.embed(text)

    # Compute similarities to each category
    similarities: Dict[str, float] = {}
    for category, anchor_emb in anchor_store.category_embeddings.items():
        similarities[category] = cosine_similarity(text_embedding, anchor_emb)

    # Find the best matching category
    best_category = max(similarities, key=similarities.get)
    best_score = similarities[best_category]

    # not_learning is the "reject" class
    not_learning_score = similarities.get("not_learning", 0.0)

    # Decision logic with thresholds
    if best_category == "not_learning":
        return (None, "", 0.0, "correction", 90)

    # Check if the best category beats not_learning by enough margin
    margin = best_score - not_learning_score

    if best_category == "correction":
        if best_score < CORRECTION_THRESHOLD or margin < 0.02:
            return (None, "", 0.0, "correction", 90)
        confidence = _score_to_confidence(best_score, 0.60, 0.85)
        return ("auto", f"embedding:{best_category}", confidence, "correction", 90)

    elif best_category == "guardrail":
        if best_score < GUARDRAIL_THRESHOLD or margin < 0.02:
            return (None, "", 0.0, "correction", 90)
        confidence = _score_to_confidence(best_score, 0.75, 0.90)
        return ("guardrail", f"embedding:{best_category}", confidence, "correction", 120)

    elif best_category == "positive":
        if best_score < POSITIVE_THRESHOLD or margin < 0.02:
            return (None, "", 0.0, "correction", 90)
        confidence = _score_to_confidence(best_score, 0.65, 0.80)
        return ("positive", f"embedding:{best_category}", confidence, "positive", 90)

    return (None, "", 0.0, "correction", 90)


def _score_to_confidence(score: float, min_conf: float, max_conf: float) -> float:
    """Map a similarity score to a confidence range."""
    # Clamp score to [0, 1]
    score = max(0.0, min(1.0, score))
    # Linear mapping
    return min_conf + (max_conf - min_conf) * score
