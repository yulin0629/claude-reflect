#!/usr/bin/env python3
"""Benchmark embedding classifier: latency, model size, and accuracy vs regex.

Usage:
    python scripts/benchmark_embedding.py

Requires model to be downloaded first:
    python scripts/download_model.py
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
from lib.embedding_classifier import (
    EmbeddingModel,
    AnchorStore,
    classify_message,
    DEFAULT_MODEL_DIR,
)
from lib.reflect_utils import detect_patterns


# Test cases: (text, expected_type, description)
TEST_CASES = [
    # Corrections (English)
    ("no, use gpt-5.1 not gpt-5", "auto", "EN correction: use X not Y"),
    ("don't use that library, use lodash instead", "auto", "EN correction: don't use"),
    ("that's wrong, the correct approach is to use async/await", "auto", "EN correction: that's wrong"),
    ("stop using var, always use const or let", "auto", "EN correction: stop using"),
    ("actually, you should import from @utils not ./utils", "auto", "EN correction: actually"),
    ("I told you to use TypeScript, not JavaScript", "auto", "EN correction: I told you"),

    # Corrections (zh-TW)
    ("不對，要用這個不是那個", "auto", "ZH correction: 不對"),
    ("不要用那個，改用這個", "auto", "ZH correction: 不要用"),
    ("錯了，正確的做法是用 async", "auto", "ZH correction: 錯了"),

    # Corrections (ja)
    ("それは違います、こちらを使ってください", "auto", "JA correction"),

    # Corrections (ko)
    ("아니요, 그거 말고 이걸 사용하세요", "auto", "KO correction"),

    # Guardrails
    ("don't add features unless I ask for them", "guardrail", "EN guardrail: don't unless"),
    ("only change what I specifically requested", "guardrail", "EN guardrail: only what asked"),
    ("不要擅自加功能，除非我要求", "guardrail", "ZH guardrail"),

    # Positive feedback
    ("perfect, that's exactly what I wanted!", "positive", "EN positive: perfect"),
    ("great job, keep doing it this way", "positive", "EN positive: great job"),
    ("太好了，就是這樣！", "positive", "ZH positive"),

    # Not learning (should return None)
    ("help me fix this bug in the login page", None, "EN task request"),
    ("can you review this pull request?", None, "EN question"),
    ("please implement user authentication", None, "EN feature request"),
    ("what does this error mean?", None, "EN question about error"),
    ("幫我修這個 bug", None, "ZH task request"),
    ("請幫我寫登入功能", None, "ZH feature request"),
    ("ok let's continue with the next step", None, "EN continuation"),
    ("run the tests", None, "EN command"),
]


def format_time(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds*1_000_000:.0f} us"
    elif seconds < 1.0:
        return f"{seconds*1000:.1f} ms"
    else:
        return f"{seconds:.2f} s"


def main() -> int:
    model_dir = DEFAULT_MODEL_DIR
    custom = os.environ.get("CLAUDE_REFLECT_MODEL_DIR")
    if custom:
        model_dir = Path(custom)

    int8_path = model_dir / "model_int8.onnx"
    if not int8_path.exists():
        print(f"Model not found at {int8_path}")
        print("Run: python scripts/download_model.py")
        return 1

    # 1. Model size
    model_size_mb = int8_path.stat().st_size / (1024 * 1024)
    print(f"{'='*60}")
    print(f"Embedding Classifier Benchmark")
    print(f"{'='*60}")
    print(f"\nModel: {int8_path}")
    print(f"Model size: {model_size_mb:.1f} MB")

    # 2. Model load time
    print(f"\n--- Latency ---")
    model = EmbeddingModel(model_dir)

    t0 = time.perf_counter()
    model.load()
    load_time = time.perf_counter() - t0
    print(f"Model load:      {format_time(load_time)}")

    # 3. Anchor computation time
    anchor_store = AnchorStore(model)
    t0 = time.perf_counter()
    anchor_store.compute()
    anchor_time = time.perf_counter() - t0
    print(f"Anchor compute:  {format_time(anchor_time)}")

    # 4. Single inference time (warm-up + measure)
    _ = classify_message("warm up test", model, anchor_store)

    times = []
    for text, _, _ in TEST_CASES:
        t0 = time.perf_counter()
        classify_message(text, model, anchor_store)
        times.append(time.perf_counter() - t0)

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    p95_time = sorted(times)[int(len(times) * 0.95)]
    print(f"Inference avg:   {format_time(avg_time)}")
    print(f"Inference min:   {format_time(min_time)}")
    print(f"Inference max:   {format_time(max_time)}")
    print(f"Inference p95:   {format_time(p95_time)}")

    # 5. Accuracy comparison: embedding vs regex
    print(f"\n--- Accuracy: Embedding vs Regex ---")
    print(f"{'Description':<40} {'Expected':<12} {'Embed':<12} {'Regex':<12} {'E':>2} {'R':>2}")
    print(f"{'-'*40} {'-'*12} {'-'*12} {'-'*12} {'--':>2} {'--':>2}")

    embed_correct = 0
    regex_correct = 0
    total = len(TEST_CASES)

    for text, expected_type, desc in TEST_CASES:
        # Embedding result
        e_type, _, e_conf, _, _ = classify_message(text, model, anchor_store)
        # Regex result
        r_type, _, r_conf, _, _ = detect_patterns(text)

        e_match = e_type == expected_type
        r_match = r_type == expected_type
        if e_match:
            embed_correct += 1
        if r_match:
            regex_correct += 1

        e_str = f"{e_type or 'None'}"
        r_str = f"{r_type or 'None'}"
        e_mark = "OK" if e_match else "X"
        r_mark = "OK" if r_match else "X"

        print(f"{desc:<40} {str(expected_type):<12} {e_str:<12} {r_str:<12} {e_mark:>2} {r_mark:>2}")

    print(f"\n--- Summary ---")
    print(f"Embedding accuracy: {embed_correct}/{total} ({embed_correct/total*100:.0f}%)")
    print(f"Regex accuracy:     {regex_correct}/{total} ({regex_correct/total*100:.0f}%)")
    print(f"Model size:         {model_size_mb:.1f} MB (target: < 200 MB)")
    print(f"Model load:         {format_time(load_time)} (one-time, in daemon)")
    print(f"Inference avg:      {format_time(avg_time)} (target: < 50 ms)")

    # 6. POC checklist
    print(f"\n--- POC Checklist ---")
    checks = [
        (model_size_mb < 200, f"Model size < 200 MB: {model_size_mb:.1f} MB"),
        (avg_time < 0.050, f"Inference < 50 ms: {format_time(avg_time)}"),
        (embed_correct / total >= 0.80 * (regex_correct / total) if regex_correct > 0 else embed_correct > 0,
         f"Accuracy >= 80% of regex: {embed_correct/total*100:.0f}% vs {regex_correct/total*100:.0f}%"),
    ]

    all_pass = True
    for passed, desc in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {desc}")

    if all_pass:
        print(f"\nAll POC checks passed. Proceed to Phase 2.")
    else:
        print(f"\nSome POC checks failed. Review before proceeding.")

    return 0 if all_pass else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
