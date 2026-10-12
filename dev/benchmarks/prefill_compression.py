#!/usr/bin/env python3
"""Compare full and context-compressed prompts against a running Splash server.

This is a client-side experiment, not a change to Splash's Metal kernels or model.
Only the Python standard library is required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

WORD = re.compile(r"[\w'-]+", re.UNICODE)
STOP = set("""about after again all also and any are because been before being
between both but can could did does doing down each few for from further had has
have having here how into its just more most not now only other our out over same
should some such than that the their them then there these they this those through
too under until very was were what when where which while who will with would you
your""".split())
GAP = "[... omitted context ...]"


def words(text: str) -> list[str]:
    return [w.casefold() for w in WORD.findall(text)
            if len(w) > 2 and w.casefold() not in STOP]


def chunks(text: str, width: int = 650) -> list[str]:
    """Split paragraphs near sentence boundaries; hard-split unusually long lines."""
    result = []
    for paragraph in re.split(r"\n\s*\n", text.strip()):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        sentences = re.split(r"(?<=[.!?。！？])\s+", paragraph)
        current = ""
        for sentence in sentences:
            sentence = sentence.strip()
            while len(sentence) > width:
                if current:
                    result.append(current)
                    current = ""
                cut = sentence.rfind(" ", 0, width + 1)
                cut = cut if cut >= width // 2 else width
                result.append(sentence[:cut].strip())
                sentence = sentence[cut:].strip()
            if not sentence:
                continue
            candidate = (current + " " + sentence).strip()
            if current and len(candidate) > width:
                result.append(current)
                current = sentence
            else:
                current = candidate
        if current:
            result.append(current)
    return result or ([text] if text else [])


def extractive(context: str, question: str, ratio: float) -> str:
    """Select query-relevant verbatim chunks, retaining their source order."""
    if not 0 < ratio < 1:
        raise ValueError("ratio must be strictly between 0 and 1")
    parts = chunks(context)
    if len(parts) < 2:
        return context
    query = set(words(question))
    counts: dict[str, int] = {}
    part_words = [words(p) for p in parts]
    for tokens in part_words:
        for word in set(tokens):
            counts[word] = counts.get(word, 0) + 1
    ranked = []
    for i, tokens in enumerate(part_words):
        score = sum(
            min(tokens.count(word), 3) * (1 + len(parts) / (1 + counts[word]))
            for word in query if word in tokens
        )
        ranked.append((score, i))
    budget = max(1, int(len(context) * ratio))
    chosen = {0}  # retain the opening context/instructions
    used = len(parts[0])
    for _, i in sorted(ranked, key=lambda item: (-item[0], item[1])):
        if i in chosen:
            continue
        if used + len(parts[i]) <= budget:
            chosen.add(i)
            used += len(parts[i])
    if len(chosen) == 1 and len(parts[0]) > budget:
        best = max(ranked, key=lambda item: (item[0], -item[1]))[1]
        chosen = {best}
    output = []
    previous = -2
    for i in sorted(chosen):
        if output and i != previous + 1:
            output.append(GAP)
        output.append(parts[i])
        previous = i
    return "\n\n".join(output)


def request_json(url: str, payload: dict | None, timeout: float,
                 api_key: str | None = None) -> tuple[dict, float]:
    headers = {"Accept": "application/json"}
    data = None
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode()
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=data, headers=headers,
                                 method="POST" if payload is not None else "GET")
    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail[:1000]}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"request to {url} failed: {exc}") from exc
    elapsed = (time.monotonic() - start) * 1000
    try:
        return json.loads(raw), elapsed
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{url} returned invalid JSON") from exc


def chat(url: str, model: str, messages: list[dict], max_tokens: int,
         timeout: float, api_key: str | None) -> tuple[dict, float]:
    return request_json(url, {
        "model": model, "messages": messages,
        "max_completion_tokens": max_tokens, "temperature": 0, "stream": False,
    }, timeout, api_key)


def message_text(response: dict) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"chat endpoint did not return a message: {response!r}") from exc
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("chat endpoint returned empty/non-text content")
    return content.strip()


def status_url(chat_url: str) -> str | None:
    suffix = "/v1/chat/completions"
    return chat_url[:-len(suffix)] + "/status" if chat_url.endswith(suffix) else None


def target_request(args, content: str, variant: str, sample: int) -> dict:
    nonce = f"{sample}-{variant}-{uuid.uuid4().hex}"
    prompt = (
        f"Prefill experiment id: {nonce}\n\nQuestion:\n{args.question}\n\n"
        "Use the supplied context as the source of truth. If it does not contain "
        "the answer, say so instead of guessing.\n\n"
        f"Context:\n{content}\n\nAnswer directly and concisely."
    )
    key = os.environ.get(args.target_key_env) if args.target_key_env else None
    stat_url = args.status_url or status_url(args.url)
    before = None
    if stat_url:
        before, _ = request_json(stat_url, None, min(args.timeout, 30), key)
    response, wall_ms = chat(
        args.url, args.model, [{"role": "user", "content": prompt}],
        args.max_output_tokens, args.timeout, key,
    )
    after = None
    if stat_url:
        after, _ = request_json(stat_url, None, min(args.timeout, 30), key)
    metrics = response.get("metrics", {})
    usage = response.get("usage", {})
    latency = metrics.get("request_latency", {})
    prefill = metrics.get("prefill", {})
    cache = metrics.get("cache", {})
    prefill_wall = None
    if isinstance(before, dict) and isinstance(after, dict):
        a = before.get("metrics", {}).get("prefill_wall_ms")
        b = after.get("metrics", {}).get("prefill_wall_ms")
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            prefill_wall = max(0.0, b - a)
    prefill_tokens = prefill.get("tokens")
    prefill_rate = prefill_tokens * 1000 / prefill_wall if prefill_tokens and prefill_wall else None
    return {
        "variant": variant, "sample": sample,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "context_chars": len(content),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cached_tokens": cache.get("matched_tokens"),
        "target_http_wall_ms": round(wall_ms, 3),
        "target_ttft_ms": latency.get("ttft_ms"),
        "target_prefill_wall_ms": round(prefill_wall, 3) if prefill_wall is not None else None,
        "target_prefill_tokens": prefill_tokens,
        "target_prefill_tokens_per_second": round(prefill_rate, 3) if prefill_rate is not None else None,
        "answer": message_text(response),
    }


def compress_with_model(args, context: str) -> tuple[str, float, dict]:
    if not args.compressor_url or not args.compressor_model:
        raise ValueError("--mode model requires --compressor-url and --compressor-model")
    prompt = (
        "Select the most useful excerpts for answering the question. Output verbatim "
        "excerpts only; preserve names, numbers, code, dates and qualifiers. Do not "
        "answer, summarize, invent facts, or rewrite excerpts. Put an omission marker "
        "between non-adjacent excerpts. Keep it compact.\n\n"
        f"QUESTION:\n{args.question}\n\nCONTEXT:\n{context}"
    )
    max_tokens = min(args.compressor_max_tokens, max(128, int(len(context) / 4 * args.ratio * 1.5)))
    key = os.environ.get(args.compressor_key_env) if args.compressor_key_env else None
    result, elapsed = chat(
        args.compressor_url, args.compressor_model,
        [{"role": "system", "content": "You are a faithful context excerpt selector."},
         {"role": "user", "content": prompt}],
        max_tokens, args.timeout, key,
    )
    return message_text(result), elapsed, result.get("usage", {})


def median(values: list) -> float | None:
    values = [float(v) for v in values if isinstance(v, (int, float))]
    return round(statistics.median(values), 3) if values else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--status-url", help="optional /status URL; inferred from --url")
    parser.add_argument("--model", default="unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M")
    parser.add_argument("--mode", choices=("extractive", "model"), default="extractive")
    parser.add_argument("--ratio", type=float, default=0.5, help="approximate context character ratio (0, 1)")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--max-output-tokens", type=int, default=96)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--compressor-url", help="second OpenAI-compatible /v1/chat/completions URL")
    parser.add_argument("--compressor-model", help="model name accepted by compressor endpoint")
    parser.add_argument("--compressor-max-tokens", type=int, default=2048)
    parser.add_argument("--target-key-env", default="SPLASH_API_KEY")
    parser.add_argument("--compressor-key-env", default="COMPRESSOR_API_KEY")
    parser.add_argument("--output", type=Path, default=Path("prefill-compression-results.json"))
    args = parser.parse_args(argv)
    if not 0 < args.ratio < 1 or args.samples < 1 or args.timeout <= 0:
        parser.error("--ratio must be in (0,1), --samples and --timeout must be positive")
    if args.mode == "model" and (not args.compressor_url or not args.compressor_model):
        parser.error("--mode model requires --compressor-url and --compressor-model")
    if args.target_key_env == "":
        args.target_key_env = None
    if args.compressor_key_env == "":
        args.compressor_key_env = None
    try:
        context = args.document.read_text(encoding="utf-8")
    except OSError as exc:
        parser.error(f"cannot read {args.document}: {exc}")
    if not context.strip():
        parser.error("--document is empty")

    records = []
    print(f"Context: {len(context):,} chars | mode={args.mode} | target ratio={args.ratio:.2f}")
    try:
        for sample in range(1, args.samples + 1):
            pair_order = ("full", "compressed") if sample % 2 else ("compressed", "full")
            for variant in pair_order:
                started = time.monotonic()
                compressor_usage = {}
                if variant == "full":
                    candidate_context = context
                elif args.mode == "extractive":
                    candidate_context = extractive(context, args.question, args.ratio)
                else:
                    candidate_context, _, compressor_usage = compress_with_model(args, context)
                compression_ms = (time.monotonic() - started) * 1000 if variant == "compressed" else 0.0
                row = target_request(args, candidate_context, variant, sample)
                row["compression_ms"] = round(compression_ms, 3)
                row["end_to_end_ms"] = round(compression_ms + row["target_http_wall_ms"], 3)
                if variant == "compressed":
                    row["actual_char_ratio"] = round(len(candidate_context) / len(context), 4)
                    row["compressor_usage"] = compressor_usage
                    row["compressed_context"] = candidate_context
                records.append(row)
                print(
                    f"sample={sample} {variant:10s} tokens={row.get('prompt_tokens')} "
                    f"TTFT={row.get('target_ttft_ms')}ms end-to-end={row['end_to_end_ms']}ms",
                    flush=True,
                )
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    full = [r for r in records if r["variant"] == "full"]
    compressed = [r for r in records if r["variant"] == "compressed"]
    summary = {
        "median_full_prompt_tokens": median([r.get("prompt_tokens") for r in full]),
        "median_compressed_prompt_tokens": median([r.get("prompt_tokens") for r in compressed]),
        "median_full_target_ttft_ms": median([r.get("target_ttft_ms") for r in full]),
        "median_compressed_target_ttft_ms": median([r.get("target_ttft_ms") for r in compressed]),
        "median_full_prefill_wall_ms": median([r.get("target_prefill_wall_ms") for r in full]),
        "median_compressed_prefill_wall_ms": median([r.get("target_prefill_wall_ms") for r in compressed]),
        "median_compression_ms": median([r.get("compression_ms") for r in compressed]),
        "median_full_end_to_end_ms": median([r.get("end_to_end_ms") for r in full]),
        "median_compressed_end_to_end_ms": median([r.get("end_to_end_ms") for r in compressed]),
    }
    if summary["median_full_end_to_end_ms"] and summary["median_compressed_end_to_end_ms"]:
        summary["end_to_end_speedup"] = round(
            summary["median_full_end_to_end_ms"] / summary["median_compressed_end_to_end_ms"], 3)
    else:
        summary["end_to_end_speedup"] = None
    output = {
        "experiment": "splash-prefill-context-compression-v1",
        "target_url": args.url, "target_model": args.model,
        "document_path": str(args.document),
        "document_sha256": hashlib.sha256(context.encode()).hexdigest(),
        "question": args.question,
        "settings": {"mode": args.mode, "ratio": args.ratio, "samples": args.samples},
        "summary": summary, "records": records,
        "notes": [
            "target_ttft_ms excludes compression; end_to_end_ms includes compression plus target HTTP time.",
            "extractive mode is a deterministic lexical baseline, not learned speculative prefill.",
            "Inspect the full and compressed answers before accepting a quality/speed trade-off.",
            "The optional compressor endpoint receives the full document; do not send sensitive text to an untrusted endpoint.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\nSummary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"\nSaved full results and answers to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
