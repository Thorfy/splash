# Prefill context-compression experiment

This is a client-side experiment against a running Splash server on Apple Silicon. It does not alter the model weights, Metal kernels, attention implementation, cache, or default inference path.

It measures whether shortening a long context saves enough target prefill time to pay for the compression step.

## 1. Run the regular Splash build

Start the model you usually use and wait until it is ready. For the documented GGUF variant:

```sh
splash serve --model unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M
```

Use the actual model value for your installation; MLX variants are also supported. Keep the server running on its usual port, and avoid sending other requests to it during the benchmark so the status-counter deltas stay meaningful.

## 2. Prepare a test document

Save a representative, non-sensitive long prompt/context as context.txt: for example, a source file, a bundle of project notes, documentation, or a long conversation export. Ask a specific question whose answer really is present in the document. Avoid secrets and private data, especially when testing a remote compressor endpoint.

## 3. Run the local extractive baseline

First, run the lightweight selector tests from the repository root:

```sh
PYTHONPATH=. python3 -m unittest dev.benchmarks.test_prefill_compression -v
```

From the repository root, on branch experiment/speculative-prefill-qwen38:

```sh
python3 -m dev.benchmarks.prefill_compression  --document context.txt  --question "What is the production database timeout?"  --model unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M  --mode extractive --ratio 0.50 --samples 3
```

If your model label differs, pass it with --model. If the server uses a different endpoint, pass --url. When authentication is enabled, the script reads SPLASH_API_KEY; set --target-key-env "" to skip the key. Results are written to prefill-compression-results.json unless --output specifies another path.

Try --ratio 0.70 and --ratio 0.30 afterwards. A higher ratio keeps more text and usually sacrifices less information.

## 4. Optional learned compressor

The model mode calls a second OpenAI-compatible chat-completions endpoint to select verbatim excerpts:

```sh
python3 -m dev.benchmarks.prefill_compression  --document context.txt  --question "What is the production database timeout?"  --model unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M  --mode model  --compressor-url http://127.0.0.1:8001/v1/chat/completions  --compressor-model YOUR_SMALL_MODEL_NAME  --ratio 0.50 --samples 3
```

The compressor endpoint receives the entire document. Use only a trusted endpoint. This mode measures compressor latency as part of compression_ms and end_to_end_ms. Running two local models at once may cause memory pressure; the extractive mode requires no second model.

## How to read the result

The JSON records full and compressed answers alongside:

- actual prompt_tokens reported by Splash;
- target TTFT and request wall time;
- prefill wall-time delta from Splash's /status counters, if available;
- compression time and combined end-to-end time;
- a summary of median values across samples.

The script adds a unique nonce near the start of every target prompt so that the long document body is not reused as a cached prefix across runs (a few shared chat-template tokens may still be cached). Requests alternate order across samples to reduce order bias, but thermal state, other applications and memory pressure can still affect measurements. Prefer at least 5 samples for a serious comparison.

The key comparison is median_full_end_to_end_ms against median_compressed_end_to_end_ms; target-only TTFT excludes the compression cost and can therefore be misleading. Inspect the answer values before accepting a speed/quality trade-off: reject a ratio that drops essential facts, identifiers, exceptions, or code lines even if it is faster.

## Scope and next research step

The default extractive selector is a deterministic lexical baseline, **not** a learned speculative-prefill model and not DeepSeek-style compressed attention. It tests the hypothesis that less context could reduce TTFT for the workload.

A true model-aware prefill optimization inside Splash is a separate native-engine project. Qwen3.8-27B uses recurrent Gated DeltaNet layers alongside full attention; simply deleting prompt tokens before the model would change the recurrent state and position semantics. Prefix-cache, tool-call, vision, and answer-quality behavior would need explicit treatment. Native compressed attention in the manner of DeepSeek would require new kernels and likely training/distillation, rather than a drop-in change to the existing checkpoint.
