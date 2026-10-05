# Lossless Q8 arena runtime

This opt-in implementation reduces host RAM and CUDA VRAM for an **already quantized Q8_0 model**. It preserves every int8 code and every original FP16 scale bit. It does not train or requantize the model.

Each tensor uses one backing arena: a small header, a fixed dictionary band, aligned int8 codes, and packed 9–11-bit scale indices. The native CPU GEMM and CUDA GEMV/MMQ kernels read this representation directly, preserving their arithmetic, fusion, and reduction schedules. The scheduler also transfers compact CPU weights to CUDA for large prefill batches. Whole-model or per-matrix expansion is not part of the serving path.

Loading faults and releases source ranges incrementally instead of populating the entire original mapping first. This accounts for most of the **peak** host-memory improvement; it is separate from the smaller resident weight representation.

## Measured full-model results

Checkpoint: `Qwen3.8-27B-Q8_0.gguf` (GGUF architecture `qwen35`). Original file SHA-256: `f0fc43cfd802bd87bc7b95f2120553a6407848e3229126b623cc397427857e8a`.

RTX 4080 SUPER, Ryzen 9 7950X3D, 32 GPU layers, 16 CPU threads, 2048-token context, batch/ubatch 128, Flash Attention enabled, mmap loading, lazy loading disabled. The pristine baseline was commit `7049ff0cbeb1f5ead231de4522af6b75d8d773c0`. The fork's subsequent Hexagon-only commit does not change these CPU/CUDA paths.

Four fresh-process runs in recorded shuffled order (baseline, arena, baseline, arena) replayed identical tokens. Each run evaluated four prompts, including a 570-token prompt, followed by 48 forced decode steps per prompt. The reported values are medians of two runs per implementation.

| Metric | Pristine baseline | Arena |
| --- | ---: | ---: |
| Peak host RSS | 27.450 GiB | 16.628 GiB |
| Steady host RSS | 14.745 GiB | 14.437 GiB |
| Peak sampled process VRAM | 13,476 MiB | 13,184 MiB |
| Decode | 4.376 tokens/s | 4.301 tokens/s |
| Prefill, 634 tokens across four prompts | 5.105 s | 5.588 s |
| Full-logit differences against reference | 0 | 0 |

Savings: **10.82 GiB peak host RAM, 315.9 MiB steady host RAM, and 292 MiB process VRAM**. Decode is **1.7% slower** and prefill **9.5% slower** in these runs. This is a memory improvement with an explicit speed tradeoff, not a throughput improvement.

Every one of the **47,677,440 logits per run** was bit-identical to the reference. All 498 Q8 tensors loaded for ordinary inference were packed (265 CPU, 233 CUDA); eight MTP tensors in the file were not loaded by either implementation. The packed active weights occupy **627.7 MiB less backing storage**, including dictionary padding and indices.

`results/summary.json` contains individual measurements and exact parity results. The four `*-process.json` files contain one-second RAM/VRAM samples; paths were normalized for publication. RSS comes from `/proc` and `getrusage`; process VRAM comes from `nvidia-smi` and is sampled, not an allocation-level peak. Neither implementation swapped in these runs.

A full-model Compute Sanitizer memcheck also passed with **zero errors**, using 24 GPU layers, a 128-token prefill, four decode tokens, and no benchmark warmup. The smaller GPU-layer count leaves instrumentation headroom. Its log is included in `results/memcheck.txt`; its timings are not performance evidence.

## Build and run

Commands below run from the repository root. The tested build targets Ada (`89`) and an AVX2-capable CPU.

```sh
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-cli llama-bench -j 8
python -m venv .venv-q8arena
.venv-q8arena/bin/pip install -r examples/q8arena/requirements.txt

MODEL=/path/to/Qwen3.8-27B-Q8_0.gguf
.venv-q8arena/bin/python examples/q8arena/build_index.py "$MODEL" model.q8index
GGML_Q8_ARENA="$PWD/model.q8index" build/bin/llama-cli \
  -m "$MODEL" -ngl 32 -lm mmap -lzm off -c 2048 -b 128 -ub 128 -t 16 -fa on
```

The index is approximately 627 KB for this checkpoint and contains scale dictionaries and layout metadata only. The original GGUF is still required. The runtime packs the actual GGUF bytes while loading; a scale absent from the supplied dictionary causes an explicit failure.

Leave `GGML_Q8_ARENA` unset to use the ordinary representation. Set the variable before starting the process; switching indices within a process is not supported. No global model files or installed application backends are modified.

This implementation is scoped to the tested dense Q8_0 model and CPU/CUDA configuration. It requires whole-tensor loading (`-lm mmap -lzm off`), AVX2, two-dimensional Q8 matrices, and at most 2048 distinct scales per matrix. It is not a general codec for other quantization formats, arbitrary tensor views, tensor-parallel splits, or mutable/training weights. Same-layout zero-offset scheduler aliases are supported.

## Reproduce full-logit validation

Build the helper against the candidate library:

```sh
c++ -O3 -std=c++17 examples/q8arena/validate_model.cpp \
  -Iinclude -Iggml/include -Ivendor/nlohmann -Lbuild/bin \
  -Wl,-rpath,'$ORIGIN/bin' -lllama -lggml -lggml-base -o build/q8arena-validate
```

Build the pristine base in a separate checkout. Point `BASELINE_LIBS` at that build's `bin` directory. The helper fixes the configuration listed above and records or replays the complete token sequence.

```sh
env -u GGML_Q8_ARENA LD_LIBRARY_PATH="$BASELINE_LIBS" \
  .venv-q8arena/bin/python examples/q8arena/measure_process.py --output baseline-process.json -- \
  build/q8arena-validate "$MODEL" baseline record tokens.json

GGML_Q8_ARENA="$PWD/model.q8index" \
  .venv-q8arena/bin/python examples/q8arena/measure_process.py --output arena-process.json -- \
  build/q8arena-validate "$MODEL" arena replay tokens.json

.venv-q8arena/bin/python examples/q8arena/compare_logits.py \
  baseline.logits arena.logits --output comparison.json
```

The comparison records exact bit differences as well as KL divergence and greedy-token agreement. The helper fails on any bit difference; the accepted result above has zero bit differences. Each logits artifact is approximately 182 MiB. Run baseline and candidate sequentially to avoid GPU contention.

For the sanitizer run:

```sh
GGML_Q8_ARENA="$PWD/model.q8index" compute-sanitizer --tool memcheck --error-exitcode 3 \
  --log-file memcheck.log build/bin/llama-bench -m "$MODEL" -ngl 24 \
  -p 128 -n 4 -b 128 -ub 128 -t 16 -r 1 -fa on -lm mmap -lzm off --no-warmup
```

Exit-time `q8arena_stats` report real runtime usage: kinds 0/1 are CPU/CUDA weight packing, 2/3 CPU decode/prefill dispatch, 4/5 CUDA decode/prefill host dispatch, 6 CPU embedding gathers, and 7 compact host-to-device copies. CUDA dispatch counts exclude repeated executions of already captured graphs and are not GPU kernel-launch counts.
