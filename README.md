# qwen-mtp-optimizations

> Six speculative-decoding optimization variants for **Qwen3.5-27B** Multi-Token Prediction in llama.cpp. **Variant 01 — adaptive chained MTP — delivers 1.99× over K=1 vanilla** with `MTP_CHAIN_KMAX=2 MTP_CHAIN_THRESH=0.85`. The others are documented exploration on top of the [qwen-mtp-llamacpp](https://github.com/quivent/qwen-mtp-llamacpp) infrastructure.

## 🏆 The winning recipe — adaptive chained MTP

```bash
MTP_CHAIN_KMAX=2 MTP_CHAIN_THRESH=0.85 \
    ./build/bin/llama-mtp-speculative -m qwen3.5-27b-q4km.gguf \
    -p "Explain photosynthesis." -n 64 -ngl 99 -c 2048
```

**5-prompt benchmark** (Qwen3.5-27B Q4_K_M, M4 Max, quiet GPU, output coherence verified against plain decode):

| Prompt | K=1 vanilla | K=2 adaptive chain | Speedup |
|---|---|---|---|
| Write a haiku about spring. | 4.6 tok/s | **13.0 tok/s** | **2.83×** |
| Explain photosynthesis in one paragraph. | 7.1 tok/s | **14.7 tok/s** | 2.07× |
| Write a Python function to compute Fibonacci. | 6.6 tok/s | **14.0 tok/s** | 2.12× |
| List the planets of the solar system. | 8.3 tok/s | **13.8 tok/s** | 1.66× |
| Translate hello world to French. | 8.5 tok/s | **14.4 tok/s** | 1.69× |
| **Mean** | **7.02** | **13.98** | **1.99×** |

Plain decode baseline: 17.90 tok/s. Adaptive chain reaches **0.78× of plain decode** — the closest any Qwen3.5-27B speculative path has come in llama.cpp. The same recipe MLX `stacked_v2.py` uses to hit 1.73× over its baseline. See [qwen-mtp-research/docs/the-recipe.md](https://github.com/quivent/qwen-mtp-research/blob/main/docs/the-recipe.md) for the full cost breakdown and the remaining-lever analysis.

## The variants

### 01 — Adaptive chain (`MTP_CHAIN_THRESH`, `MTP_CHAIN_KMAX`) 🏆 **THE WINNER**
Top-1 probability gating on a chained recurrent MTP path. On each draft call, run the MTP head once, check the top-1 confidence, and if above threshold chain again — feeding the MTP head's own output hidden (`t_mtp_out_hidden`) as the next step's `prev_hidden`. This is the same recurrent-stack technique MLX `stacked_v2.py` uses. Combined with the rollback bookkeeping fix from [qwen-mtp-llamacpp](https://github.com/quivent/qwen-mtp-llamacpp) patch 11, this delivers **1.99× over K=1 vanilla** at `MTP_CHAIN_KMAX=2 MTP_CHAIN_THRESH=0.85` with coherent output on all benchmark prompts.

### 02 — Debug verify (`MTP_DEBUG_VERIFY`)
Diagnostic instrumentation. Dumps draft vs target argmax per verify step so you can see exactly where the drafter diverges from the main model. Essential debugging primitive.

### 03 — Drift refresh (`MTP_REFRESH_EVERY=N`)
Periodic T=1 plain-decode every N committed tokens to refresh the recurrent hidden state and bound DeltaNet drift. The cleanest single-knob variant — works on any hybrid model.

### 04 — Predictive hidden draft
Identity / linear-extrapolation predictor for `prev_hidden`. Avoids the cost of running the main model just to get a fresh hidden state for the next draft step.

### 05 — Perturbed-head ensemble (`MTP_ENSEMBLE_K=N`)
Top-K sampling from a single MTP forward pass. K sibling candidates verified as a tree-fork in one batch. The ensemble path explores multiple draft hypotheses for the cost of one MTP pass.

### 06–07 — Branching speculative tree (`MTP_TREE_B`, `MTP_TREE_DEPTH`)
Full B*D tree with multi-sequence batching. Each branch root is independently chained for D steps; all `B*D` leaves verified in one `llama_decode` call using a multi-seq batch. Uses unified KV (`kv_unified=true`) for cross-stream `seq_cp`.

### 08 — Ensemble fast-path skip
Optimization on top of the ensemble path: on a top-1 hit, trim attn KV + force the recurrent position metadata backward instead of doing a full snapshot/re-decode. Skips the second forward pass on ~55% of cycles. Requires the `llama_memory_seq_force_recurrent_pos` primitive added in this branch.

### 09 — Stacked hidden-noise validator (NEGATIVE result)
Run the MTP head N times per draft step with small Gaussian noise added to `prev_hidden`, ensemble-vote the results. Hypothesis: noisy logits would average out into a more reliable argmax. **Result: doesn't work.** The MTP head is structurally saturated — small perturbations don't move the argmax at all (accept count was byte-identical at N=1, 2, 4, 8), and large perturbations only shift it within run-to-run noise. The bottleneck isn't noisy logits; it's that the head's top-1 is structurally wrong ~90% of the time. Per-pass cost makes it strictly worse (5.4 tok/s at N=2 vs 7.8 vanilla). Optimal N=1, i.e. don't enable. **Decisively negative — published as the implementation record.**

## Variant measurement status (post-bug-fix)

All variants are implemented on top of the [qwen-mtp-llamacpp](https://github.com/quivent/qwen-mtp-llamacpp) infrastructure, which contains a one-line cache-bookkeeping fix (patch 11) that unblocks correct output across the spec path. Measurements below are on the fixed tree:

| Variant | Output coherent | Speedup vs K=1 vanilla | Status |
|---|---|---|---|
| **01 adaptive chain** | ✓ | **1.99×** (13.98 vs 7.02 tok/s mean) | 🏆 **Winner — the MLX stacked_v2 recipe** |
| 03 drift refresh | ✓ (pre-fix) | 10× accept jump pre-fix | Redundant post-fix (chain already bounds drift via re-decode) |
| 04 predictive hidden | ✓ (pre-fix) | K=2 accept 8→83% on short prefixes pre-fix | Superseded by variant 01 |
| 05 ensemble slow-path | ✓ | needs post-fix re-validation | Orthogonal to 01 — tree-fork variant of the same idea |
| 06–07 branching tree | ✓ | needs post-fix re-validation | Orthogonal — multi-sequence parallelism |
| 08 ensemble fast-path | ✗ | — | **Broken** — recurrent contamination on this hybrid model |
| 09 stacked hidden-noise | ✓ | 0.58× to 0.69× | **Decisively negative** — MTP head is saturated, ensembling doesn't decorrelate |

The fast-path optimization in #08 is the one variant we have *post-fix* evidence about, and it breaks output. The others were "wins" pre-fix and we don't yet have the post-fix numbers to know if they're real. The patches are preserved here as the implementation record; the [qwen-mtp-research](https://github.com/quivent/qwen-mtp-research) repo discusses what each variant was trying to attack.

## Why publish negative-result-pending patches?

Because the *infrastructure work* in each patch is reusable. The branching tree path's discovery that hybrid recurrent memory needs `kv_unified=true` for `seq_cp`, the `llama_memory_seq_force_recurrent_pos` primitive, the in-graph AR loop, the snapshot/restore plumbing — all of that is correct, hard-won work that the next person attacking spec decoding on a hybrid model shouldn't have to rediscover.

## Applying the patches

These patches apply on top of [qwen-mtp-llamacpp](https://github.com/quivent/qwen-mtp-llamacpp). Each one is independent — apply only the variants you want to test.

```bash
# After applying the qwen-mtp-llamacpp patches:
git am path/to/qwen-mtp-optimizations/patches/03-feat-mtp-MTP_REFRESH_EVERY*.patch
cmake --build build -j 12 --target llama-mtp-speculative

MODEL=path/to/qwen3.5-27b-q4km.gguf
MTP_REFRESH_EVERY=8 ./build/bin/llama-mtp-speculative -m $MODEL \
    -p "Explain photosynthesis in one paragraph." -n 64 -ngl 99
```

## Related repos

- **[qwen-mtp-llamacpp](https://github.com/quivent/qwen-mtp-llamacpp)** — the infrastructure substrate (patches 1-11) these variants build on
- **[qwen-mtp-tensors](https://github.com/quivent/qwen-mtp-tensors)** — converter & tensor-naming work for Qwen3.5 MTP
- **[qwen-mtp-research](https://github.com/quivent/qwen-mtp-research)** — design docs, methodology, the per-position-heads (DeepSeek V3 style) plan

## License

MIT — see LICENSE.
