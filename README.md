# qwen-mtp-optimizations

> Six speculative-decoding optimization variants for **Qwen3.5-27B** Multi-Token Prediction in llama.cpp. Each variant attacks a different bottleneck — drafter quality, drift, parallelism, rollback cost — and each one is its own apply-or-skip patch on top of the [qwen-mtp-llamacpp](https://github.com/quivent/qwen-mtp-llamacpp) infrastructure.

This is the **exploration repo**. Eight patches, six distinct ideas, and one in-graph debug instrumentation patch. Each one was developed and benchmarked against Qwen3.5-27B Q4_K_M on M4 Max.

## The variants

### 01 — Adaptive chain (`MTP_CHAIN_THRESH`, `MTP_CHAIN_KMAX`)
Top-1 probability gating on the MTP draft chain. Stop chaining when confidence drops below threshold. Avoids wasted draft passes on tokens the head is unsure about.

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

## ⚡ The headline result (post-bug-fix re-validation)

Variants 01 (adaptive chain) + the in-graph chained-recurrent threading from [qwen-mtp-llamacpp](https://github.com/quivent/qwen-mtp-llamacpp) patch 09 deliver **1.99× over K=1 vanilla** when combined with the rollback bookkeeping fix from patch 11:

```bash
MTP_CHAIN_KMAX=2 MTP_CHAIN_THRESH=0.85 ./build/bin/llama-mtp-speculative -m $MODEL ...
```

5-prompt mean (Qwen3.5-27B Q4_K_M, M4 Max): **K=1 vanilla 7.02 tok/s → chained recipe 13.98 tok/s** (0.78× of plain decode 17.90). Coherent output verified against plain decode on all 5 prompts. This is the same recipe MLX `stacked_v2.py` uses to hit 1.73× on its baseline.

See [qwen-mtp-research/docs/the-recipe.md](https://github.com/quivent/qwen-mtp-research/blob/main/docs/the-recipe.md) for the full breakdown.

## Honest measurement caveat

These variants were originally developed against an MTP path that contained a one-line cache-bookkeeping bug (fixed in [qwen-mtp-llamacpp](https://github.com/quivent/qwen-mtp-llamacpp) patch 11). With that bug present, every variant's measurements were on degraded text. After the fix, the K=1 vanilla baseline produces correct output at 7.64 tok/s vs plain decode at 17.90 tok/s on Qwen3.5-27B Q4_K_M (M4 Max).

**Re-validation status of each variant on the post-fix tree:**

| Variant | Code applies | Output coherent | Throughput vs K=1 vanilla |
|---|---|---|---|
| 01 adaptive chain | ✓ | needs re-validation | TBD |
| 03 drift refresh | ✓ | needs re-validation | TBD |
| 04 predictive hidden | ✓ | needs re-validation | TBD |
| 05 ensemble (slow path) | ✓ | needs re-validation | TBD |
| 06–07 tree | ✓ | needs re-validation | TBD |
| 08 ensemble fast-path | ✓ | **broken** — recurrent contamination corrupts output on this hybrid model | — |
| 09 stacked hidden-noise | ✓ | ✓ | **strictly worse** at every N (0.69× at N=2, 0.58× at N=8) — head is saturated |

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
