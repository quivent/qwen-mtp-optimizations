<div align="center">

```
  ___  __  __ _____ ____  
 / _ \|  \/  |_   _|  _ \ 
| | | | |\/| | | | | |_) |
| |_| | |  | | | | |  __/ 
 \__\_\_|  |_| |_| |_|    
 O P T I M I Z A T I O N S
```

**Six speculative-decoding optimization variants for Qwen3.5-27B in llama.cpp.**

*Delivers up to 1.99× speedup over K=1 vanilla with adaptive chained MTP.*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

</div>

---

## 📑 Table of Contents
- [🏆 The Winning Recipe](#-the-winning-recipe)
- [🧩 The Variants](#-the-variants)
- [📊 Measurement Status](#-measurement-status)
- [💡 Why Publish Negative Results?](#-why-publish-negative-results)
- [🚀 Quick Start](#-quick-start)
- [🔗 Related Repositories](#-related-repositories)
- [📄 License](#-license)

---

## 🏆 The Winning Recipe

**Variant 01 — adaptive chained MTP — delivers 1.99× over K=1 vanilla**.

```bash
MTP_CHAIN_KMAX=2 MTP_CHAIN_THRESH=0.85 \
    ./build/bin/llama-mtp-speculative -m qwen3.5-27b-q4km.gguf \
    -p "Explain photosynthesis." -n 64 -ngl 99 -c 2048
```

**5-prompt benchmark** (Qwen3.5-27B Q4_K_M, M4 Max, quiet GPU, output coherence verified):

| Prompt | K=1 vanilla | K=2 adaptive chain | Speedup |
|---|---|---|---|
| Write a haiku about spring. | 4.6 tok/s | **13.0 tok/s** | **2.83×** |
| Explain photosynthesis... | 7.1 tok/s | **14.7 tok/s** | 2.07× |
| Python function Fibonacci. | 6.6 tok/s | **14.0 tok/s** | 2.12× |
| List the planets. | 8.3 tok/s | **13.8 tok/s** | 1.66× |
| Translate hello world. | 8.5 tok/s | **14.4 tok/s** | 1.69× |
| **Mean** | **7.02** | **13.98** | **1.99×** |

> [!TIP]
> Adaptive chain reaches **0.78× of plain decode** (baseline: 17.90 tok/s) — the closest any Qwen3.5-27B speculative path has come in llama.cpp.

---

## 🧩 The Variants

<details>
<summary><b>01 — Adaptive chain 🏆 THE WINNER</b></summary>
Top-1 probability gating on a chained recurrent MTP path. Same recurrent-stack technique MLX uses.
</details>

<details>
<summary><b>02 — Debug verify</b></summary>
Diagnostic instrumentation dumping draft vs target argmax.
</details>

<details>
<summary><b>03 — Drift refresh</b></summary>
Periodic T=1 plain-decode to bound DeltaNet drift.
</details>

<details>
<summary><b>04 — Predictive hidden draft</b></summary>
Predictor for `prev_hidden` to avoid main forward pass costs.
</details>

<details>
<summary><b>05 — Perturbed-head ensemble</b></summary>
Top-K sampling from a single MTP pass.
</details>

<details>
<summary><b>06–07 — Branching speculative tree</b></summary>
Full B*D tree with multi-sequence batching.
</details>

<details>
<summary><b>08 — Ensemble fast-path skip</b></summary>
Optimization on top of the ensemble path. Skips second forward pass on hits.
</details>

<details>
<summary><b>09 — Stacked hidden-noise validator (NEGATIVE result)</b></summary>
Ensemble-voting with Gaussian noise added to `prev_hidden`. Result: doesn't work; head is structurally saturated.
</details>

---

## 📊 Measurement Status

Measurements post bug-fix:

| Variant | Output coherent | Speedup vs K=1 | Status |
|---|---|---|---|
| **01 adaptive chain** | ✓ | **1.99×** | 🏆 **Winner** |
| 03 drift refresh | ✓ (pre-fix) | — | Redundant post-fix |
| 04 predictive hidden | ✓ (pre-fix) | — | Superseded by variant 01 |
| 05 ensemble slow-path | ✓ | TBD | Orthogonal to 01 |
| 06–07 branching tree | ✓ | TBD | Orthogonal |
| 08 ensemble fast-path | ✗ | — | **Broken** — recurrent contamination |
| 09 stacked hidden-noise | ✓ | 0.58× to 0.69× | **Decisively negative** |

---

## 💡 Why Publish Negative Results?

The infrastructure work in each patch is highly reusable. Discoveries like hybrid recurrent memory needing `kv_unified=true` for `seq_cp` and the `llama_memory_seq_force_recurrent_pos` primitive are hard-won lessons that should not be lost.

---

## 🚀 Quick Start

Apply these patches on top of [qwen-mtp-llamacpp](https://github.com/quivent/qwen-mtp-llamacpp).

```bash
# After applying the qwen-mtp-llamacpp patches:
git am path/to/qwen-mtp-optimizations/patches/03-feat-mtp-MTP_REFRESH_EVERY*.patch
cmake --build build -j 12 --target llama-mtp-speculative

MODEL=path/to/qwen3.5-27b-q4km.gguf
MTP_REFRESH_EVERY=8 ./build/bin/llama-mtp-speculative -m $MODEL \
    -p "Explain photosynthesis in one paragraph." -n 64 -ngl 99
```

---

## 🔗 Related Repositories

- **[qwen-mtp-llamacpp](https://github.com/quivent/qwen-mtp-llamacpp)**
- **[qwen-mtp-tensors](https://github.com/quivent/qwen-mtp-tensors)**
- **[qwen-mtp-research](https://github.com/quivent/qwen-mtp-research)**

---

## 📄 License

MIT.
