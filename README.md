<p align="center">
  <img src="assets/banner.jpg" alt="Lucebox" width="100%">
</p>

<p align="center">
  <a href="https://lucebox.com"><img src="https://img.shields.io/badge/lucebox.com-f5c842?style=for-the-badge&logo=safari&logoColor=090909&labelColor=090909" alt="lucebox.com"></a>
  <a href="https://discord.gg/yHfswqZmJQ"><img src="https://img.shields.io/badge/Discord-f5c842?style=for-the-badge&logo=discord&logoColor=090909&labelColor=090909" alt="Discord"></a>
  <a href="https://lucebox.com/blog"><img src="https://img.shields.io/badge/Blog-f5c842?style=for-the-badge&logo=rss&logoColor=090909&labelColor=090909" alt="Blog"></a>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-e8e8ed?style=for-the-badge&labelColor=090909" alt="MIT"></a>
  <a href="https://developer.nvidia.com/cuda-toolkit"><img src="https://img.shields.io/badge/CUDA-12%2B-76b900?style=for-the-badge&logo=nvidia&logoColor=76b900&labelColor=090909" alt="CUDA 12+"></a>
  <a href="https://isocpp.org"><img src="https://img.shields.io/badge/C%2B%2B-17-e8e8ed?style=for-the-badge&logo=cplusplus&logoColor=e8e8ed&labelColor=090909" alt="C++17"></a>
</p>

<p align="center">
  <strong>Open LLM inference, rewritten by hand for one specific chip at a time.</strong><br/>
  Kernels, speculative decoding, and quantization, tailored per target.<br/>
  We don't wait for better silicon. We rewrite the software.
</p>

---

> The efficiency gap between consumer and datacenter silicon isn't inherent.
> It's an artifact of running generic software on capable hardware.

Local AI is becoming the default for developers shipping agents, researchers reproducing results, teams tired of per-token bills, and anyone whose data shouldn't leave the room. The hardware to run a 20B+ hybrid model at home mostly exists already, a second-hand RTX 3090, a Ryzen AI MAX+ 395, an Apple M-series chip. The kernels to run one *well* don't.

Lucebox is where we build them. One model, one hardware target, one optimization problem at a time. Open weights, open kernels, open benchmarks.

[Read the first writeup →](https://lucebox.com/blog/megakernel)

---

## Inside the box

Two projects today, more coming. Each one is a self-contained release with its own benchmarks and paper-style writeup.

<p align="center">
  <a href="megakernel/"><img src="assets/svg/card-megakernel-dark.svg" alt="Megakernel" width="46%"></a>
  &nbsp;&nbsp;
  <a href="dflash/"><img src="assets/svg/card-dflash-dark.svg" alt="DFlash 27B" width="46%"></a>
</p>

---

## 01 · Megakernel

**The first megakernel for hybrid DeltaNet/Attention LLMs.** All 24 layers of Qwen 3.5-0.8B in a single CUDA dispatch, 1.87 tok/J on a 2020 GPU, matching Apple's latest silicon at 2× the throughput.

```bash
git clone https://github.com/Luce-Org/lucebox-hub
cd lucebox-hub/megakernel
pip install -e .
python final_bench.py
```

| Method | Prefill pp520 | Decode tg128 | tok/J |
|--------|:-------------:|:------------:|:-----:|
| **Megakernel** `@220W` | **37,800** | **413** | **1.87** |
| llama.cpp BF16 `@350W` | 11,247 | 267 | 0.76 |
| PyTorch HF | 7,578 | 108 | — |

**What makes it work:** 82 blocks, 512 threads, one persistent kernel. No CPU round-trips between layers. Weights streamed straight from HuggingFace. Cooperative grid sync instead of ~100 kernel launches per token. Power ceiling hit before compute ceiling, so DVFS converts tight execution straight into saved watts.

[Full writeup →](megakernel/README.md) · [Benchmarks →](megakernel/RESULTS.md) · [Blog post →](https://lucebox.com/blog/megakernel)

---

## 02 · DFlash 27B &nbsp;<sub><sup>`· coming soon`</sup></sub>

**Hybrid DeltaNet quantization for Qwen 3.5-27B, on one RTX 3090.** A 27B hybrid model shouldn't fit in 24 GB of VRAM. With DeltaNet-aware mixed precision and a fused decode path borrowed from the megakernel, it does, without the recurrence collapse that kills naive int4 on linear attention.

```
    baseline 27B BF16              → 54 GB        (two GPUs, or CPU offload)
    naive w4a16 GPTQ               → 13.5 GB      (works for attention, breaks DeltaNet)
    dflash w4a16 + DeltaNet-aware  → 14.8 GB      (attention int4, DeltaNet int8 state)
                                                   fits in 24 GB with room for KV cache
```

Target: a 27B model that runs locally on a single consumer card, with recurrence quality intact. Benchmarks and open weights land in Q2 2026.

[Design notes →](dflash/README.md)

---

## Why this exists

Conventional wisdom says NVIDIA is fast but power-hungry; Apple is efficient but slower. On paper, that checks out, 267 tok/s at 350 W on llama.cpp vs 229 tok/s at 130 W on an M5 Max.

We've thought the problem was never the hardware. It was ~100 kernel launches per token. It was quantization schemes designed for dense attention, applied blindly to linear recurrence. It was generic software on capable silicon.

Lucebox is where we publish the fixes. Each project is a standalone artifact, a kernel, a quantization scheme, a benchmark harness, with full writeup and MIT source. Together they build toward a single command surface, `lucebox`, that makes hybrid-architecture local inference a one-line install.

---

## Quickstart

```bash
git clone https://github.com/Luce-Org/lucebox-hub
cd lucebox-hub

# megakernel (Qwen 3.5-0.8B, batch 1)
cd megakernel && pip install -e . && python final_bench.py

# dflash 27B                                                [coming soon]
```

**Requirements:** NVIDIA GPU (Ampere+), CUDA 12+, PyTorch 2.0+. Tested on RTX 3090 (2020).

**Optional, find your GPU's sweet spot:** `sudo nvidia-smi -pl 220`

---

## Repository layout

```
lucebox-hub/
├── megakernel/             · fused forward pass for Qwen 3.5-0.8B
├── dflash/                 · DeltaNet-aware quantization for 27B   [wip]
├── benchmarks/             · shared measurement harness            [wip]
├── shared/                 · cross-project utilities               [wip]
├── docs/                   · longform writeups and design notes
└── assets/                 · banners, cards, diagrams
```

---

## Roadmap

```
  Q1 2026    ▮▮▮▮▮▮▮▮▮▮    RTX 3090 kernels & optimizations
  Q2 2026    ▮▮▮▮▮▯▯▯▯▯    Ryzen AI MAX+ 395 optimizations
  Q3 2026    ▮▮▯▯▯▯▯▯▯▯    heterogeneous CPU + GPU with unified memory
```

---

## Citation

```bibtex
@software{lucebox_2026,
  title  = {Lucebox: Open LLM Inference, Rewritten by Hand for One Specific Chip at a Time},
  author = {Lucebox},
  url    = {https://github.com/Luce-Org/lucebox-hub},
  year   = {2026}
}
```

Per-project citations live in each subproject's README.

---

## Inspired by

- [Hazy Research](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) — for the megakernel idea and the intelligence-per-watt methodology.
- [DDTree (Ringel et al., 2025)](https://arxiv.org/abs/2502.20762) — tree-structured verify for speculative decoding; what DFlash 27B leans on for its 3.5× speedup.
- [AlpinDale/qwen_megakernel](https://github.com/AlpinDale/qwen_megakernel), [Infatoshi/MegaQwen](https://github.com/Infatoshi/MegaQwen) — prior art on fused Qwen kernels.
- [z-lab/dflash](https://github.com/z-lab/dflash) — the linear-attention kernel work whose name we borrow for project 02.

---

## Community

- **Discord** — [discord.gg/yHfswqZmJQ](https://discord.gg/yHfswqZmJQ)
- **Website** — [lucebox.com](https://lucebox.com)
- **Issues** — [github.com/Luce-Org/lucebox-hub/issues](https://github.com/Luce-Org/lucebox-hub/issues)
- **Blog** — [lucebox.com/blog](https://lucebox.com/blog)

---

<p align="center">
  <sub><a href="LICENSE">MIT</a> · <a href="https://lucebox.com">Lucebox.com</a></sub>
</p>
