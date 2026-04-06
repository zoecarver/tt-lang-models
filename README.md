# tt-lang-models

Reference models that are partially or entirely implemented using [TT-Lang](https://github.com/tenstorrent/tt-lang).

---

## [DFlash](https://github.com/zoecarver/dflash)

DFlash is a lightweight cross-attention draft model for speculative decoding on Tenstorrent hardware. It proposes 16 tokens in parallel, verified by a target Qwen3-30B LLM, achieving a 5-6x decoding speedup. Draft model kernels (RoPE, RMSNorm, SiLU, residual adds) run entirely on device via TT-Lang.

Acceptance rate matches the PyTorch reference model. With caching and 120k context, the draft forward pass runs in 93ms (vs 887ms without caching).

Also includes a full [Qwen3-Coder-30B-A3B inference implementation](dflash/qwen3_inference.py), a 48-layer MoE target model running on 4-chip TP with traced execution and zero host transfers in the hot loop. TT-Lang kernels cover RMSNorm, per-head RMSNorm, RoPE, SiLU, residual adds, softmax, cross-attention, and argmax.

## [Engram](https://github.com/zoecarver/Engram)

A port of the DeepSeek Engram conditional memory module to TT-Lang on Wormhole. Engram uses streaming dataflow kernels with inter-core boundary sharing via PipeNet for overlap-aware depthwise convolution.

| | Gating | Conv | All Kernels |
|---|---|---|---|
| **TTNN** | 3.86 ms | 0.99 ms | 4.84 ms |
| **TT-Lang** | 1.15 ms | 1.02 ms | 2.17 ms |

## [Gemma 4](https://github.com/zoecarver/gemma4)

Autoregressive inference for Google's Gemma 4 E4B on a single Blackhole chip. TT-Lang kernels cover linear, flash attention, RoPE, SwiGLU, and softcap across all 42 layers with sliding/global attention and KV sharing. Runs at ~17 tok/s.

## [nanochat](https://github.com/zoecarver/nanochat)

Inference and training for nanochat entirely in TT-Lang. Every kernel has a backwards version. Single-file implementations at [`nanochat/ttlang/inference.py`](nanochat/ttlang/inference.py) and [`nanochat/ttlang/train.py`](nanochat/ttlang/train.py).

Notable fusions:
- [Fused MLP projection](https://github.com/zoecarver/nanochat/commit/f849d3f) -- replaces 7 dispatches (4 slice matmuls + 3 residual adds) with a single kernel using L1 accumulation via ping-pong DFBs. 13.13 to 15.89 tok/s (+21%).
- [Fused QKV projection](https://github.com/zoecarver/nanochat/commit/9034746) -- reads input once and computes Q, K, V in one dispatch, reducing DRAM reads. 12.30 to 13.13 tok/s (+6.7%).

## [Oasis](https://github.com/zoecarver/open-oasis)

Real-time Minecraft world generation on Tenstorrent Blackhole using the Oasis 500M diffusion transformer. Runs end-to-end inference (DiT denoising, VAE decode, video output) in a single captured trace at 8 FPS. Supports multi-chip 4-way tensor parallelism.

![oasis](doc/oasis-preview.gif)

## [Qwen-Image](https://github.com/zoecarver/qwen-image-tt-xla)

TTNN + TT-Lang implementation of Qwen-Image 20B image generation across 4 Blackhole chips.

**TT-Lang:**
| Resolution | Steps | Time |
|---|---|---|
| 256x256 | 4 | 1.1s |
| 256x256 | 20 | 5.3s |
| 512x512 | 60 | 37.7s |
| 1024x1024 | 60 | 146.6s |

**XLA:**
| Resolution | CFG | Steps | Per-step | Total |
|---|---|---|---|---|
| 256x256 | 4.0 | 15 | 1.75s | 28s |
| 256x256 | 1.0 | 15 | 1.04s | 18s |
| 512x512 | 4.0 | 20 | 5.42s | 112s |

Normalized per-step, TT-Lang is ~4-7x faster at 256x256 and ~8.6x faster at 512x512.

![qwen-image](doc/qwen-image-preview.png)

## [Micelle MD](micelle-demo/)

Cell-list molecular dynamics on Tenstorrent hardware using TT-Lang. Full Ewald electrostatics with LJ short-range forces, periodic boundary conditions, and on-device Verlet integration. Validated at 10K atoms, 10K steps, 1.1ms/step.

![micelle](doc/micelle-preview.gif)

## [Diamond](https://github.com/zoecarver/diamond)

UNet-based diffusion world model ([DIAMOND](https://diamond-wm.github.io), NeurIPS 2024) running on a single Blackhole card. Generates Atari game frames autoregressively using a 4-level encoder/decoder with 3 Euler denoising steps per frame. Runs at ~14 FPS, with interactive browser play across 26 Atari games.

![diamond](doc/diamond-preview.png)

## [LingBot-World](https://github.com/zoecarver/lingbot-world) (WIP)

Video generation from the [LingBot-World-Fast](https://huggingface.co/robbyant/lingbot-world-fast) 14B DiT model on a 4-chip QuietBox with tensor parallelism. Generates 480x832 video with camera pose conditioning at 0.47 fps. TT-Lang kernels cover 3D RoPE and AdaLN broadcast modulation.

![lingbot-world](doc/lingbot-world-preview.gif)

## [Toy World Model](https://github.com/zoecarver/toy-wm)

A Pong world model based on a diffusion transformer, trained on 9 hours of gameplay, running interactively on a single Blackhole card.

![toy-wm](doc/toy-wm-preview.gif)
