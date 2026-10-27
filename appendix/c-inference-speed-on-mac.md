# Appendix C · Inference speed on a Mac

> On a Mac, a small policy is often faster on the CPU than on the GPU: the GPU pays a fixed launch-and-sync cost that only large batches or large models amortize.

PyTorch's MPS backend runs tensors on the Apple GPU through Metal Performance Shaders; it needs macOS 14.0+ and is selected with `torch.device("mps")` ([PyTorch MPS notes](https://docs.pytorch.org/docs/2.14/notes/mps.html)). For a robot the relevant number is the latency of one forward pass at batch 1, because the control loop asks for one action (or one chunk) at a time.

**Measurement.** Two policies of the size Part 7 trains: an MLP (64 → 256 → 256 → 6, 83,974 parameters) and a small CNN on a 3 × 96 × 96 image that outputs a 50 × 6 chunk (352,492 parameters). Median (min) of 50 forward passes after 10 warm-up passes, `torch.no_grad()`, MPS timings include `torch.mps.synchronize()`. PyTorch 2.14.1, Python 3.14, Apple M5, macOS 26.3.1, 4 CPU threads. Script: [`experiments/appendix-c-inference-speed/experiment.py`](../experiments/appendix-c-inference-speed/experiment.py), output: [`output.txt`](../experiments/appendix-c-inference-speed/output.txt).

**The machine was heavily loaded** (many other jobs running in parallel) during this run. These numbers show the shape of the trade-off, not a benchmark; rerun on an idle machine before quoting them.

| policy | batch | CPU [ms] | MPS [ms] |
|---|---|---|---|
| MLP | 1 | 0.010 (0.010) | 0.344 (0.274) |
| MLP | 8 | 0.022 (0.021) | 0.338 (0.282) |
| MLP | 64 | 0.039 (0.034) | 0.433 (0.312) |
| MLP | 256 | 0.290 (0.131) | 0.439 (0.407) |
| CNN | 1 | 0.534 (0.408) | 1.159 (0.979) |
| CNN | 8 | 3.722 (2.748) | 2.714 (2.408) |
| CNN | 64 | 473.753 (278.449) | 4.280 (3.333) |
| CNN | 256 | 1894.905 (1054.068) | 11.946 (11.118) |

**Reading it.** The MLP is faster on the CPU at every batch size we tried; MPS stays between 0.338 and 0.439 ms (median), roughly its fixed cost per call. The CNN at batch 1 is also faster on the CPU (0.534 vs 1.159 ms); from batch 8 on, MPS wins. Do not read the CPU rows at batch 64 and 256 as a scaling law: 8× more work than batch 8 took 127× longer. Competing processes are one suspect, but a quick rerun with a single CPU thread showed a similar jump above batch 8, so a slower CPU convolution path for larger batches is also possible; we did not isolate the cause. It does not affect batch-1 control.

**For the control loop.** At 30 Hz the budget is 33 ms per step; both policies fit on either device at batch 1. For a VLA the per-call cost is far larger (π0: 73 ms per chunk on an RTX 4090, [arXiv 2410.24164](https://arxiv.org/abs/2410.24164)), which is why chunking and asynchronous inference ([SmolVLA, arXiv 2506.01844](https://arxiv.org/abs/2506.01844)) matter more than the device choice. We did not measure a VLA on this Mac.

**Practical rules.** Time the exact model at batch 1 on both devices with synchronization; keep the model resident on one device (moving tensors between CPU and MPS each step adds copies); reuse the measurement script above on an idle machine.
