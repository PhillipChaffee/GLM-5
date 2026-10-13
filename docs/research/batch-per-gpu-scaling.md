# Batch-per-GPU scaling: the published answer for M=4 to M=8 at hidden-4096 shapes (ticket #60)

Written by the researcher-mid agent on 2026-10-12, in the uncommitted worktree
`/Users/phillipchaffee/worktrees/GLM-5/60-batch-scaling` (branch `60-batch-scaling`,
forked from `origin/main` at `922dcf3`). Nothing in this file is committed; the
document is the only artifact. Every number below is labeled MEASURED (from a
primary source or the team's pipeline), GUIDANCE (a vendor's recommendation), or
DERIVED (arithmetic on pinned numbers, shown so it can be checked).

## Bottom line

- **Expected gain from M=4 to M=8 (batch-per-GPU) at these shapes: roughly 1-3% of
  step time, with honest bounds of ~0% to ~8%.** M=4 to M=16 roughly doubles that
  again: ~2-6% (bounds ~0% to ~15%), and M=16 is memory-infeasible anyway (see
  below). This is guidance-grade, not a measured curve: no published benchmark
  plots throughput vs batch-per-GPU at 4/8/16 for hidden-4096 dense training, and
  the team's own M=8 point does not exist yet because M=8 does not fit in 178 GiB
  without the loss-chunking patch.
- Why the gain is small, in three pinned steps: (1) the GEMM M dimension that
  cuBLAS sees is token positions, not rows - M=4 rows at up to 4096 tokens is up
  to 16,384 tokens per pass, ~130x above NVIDIA's published memory-bound cliff of
  ~128 tokens for hidden-4096 GEMMs, so both M=4 and M=8 are deep in the
  compute-bound regime where cuBLAS efficiency is nearly flat
  (<https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html>);
  (2) the remaining M-sensitivity is tile/wave quantization, which is ~1-2% of
  GEMM time at 16k tokens and shrinks as M grows (DERIVED from NVIDIA's published
  tile-size and wave-quantization data,
  <https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html>);
  (3) whatever M=8 saves therefore comes from amortizing per-pass fixed costs
  (grad-accum chain bookkeeping halves at 16 passes instead of 32, per-pass
  dispatch and memory setup), and the measured launch/CPU-overhead literature
  puts those costs in the low single digits of a ~1-second pass at this token
  count (<https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/>,
  <https://developer.nvidia.com/blog/cuda-graphs/>).
- **Decision implication: the ~10-line loss-chunking patch is the right first
  move; the CUDA-13.1 stack bump is not justified by throughput.** Both routes to
  M=8 land in the same few-percent band, and the patch is orders of magnitude
  cheaper than a toolchain migration. The stack bump buys one real thing - memory
  capacity: B300 (Blackwell Ultra) has ~50-60% more HBM than the team's B200s
  (288 GB vs 178.34 GiB visible) at the same 8 TB/s per GPU and the **same dense
  BF16/FP8 tensor FLOPs class** (NVIDIA's own GB200 vs GB300 NVL72 tables:
  FP8/FP6 720 PF sparse and FP16/BF16 360 PF sparse on both; only dense FP4 moves,
  720 to 1080 PF, and attention kernels claim 2x -
  <https://www.nvidia.com/en-us/data-center/gb300-nvl72/>,
  <https://www.nvidia.com/en-us/data-center/gb200-nvl72/>). For a bf16 training
  loop, B300 is a memory upgrade, not a math upgrade, and it costs ~14% more per
  GPU-hour on Modal ($7.10 vs $6.25, <https://modal.com/pricing>). Move stacks
  only if memory headroom (bigger M without a patch, larger draft blocks, or
  FP4-path experiments) is worth the toolchain bump - not for the M=8 throughput.

## Measured context (team pipeline, taken as given, not re-derived)

- MEASURED: 31.8 s per optimizer step, compiled, bf16, at M=4 batch-per-GPU
  across 4 GPUs; 5-layer 1.1B dense drafter, hidden size 4096, block size 7 draft
  positions; each optimizer step consumes 512 rows (4 rows/GPU x 32 chained
  micro-passes x 4 GPUs), each row at most 4096 tokens; gradient accumulation 32.
- MEASURED: memory peaks near ~150-166 GiB of the 178.34 GiB visible per B200.
- DERIVED (sanity check, assumption flagged): if rows average near 4096 tokens,
  per-GPU throughput is 128 rows x 4096 tokens / 31.8 s = ~16.5k tokens/s/GPU;
  at ~7.6e9 FLOPs/token (6 x 1.1e9 parameters plus the seq-4096 attention
  quadratic term, x3 for forward+backward) that is ~125 TFLOP/s sustained, ~5% of
  B200 dense bf16 (2.5 PF/GPU, DERIVED by dividing the GB200 NVL72 table's
  360 PF sparse bf16 by 72). A step at ~5% of peak with 16k-token GEMMs is not
  GEMM-limited; memory-streaming ops and fixed per-pass costs dominate. The row
  length distribution is unpublished in the ticket, so this utilization figure is
  approximate - but shorter rows would only lower utilization, so the conclusion
  (step not GEMM-bound, M=8 gain bounded by non-GEMM time) is robust to it.

## 1. GEMM M-dimension efficiency at hidden-4096 shapes

- The GEMM M that matters is token positions per pass: 4 rows x up to 4096 tokens
  = up to 16,384 at M=4; 32,768 at M=8. With hidden 4096 and a Llama-class FFN of
  11,008 (the parameterization NVIDIA's own Transformer Engine benchmark uses for
  hidden 4096: ffn_hidden_size 11008, <https://arxiv.org/abs/2402.13499>, Table
  II), the shapes are fc1 M=16,384 x K=4096 x N=11,008 and fc2 M=16,384 x
  K=11,008 x N=4096, per micro-pass.
- Arithmetic intensity at M=16,384 (NVIDIA's formula,
  <https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html>):
  ~1,820 FLOPS/B for fc2 and ~2,530 FLOPS/B for fc1 (DERIVED) against a B200-class
  ops:byte ratio of ~280-310 (2.5 PF dense bf16 / 8 TB/s, DERIVED from the GB200
  NVL72 spec table). That is 6-9x above the memory-bound threshold; doubling M to
  32,768 only raises intensity further. NVIDIA's measured A100 data shows GEMM
  throughput rising with the M-N footprint but with duration growing slower than
  size - the efficiency curve saturates well before 16k tokens.
- Where the "tiny batch cliff" actually sits: NVIDIA's fully-connected guide
  states a 4096x4096 FC layer is memory-bound below token-batch 128 on A100 and
  "will always be bandwidth-limited for very small batch sizes, for example, 8
  and below" - that is 8 TOKENS, not 8 rows
  (<https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html>,
  Section 3.2). 16,384 tokens per pass is ~130x above that cliff. The scaling
  book (Google DeepMind/JAX authors) independently publishes the same framing:
  per-device batches above ~240 tokens are compute-bound on TPUv5p-class hardware
  (<https://jax-ml.github.io/scaling-book/training/>); 16,384 is ~68x that.
- Remaining M-sensitivity is tile and wave quantization (MEASURED on A100 by
  NVIDIA, DERIVED for this workload): cuBLAS uses tiles up to 256x128 ("most
  efficient"); at 16,384-token M with 256x128 tiles, fc2 generates 2,048 tiles =
  ~19 waves on a 108-SM A100, so a worst-case tail wave costs at most ~5% and
  typically ~1-2%; at 32,768 tokens it is ~38 waves, tail at most ~2.6%. cuBLAS
  heuristics also deliberately pick narrower tiles to damp quantization. Net
  per-GEMM efficiency delta from M=4 rows to M=8 rows: ~1-3% of GEMM time.
- Instruction-level corroboration: on Hopper, warp-group MMA (the path cuBLAS/
  CUTLASS use) reaches >=95% of theoretical peak whenever the per-instruction N
  is >=64, independent of the problem's M at these scales
  (<https://arxiv.org/abs/2402.13499>, Tables VIII and X).
- NOT PINNED: a published cuBLAS bf16 GEMM sweep vs M on B200 at K=4096,
  N in {4096, 11008}. No Blackwell-dissection paper exists on arXiv as of
  October 2026 (verified via arXiv search - the phrase "Benchmarking and
  Dissecting" returns only the Hopper paper). Closeable locally in ~30 minutes
  with the CUTLASS profiler or a cublasLt benchmark on the team's B200s.

## 2. Published throughput-vs-batch-per-GPU scaling curves

- NOT PINNED: any published measured curve with points at batch-per-GPU 4 and 8
  for dense hidden-4096 transformer training. Checked NVIDIA's Deep Learning
  Performance documentation, HuggingFace's scaling guides
  (<https://huggingface.co/docs/transformers/en/perf_train_gpu_many>, which
  covers parallelism strategy but carries no batch-size curve), and arXiv. The
  measured side of this curve must come from the team's own A/B once the chunking
  patch lands.
- GUIDANCE, the curve's published shape: NVIDIA's performance docs frame batch
  size as the lever for moving ops from memory-bound to math-bound, with
  diminishing returns once math-bound ("larger GEMMs achieve higher throughput";
  choose sizes divisible by 64-256 to tame quantization)
  (<https://docs.nvidia.com/deeplearning/performance/dl-performance-getting-started/index.html>).
  The gains concentrate below the cliff; above it the curve is near-flat.
- MEASURED, nearest published points: PyTorch's MLPerf v1.0 measurements show
  what per-GPU batch does to end-to-end step time at the two ends of the curve -
  Mask R-CNN at per-GPU batch 1 was CPU-dominated (GPU idle between kernels;
  graphing the backbone cut it 31 ms to 6 ms, 5.1x, 1.70x end-to-end), while
  large-batch BERT saw only 1.12x from the same treatment
  (<https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/>). The
  team's M=4 x 4096 tokens sits on the BERT side of that contrast.
- MEASURED, vendor-side confirmation that small-batch GEMM is a recognized
  inefficiency class: the cuBLAS 12.5 release notes ship "Performance improvement
  to matrix multiplication targeting large language models, specifically for
  small batch sizes on Hopper GPUs"
  (<https://docs.nvidia.com/cuda/archive/12.9.0/cuda-toolkit-release-notes/index.html>,
  cuBLAS section) - the regime being patched is the one the team's shapes are
  already past.

## 3. Hidden overheads: kernel launch, scheduler, and memory streaming

- MEASURED (V100 microbenchmark, NVIDIA CUDA Graphs blog): a 2.9 us kernel costs
  9.6 us per kernel when each launch is synchronized, 3.8 us with overlapped
  launches, and 3.4 us replayed through a CUDA graph - i.e., exposed per-launch
  overhead spans ~0.5-6.7 us depending on synchronization structure
  (<https://developer.nvidia.com/blog/cuda-graphs/>).
- MEASURED (end-to-end, PyTorch MLPerf v1.0): at per-GPU batch 1, CPU-side work
  can exceed GPU work entirely ("For large batch sizes, CPU overhead is a
  negligible percentage of total run time cost, but at small batch sizes CPU
  overhead can become larger than GPU run time",
  <https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/>).
- DERIVED, applied here: one optimizer step is 31.8 s over 32 micro-passes, so
  ~994 ms per pass. Even a generous 1,000 kernels per pass at ~2 us unamortized
  CPU launch cost each is ~2 ms, ~0.2% of a pass - launch overhead is not the
  dominant per-pass cost at 16k-token passes. The amortizable overheads are
  instead the grad-accum chain bookkeeping (halves from 32 to 16 passes), per-pass
  data dispatch and memory setup, and per-step fixed costs (optimizer, gradient
  all-reduce - unchanged by M, since the all-reduce size is the parameter set).
  Their combined fraction of the 31.8 s step is NOT PINNED; a profiler pass-gap
  measurement would pin it and would replace the 1-3% verdict bound with a number.
- Memory streaming (GUIDANCE): elementwise/reduction ops (norms, residuals,
  softmax, dropout) have arithmetic intensity below ~10 FLOPS/B and stay
  memory-limited regardless of batch
  (<https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html>,
  Table 1). They scale with tokens, so doubling M doubles their work too - they
  contribute amortization savings only through their fixed per-pass launch/setup
  component, which the numbers above put in the low single digits.

## 4. B300 vs B200: exact spec deltas and the toolchain requirement

| Spec (per GPU unless noted) | B200-class | B300 (Blackwell Ultra) | Source, status |
|---|---|---|---|
| HBM capacity | 180 GB (DGX B200: 1,440 GB / 8) | 288 GB (Modal: 8xB300 = 2,304 GB; GB300 NVL72: 20 TB / 72 GPUs) | MEASURED-vs-published gap: the team's B200s show 178.34 GiB visible = ~191.5 GB, i.e. a 192 GB-class SKU (MEASURED in pipeline); NVIDIA's DGX page publishes 180 GB; GB200 NVL72 table gives 13.4 TB / 72 = ~186 GB. SKU variance NOT PINNED. |
| HBM bandwidth | 8 TB/s (DGX B200: 64 TB/s total / 8) | 8 TB/s (GB300 NVL72: 576 TB/s / 72) | MEASURED: identical class |
| FP8/FP6 dense | 5 PF/GPU (720 PF sparse per 72-GPU rack / 72 / 2) | same (720 PF sparse per GB300 rack) | MEASURED from NVIDIA spec tables; dense value DERIVED |
| FP16/BF16 dense | 2.5 PF/GPU (360 PF sparse / 72 / 2) | same (360 PF sparse) | MEASURED from NVIDIA spec tables; the decisive row for bf16 training |
| FP4 dense | 10 PF/GPU (720 PF "without sparsity" / 72) | 15 PF/GPU (1080 PF / 72) = 1.5x | MEASURED (GB300 NVL72 footnote marks 1080 as without sparsity); DGX pages agree: 72 to 108 PF dense per 8-GPU box |
| Attention kernels | - | "2x higher attention performance" | GUIDANCE (vendor workload-class claim, not an isolated published benchmark) |
| CUDA toolchain (silicon minimum) | CUDA 12.8 adds Blackwell support (sm_100); cuBLAS Blackwell kernels land in 12.8 | CUDA 12.9 adds sm_103/sm_103f/sm_103a compiler target (Blackwell Ultra) | MEASURED from CUDA 12.9 release notes |
| CUDA toolchain (Modal platform) | no extra requirement stated | "B300 requires CUDA version 13.1+" | MEASURED from Modal docs - the ticket's "CUDA 13.1+" is Modal's platform policy, stricter than NVIDIA's 12.9 silicon minimum |
| Modal price | $6.25/GPU-hr | $7.10/GPU-hr (1.14x); `B200+` runs either GPU, billed as B200 | MEASURED from <https://modal.com/pricing> |

Sources: <https://www.nvidia.com/en-us/data-center/dgx-b200/>,
<https://www.nvidia.com/en-us/data-center/dgx-b300/>,
<https://www.nvidia.com/en-us/data-center/gb200-nvl72/>,
<https://www.nvidia.com/en-us/data-center/gb300-nvl72/>,
<https://modal.com/docs/guide/gpu>, <https://modal.com/pricing>,
<https://docs.nvidia.com/cuda/archive/12.9.0/cuda-toolkit-release-notes/index.html>.

- DERIVED, the decision-relevant reading: for the team's bf16 training loop,
  moving to B300 changes memory capacity (+50-60%) and nothing about dense
  FLOPs or bandwidth class. Fitting M=8 activations (~2x of the measured
  150-166 GiB, i.e. ~300+ GiB) exceeds both the 178.34 GiB B200 budget and B300's
  ~268 GiB visible (288 GB = 268.2 GiB, DERIVED) without chunking; with the
  chunking patch M=8 fits on B200. M=16 needs ~4x (~600+ GiB) - beyond any
  current single-GPU SXM class (capacities beyond B300 NOT PINNED here).
- Sanity check against the repo's own records: docs/research/modal-cost.md
  already pins the same Modal prices and the CUDA 13.1+ note; this section
  re-verifies them against live sources and adds the NVIDIA-side spec delta.

## 5. torch.compile interaction at small batch

- MEASURED, aggregate: torch.compile "work[s] 93% of time across 165 open-source
  models and runs 20% faster on average at float32 precision and 36% faster on
  average at AMP precision" (TorchBench, PyTorch 2.0 release blog,
  <https://pytorch.org/blog/pytorch-2.0-release/>). These are per-model aggregates,
  not a batch-size curve.
- MEASURED, mechanism: the regime compile/graphs fix is exactly the
  CPU-overhead-dominated small-batch regime - the measured Mask R-CNN (per-GPU
  batch 1) and small-batch DLRM results above
  (<https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/>). At 16k
  tokens per pass, the team's micro-passes are outside that regime, so compile's
  benefit does not materially change between M=4 and M=8.
- MEASURED, shape-specialization behavior: at the 2.0 release, dynamic shapes
  were a prototype and "Dynamic does not work with non-standard modes like
  reduce-overhead or max-autotune"
  (<https://pytorch.org/blog/pytorch-2.0-release/>, Prototype section); compiled
  graphs specialize on shapes, so switching M=4 to M=8 triggers one fresh
  Dynamo/Inductor compilation and autotune cycle, then steady state. The team
  already runs compiled at fixed M=4, so an M=8 experiment pays a one-time
  recompile, not a recurring cost.
- NOT PINNED: any published measurement of Inductor autotuning comparing batch 4
  vs 8, or of compile changing the slope of the batch-scaling curve. Searched
  PyTorch blogs and docs; nothing batch-specific found. A local A/B
  (compile on/off at M=4 and M=8-with-chunking) is the way to close it.

## Open gaps

1. B200-specific cuBLAS bf16 GEMM sweep vs M at K=4096, N in {4096, 11008} -
   close with CUTLASS profiler or cublasLt on the team's B200s (Section 1).
2. The measured per-pass fixed-overhead fraction in the team's pipeline (profiler
   gap between micro-passes) - would turn the 1-3% verdict into a number
   (Section 3).
3. B200 memory SKU variance (180 GB DGX/HGX figure vs the 192 GB-class unit the
   team sees) - pin from the fleet's `nvidia-smi` plus the specific SKU datasheet
   (Section 4).
4. No published throughput-vs-batch-per-GPU points at M=4/8/16 for hidden-4096
   dense training; the measured curve must come from the team's post-patch A/B
   (Section 2).
5. The "2x attention performance" B300 claim and the Modal-vs-NVIDIA CUDA version
   gap (12.9 silicon minimum vs 13.1 platform policy) both deserve one
   confirmation pass before they drive any spend (Section 4).

## Sources

- NVIDIA Deep Learning Performance guides: GPU Background
  <https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html>;
  Matrix Multiplication Background
  <https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html>;
  Linear/Fully-Connected Layers
  <https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html>;
  Getting Started
  <https://docs.nvidia.com/deeplearning/performance/dl-performance-getting-started/index.html>
- NVIDIA CUDA Graphs blog <https://developer.nvidia.com/blog/cuda-graphs/>;
  PyTorch CUDA Graphs blog
  <https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/>; PyTorch 2.0
  release blog <https://pytorch.org/blog/pytorch-2.0-release/>
- Benchmarking and Dissecting the Nvidia Hopper GPU Architecture,
  <https://arxiv.org/abs/2402.13499> (Luo, Fan, Li, Du, Wang, Chu, 2024)
- How To Scale Your Model, Part 5 (Training)
  <https://jax-ml.github.io/scaling-book/training/> (Austin et al., Google
  DeepMind, Feb 2025)
- HuggingFace Transformers, Parallelism methods
  <https://huggingface.co/docs/transformers/en/perf_train_gpu_many>
- NVIDIA DGX B200 <https://www.nvidia.com/en-us/data-center/dgx-b200/>; DGX B300
  <https://www.nvidia.com/en-us/data-center/dgx-b300/>; GB200 NVL72
  <https://www.nvidia.com/en-us/data-center/gb200-nvl72/>; GB300 NVL72
  <https://www.nvidia.com/en-us/data-center/gb300-nvl72/>
- CUDA 12.9 Toolkit Release Notes
  <https://docs.nvidia.com/cuda/archive/12.9.0/cuda-toolkit-release-notes/index.html>
- Modal GPU docs <https://modal.com/docs/guide/gpu>; Modal pricing
  <https://modal.com/pricing>
- Repo cross-check: docs/research/modal-cost.md (worktree copy)
