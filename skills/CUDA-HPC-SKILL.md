# CUDA HPC Skill — Overview, Expert Concepts, Practical Applications

## Name
CUDA-HPC-Overview

## Description
Workspace-scoped skill that produces a concise system overview, a set of expert-level CUDA programming concepts, and practical guidance for applying CUDA in high-performance computing (HPC) environments. Designed to be used as a prompt template for code review, optimization, and teaching tasks.

## Intent / Outcome
Given a request like "overview of the system, expert-level CUDA coding concepts, and practical application on HPC", this skill returns:
- A short system overview (GPU architecture, memory hierarchy, execution model).
- A curated list of advanced CUDA techniques with when/how to apply them.
- Practical HPC patterns (multi-GPU, MPI+CUDA, NUMA/PCIe considerations, profiling and scaling).
- Concrete prompts and checks the user can run to validate quality.

## Scope
Workspace-scoped — tuned for projects that build and optimize CUDA kernels, GPU runtimes, or MPI+CUDA workloads on HPC clusters.

## Step-by-step process (workflow)
1. Gather context: determine target GPU(s), CUDA version, OS, compiler, build flags, and runtime (MPI, SLURM, container).
2. Produce system overview: summarize GPU SM count, memory sizes, PCIe topology, and relevant drivers.
3. Recommend primary performance levers: memory strategy, compute utilization, concurrency, precision choices.
4. Suggest code-level changes or experiments: kernel tiling, memory coalescing, shared-memory tiling, loop unrolling, use of intrinsics and warp-level ops.
5. Outline scaling strategy for HPC: domain decomposition, CUDA-aware MPI, load balancing, and interconnect tuning.
6. Produce verification checklist: microbenchmarks, regression tests, and profiler traces to collect.
7. Provide example prompts or code snippets and follow-up questions to refine the recommendations.

## Decision points and branching logic
- Single-GPU vs multi-GPU: if multi-GPU, prefer communication-avoiding algorithms, overlap compute with data transfer, and use NCCL or CUDA-aware MPI.
- Memory-bound vs compute-bound: detect by roofline or profiler; prioritize bandwidth reductions for memory-bound, and instruction-level optimizations for compute-bound.
- Precision tradeoffs: use FP16/mixed precision or tensor cores when accuracy allows; otherwise keep FP32/FP64 as required by the algorithm.
- Small kernels with low occupancy: consider batching inputs, kernel fusion, or cooperative groups to increase occupancy and reduce kernel launch overhead.

## Expert-level CUDA concepts (what the skill explains and when to apply)
- Memory hierarchy and access patterns: global memory coalescing, shared memory tiling, avoiding bank conflicts, and using read-only caches (e.g., __ldg) when beneficial.
- Streams and concurrency: overlapping H2D/D2H transfers with kernels using multiple streams; per-GPU stream pools and stream priorities for latency-sensitive tasks.
- Asynchronous and unified memory: when to use `cudaMemcpyAsync` and pinned host memory; pros/cons of unified memory (simplicity vs explicit control and potential page faults).
- Warp-level programming: using ballot, shfl, and warp intrinsics to reduce synchronization and improve branch divergence behavior.
- Occupancy and launch configuration: computing optimal block/grid sizes, register pressure vs occupancy tradeoffs, and using `cudaFuncSetAttribute` to tune.
- Tensor Cores and mixed precision: algorithms for FP16/FP32 accumulation, loss scaling strategies, and when tensor cores provide speedups.
- Kernel fusion and loop unrolling: reduce global memory traffic and kernel launch overhead by fusing small kernels and using template metaprogramming for compile-time unrolling.
- Memory allocators and pooling: use device memory pools, CUB caching allocator, or custom arenas to reduce allocation churn and fragmentation.
- Cooperative groups / CUDA Dynamic Parallelism: use for algorithms needing fine-grained synchronization or nested parallelism (only where supported by the target SM capability).
- Profiling-driven optimization: NVTX annotations, Nsight Systems for system-wide traces, Nsight Compute for kernel-level metrics, CUPTI for programmatic access.

## Practical HPC applications and patterns
- MPI + CUDA integration: recommend CUDA-aware MPI or GPUDirect when available; overlap communication with local compute; use non-blocking MPI calls with GPU buffers.
- Multi-GPU tiling and domain decomposition: prefer partitioning that minimizes surface-to-volume ratio and communication volume; use halo exchange patterns with pinned buffers and GPUs dedicated to communication.
- NUMA and CPU affinity: pin CPU threads to the CPU socket nearest the GPU (NUMA-aware placement); verify PCIe and NVLINK topology and set CPU/GPU affinity accordingly.
- I/O and checkpointing: use parallel I/O libraries, overlapping disk I/O with computation, and compress/aggregate checkpoints on GPUs when possible.
- Job schedulers: SLURM and CUDA_VISIBLE_DEVICES orchestration; use job prologs to set GPU persistence and environment for reproducible runs.
- Reproducibility & determinism: control RNG seeds, deterministic cuBLAS/cuDNN flags where available, and document hardware/software stack.

## Quality criteria / completion checks
- Kernel-level metrics: achieved occupancy, memory throughput vs theoretical, achieved SM utilization, and achieved FLOPS vs theoretical (roofline comparison).
- End-to-end checks: wall-clock scaling across nodes, communication overhead fraction, and accuracy/regression tests for numerical results.
- Profiling artifacts: Nsight Systems trace, Nsight Compute report for hot kernels, and NVTX-marked timeline for overlapping phases.

## Example prompts to run this skill
- "Summarize the GPU and system topology for a job running on 4 A100s with CUDA 12 and Mellanox HDR." 
- "Given this kernel (paste code), suggest tiling and memory changes to reduce global memory traffic." 
- "I see low achieved FLOPS for this compute-bound kernel—what register/loop changes would you try first?" 

## Suggested follow-ups and clarifying questions
- What GPU models, CUDA version, and OS are you targeting?
- Is the workload single-node or multi-node? Does it use MPI or NCCL?
- What accuracy constraints (FP64/FP32/FP16) exist for the algorithm?
- Can you provide a profiler capture or a short kernel reproducer?

## Example outputs
- System overview paragraph summarizing topology and limits.
- 5 targeted code-level optimization suggestions prioritized by expected win and risk.
- A minimal microbenchmark plan (bandwidth, shared-mem latency, kernel latency).

## Maintenance and improvements
- Keep lists of target GPUs and best-practice flags up to date for new CUDA/toolkit releases.
- Add sample kernel before/after snippets over time to the skill for faster remediation.

---

If you'd like, I can do one of the following: adjust depth for a beginner or expert audience; add concrete code snippets and microbenchmarks; or create a short README.md showing how to use this skill in prompts. Which would you prefer?