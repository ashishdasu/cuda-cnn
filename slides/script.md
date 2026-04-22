# Presentation script — CS5330 final project

Target runtime: ~10–12 min, safe inside the 15-min cap. 9 slides.
Per-slide time budgets in `[brackets]`; running total in **bold**.

Delivery notes:
- Speak at a conversational pace, not a recitation pace.
- The bolded phrases are the load-bearing sentences — if you are running
  long, skip the surrounding filler, not the bolded sentence.
- Every slide has one "takeaway sentence" at the end — make sure you
  deliver it.

---

## Slide 1 — Title `[10s]` — **0:10**

> "Hi, I'm Ashish Dasu. This is my CS5330 final project: **a
> from-scratch CUDA forward-pass inference engine for a LeNet-style
> MNIST network** — no PyTorch, no cuDNN at runtime."

(Advance immediately.)

---

## Slide 2 — The network and how we verified it `[75s]` — **1:25**

> "Some context: my Project 5 in this class was a PyTorch LeNet-5
> classifier for MNIST at 97.93% test accuracy. For the final, I
> wanted to answer the question 'what does model.to("cuda") actually
> do?' — so I rebuilt the forward pass of that exact network with
> hand-written CUDA kernels.
>
> The pipeline diagram walks left to right. Input x is a 28×28 digit.
> Two conv-pool-ReLU stages, a flatten, two fully-connected layers,
> log-softmax out. **Five kernels, ten kernel launches.** Activations
> stay on-device between launches — only x copied in, ŷ copied out.
>
> Before I wrote a single optimized kernel, I built a three-layer
> oracle. Layer one: a plain C++ CPU reference for every kernel.
> Layer two: each CUDA kernel diffed against the CPU reference on
> random inputs at 1e-4 tolerance. Layer three: full forward pass vs
> PyTorch-exported log-softmax over all 10,000 MNIST test images.
>
> **The end-to-end result: max absolute error 1.91×10⁻⁵ — 50× under
> the tolerance. Argmax matches PyTorch on every single image.
> Accuracy identical to PyTorch eval mode at 97.93%.**"

---

## Slide 3 — CUDA primer `[60s]` — **2:25**

> "Before the optimizations, a quick grounding in what CUDA is.
> A kernel launch is one program executed by many threads at once.
> Threads are organized into blocks; blocks form a grid. Threads in
> the same block can cooperate through shared memory and synchronize
> with __syncthreads(). Threads across blocks cannot.
>
> The memory hierarchy is what every kernel design decision comes back
> to. **Registers are ~1 cycle. Shared memory is ~20 cycles. Global
> DRAM is ~400 cycles.** The entire talk is one idea: move data from
> slow DRAM into fast shared memory or registers, reuse it, write
> results back. Every optimization I'm about to show is a variation
> on that theme."

---

## Slide 4 — Tiled conv `[75s]` — **3:40**

> "First optimization: shared-memory tiled convolution. The left side
> shows the geometry. One thread block produces one 16×16 output tile.
> Before threads start computing, **all threads cooperatively load the
> input patch they need into on-chip shared memory** — that's the blue
> tile plus an orange halo for the 5×5 filter. Then every thread reads
> from shared memory instead of DRAM to compute its output. The reuse
> factor is K_h × K_w = 25 — each input pixel is used 25 times by the
> filter, so without tiling every reuse is a global-memory load.
>
> The bar chart shows what actually happened at LeNet scale. **The
> naive kernel wins.** The 28×28 and 12×12 working sets already fit in
> L1. Tiling adds cooperative loads and sync barriers but buys no new
> reuse — it's overhead for zero benefit.
>
> **The tile strategy isn't wrong — the regime is wrong.** Next slide
> shows where it pays off."

---

## Slide 5 — Tiled GEMM `[60s]` — **4:40**

> "Same tiling idea applied to the fully-connected layers, which are
> matrix multiplies. At LeNet's tiny fc1 and fc2 shapes, tiled is only
> 1.06× and 1.01× over naive — the same regime story.
>
> But if I sweep to a realistic size — fixed 1024×1024 matrix, batch
> size varying — the picture flips entirely. At batch 1, tiled is
> already **3.8× faster** because the weight tile is being reused
> across rows. By batch 1024, **6.4× faster.**
>
> Same kernel, same strategy, huge difference — because there's finally
> enough data in flight to amortize the shared-memory load.
> **The regime is what decides whether tiling pays off.**"

---

## Slide 6 — Kernel fusion `[75s]` — **5:55**

> "Third optimization. At LeNet scale, three back-to-back kernels —
> conv, ReLU, pool — spend more time on **launch overhead and DRAM
> round-trips than on arithmetic itself.**
>
> The top row of the diagram shows the unfused path: conv writes the
> full activation to DRAM, ReLU reads it back and writes again, pool
> reads and writes once more. Three kernels, two activation-sized DRAM
> round-trips between them.
>
> The bottom row: **one fused kernel.** Input in, pool output out. The
> intermediate values never leave shared memory and registers. Each
> thread owns four conv outputs — one per 2×2 pool window. Four
> accumulators live in registers for the whole input-channel loop, then
> ReLU in place, then take the 2×2 max. One DRAM write for four conv
> outputs.
>
> Results: **conv1 at batch 1, naive is 1.76× slower than fused.**
> Launch overhead was binding. But conv2 at batch 1, fused is 0.40×
> naive — only 320 threads launching on 128 SMs, the GPU is starved.
> **Fusion is a latency trade, not a free win.**"

---

## Slide 7 — Occupancy + cuDNN `[65s]` — **7:00**

> "Two things on this slide. First: resource usage. Nsight Compute
> hardware counters were blocked on the pod — unprivileged container,
> driver parameter locked. But **static kernel characterization doesn't
> need counters.** Compiling with nvcc -Xptxas=-v gives register and
> shared-memory usage at build time. The table shows: **every compute
> kernel hits 100% theoretical occupancy.** No spills. The
> naive-versus-tiled delta at LeNet scale is not an occupancy story —
> it's a memory-access-pattern story.
>
> Second: the cuDNN reference. At LeNet batch-1, **our naive direct
> convolution actually beats cuDNN** — because cuDNN pays descriptor
> and algorithm-dispatch overhead on every launch. One numerical gotcha:
> cuDNN on Ada Lovelace defaults to TF32 tensor cores. Max absolute
> error against my CPU reference was 5.3×10⁻³ — 50× over my tolerance,
> silently. Fixed by pinning CUDNN_FMA_MATH; error drops to 2×10⁻⁶."

---

## Slide 8 — Demo `[40s]` — **7:40**

> "Demo of the thing working. Left side: build/predict on MNIST test
> image 42, tiled variant. Runs in 31 microseconds. Predicts class 7,
> correct.
>
> Right side: all 11 kernel-level tests pass — CPU reference, every
> conv variant, pooling, ReLU, linear, log-softmax, end-to-end.
>
> Plus the 10k-image parity harness: **max absolute error 1.91×10⁻⁵
> across every image in MNIST test, argmax matches PyTorch on every
> single image, accuracy identical to PyTorch eval mode.
> The thing is correct and it's fast.**"

---

## Slide 9 — Discussion + Reflection `[60s]` — **8:40**

> "Four takeaways.
>
> **One: tiling is not a default-on optimization.** Same tile strategy
> hurts at LeNet, wins 6.4× on large GEMM. Regime matters.
>
> **Two: launch overhead is real and measurable.** Fusing three
> launches buys 1.76× on conv1 but starves the SMs on conv2. Same
> code, opposite outcome depending on grid size.
>
> **Three: FP32 parity with cuDNN requires opt-ins.** TF32 and Winograd
> are both on by default on Ada Lovelace and both break a 10⁻⁴
> tolerance silently.
>
> **Four: the correctness oracle chain paid off.** Eleven per-kernel
> tests plus 10k end-to-end parity meant I chased throughput without
> any 'the logits drifted overnight' debugging.
>
> Reflection: the biggest lesson was that 'the optimization' and
> 'the speedup' are separate things. Every shared-memory or fusion
> change was conceptually a win, and three of them empirically made
> LeNet slower. Writing the CPU reference first and keeping parity
> green on every commit made it painless to find out.
>
> Code is at github.com/ashishdasu/cuda-cnn. Thanks for watching."

---

# Pacing contingencies

**If running long (hit slide 7 at 7:30 instead of 7:00):**
- Compress slide 7: skip the Nsight driver detail, just say "static
  analysis via ptxas -v shows every compute kernel at 100% occupancy."
- Compress slide 9: deliver only takeaways 1 and 4.

**If running short (hit slide 7 at 5:30):**
- Add to slide 6: walk through why four accumulators and not one —
  it's the receptive-field / pool-stride arithmetic.
- Add to slide 7: note that cuDNN's algorithm selector picks Winograd
  by default, and the Winograd transform itself is a separate 10⁻³
  error term on top of TF32.
