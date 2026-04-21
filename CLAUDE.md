# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Prime Directive

**The author is targeting full marks on this final project.** The course handout (reproduced under "Deliverables" below) is gospel — every bullet there is a grading hook. When a tradeoff comes up between "more interesting" and "checks the rubric box," check the box first, then do the interesting thing if time allows. The proposal's three scope tiers (minimum / target / stretch) exist so that the minimum tier is always submittable even if later tiers slip — **secure the minimum tier before touching the target tier**, and don't leave half-finished kernels behind.

## Hard Deadline

- Materials due **Thursday 2026-04-23**; Gradescope cutoff **Friday 2026-04-24 at 02:59**.
- No time-travel days are allowed on this project (course-wide rule).
- The April 13 group/progress check-in email to Prof. Maxwell is a separate deliverable; confirm it was sent before assuming it's done.

## Repository Status

Scaffolded but pre-implementation. On-disk layout:
- `proposal/` — the submitted proposal PDF (source `.tex` lives here too but is gitignored; it is a frozen artifact, not something we rebuild).
- `report/` — the IEEE-format final report. `report.tex` is the actual report (starts as a copy of the IEEE template), `IEEEtran.cls` is the required class file, `fig1.png` is a placeholder figure from the template that will be overwritten or removed before submission.
- `kernels/ host/ tests/ tools/ weights/ fixtures/` — empty, to be populated per the plan below.
- `CMakeLists.txt`, `.gitignore`, `readme.md` — build config + repo hygiene.

No source code yet. When kernels land, replace the planning notes below with real build/run/test commands.

## Project Plan (from the proposal)

A from-scratch forward-pass CNN inference engine in C++/CUDA — **no PyTorch, no cuDNN, no DL framework at runtime**. Target network is the LeNet-style MNIST model from the author's Project 5. Weights are exported from PyTorch once as raw binary floats, then `cudaMemcpy`'d to the GPU; inference at runtime is pure CUDA.

Kernels to implement (in this order — each is unit-tested against a hand-computed reference before integration):

1. **Convolution** — naive first for correctness, then tiled shared-memory version.
2. **GEMM** — tiled shared-memory matmul for fully connected layers.
3. **Elementwise** — ReLU, softmax (**softmax must use the max-subtraction trick**; naive softmax overflows).
4. **Max pooling** — one thread per output window.

**Scope tiers** (project is submittable at minimum even if later tiers slip):
- Minimum: correct forward pass with naive kernels matching PyTorch logits on the MNIST test set.
- Target: tiled shared-memory optimization for conv + GEMM.
- Stretch: close the gap to cuDNN on convolution throughput.

**Known correctness traps** called out in the proposal:
- Softmax numerical stability (subtract max before exp).
- Bank conflicts and boundary handling in the shared-memory convolution.
- Padding conventions must match PyTorch exactly or end-to-end logits diverge.

## Environment

- Dev target: **Google Colab, NVIDIA T4 GPU**. Local machine is macOS (no CUDA) — kernels cannot be compiled or run locally. Any code change must be tested in Colab before being called done.
- Toolchain: CUDA Toolkit, C++17, CMake, Nsight Compute for profiling.
- Python is used **only** for one-shot weight export from the Project 5 PyTorch model; it is not a runtime dependency of the inference engine.

## Deliverables (course handout, verbatim-relevant parts)

All submitted to Gradescope with every group member added. No videos or datasets on Gradescope — use Google Drive / YouTube URLs.

- [ ] **Code** — well-commented, functional, logically organized. Graded directly.
- [ ] **Report (PDF, IEEE 2-column conference format, ≤ 8 pages)** — use the `IEEE-conference-template-062824/` already in the repo. Required sections, in order:
  1. Title and authors
  2. Abstract (≤ 200 words)
  3. Introduction — task + main concepts at a high level
  4. Related work — **≥ 3 peer-reviewed papers**. CVPR / ICCV / ECCV / ACCV / BMVC / WACV preferred; cite the peer-reviewed venue, not arXiv, even if the PDF comes from arXiv.
  5. Methods — reproducible detail; cite anyone whose work you're following.
  6. Experiments and results — graphs/tables; compare to prior work where possible (cuDNN is the natural baseline per the proposal).
  7. Discussion and summary
  8. References / bibliography
- [ ] **Presentation** — recorded video ≤ 15 min (or live during final class), all group members participate, ≤ 8 slides recommended.
- [ ] **README** — name + group members, the 1-paragraph project description used in the check-in email, URLs for demo video / datasets.
- [ ] **Check-in email (2026-04-13)** — one person emails Prof. Maxwell: 1-paragraph description, group members, progress update.

**Writing**: write the report yourself. AI is allowed only for grammar/fine-tuning, not for generating sections. Keep this in mind when I help draft — I should produce outlines, bullet points, and targeted sentence fixes, not ghostwritten prose.

## Coding Standards (non-negotiable for grading)

- **Every kernel and host function gets a comment block** explaining: what it computes, thread/block layout assumed, memory layout assumed, and any non-obvious correctness invariant. Inline comments only where the why isn't obvious from the code.
- **Logical file layout.** Separate `kernels/` (CUDA), `host/` (driver + weight loader), `tests/` (per-kernel unit tests), `tools/` (Python weight export). Keep headers next to implementations.
- **Unit test each kernel against a hand-computed reference before wiring it into the full forward pass.** This is called out in the proposal as the risk-mitigation strategy — it's also how we avoid end-to-end logit-mismatch debugging at 2am.
- **Numerical parity target**: forward-pass logits must match PyTorch to within a documented tolerance (e.g., 1e-4 absolute on MNIST). Pick a tolerance, write it down, test for it.

## Commit Workflow

The author commits personally. Claude's job:

1. After a logically complete unit of work (one kernel written, one bug fixed, one section of the report drafted), **stage the relevant files and draft a commit message**.
2. Hand the staged state and proposed message to the author; let them run `git commit` themselves.
3. **Never** run `git commit`, `git push`, `git reset --hard`, or any rewriting command without an explicit ask.

**Commit hygiene for a clear history** (the author has asked for this explicitly):

- One logical change per commit. Naive conv, tiled conv, conv unit tests, and conv integration are four commits, not one.
- Imperative mood, ≤ 72-char subject line, optional body explaining the *why* if non-obvious.
- Don't mix code and report commits. Prefix report commits with `report:` for easy filtering.
- Don't commit build artifacts (`.aux`, `.log`, `.out`, `.synctex.gz`, compiled binaries, Colab `.ipynb_checkpoints`). A `.gitignore` should land in the first code commit.

## LaTeX

The final report lives at `report/report.tex`. Build from inside `report/`:
```
cd report && pdflatex report.tex && pdflatex report.tex
```
Two passes because IEEEtran resolves cross-references on the second pass. `IEEEtran.cls` and `fig1.png` must sit alongside `report.tex`. Build artifacts (`.aux`, `.log`, `.out`, `.synctex.gz`, `.toc`, `.bbl`, `.blg`) are gitignored — never commit them.

The proposal PDF at `proposal/final-project-proposal_cs5330_adasu.pdf` is the authoritative submitted artifact; the `.tex` next to it is kept locally for reference but is gitignored (frozen document, no rebuild path).
