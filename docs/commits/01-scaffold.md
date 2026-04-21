# 01 — Scaffold

First commit. No source code yet — this lays down the directory skeleton,
the build configuration, the project-wide documentation, and the report
template so that every subsequent commit has a clean place to land.

## What's in this commit

| Path | Purpose |
|---|---|
| `CLAUDE.md` | Project-wide guidance: prime directive, rubric, coding standards, commit workflow. Persistent across sessions. Read this first. |
| `readme.md` | Rubric-required README. Name, solo status, project description (same paragraph used in the check-in email), build instructions, demo-video placeholder. |
| `.gitignore` | Keeps LaTeX build artifacts (`.aux`, `.log`, `.out`, `.synctex.gz`, etc.), CUDA intermediates (`.cubin`, `.ptx`), Python caches, macOS junk, and the frozen proposal `.tex` out of history. |
| `CMakeLists.txt` | Build configuration. Compiles `kernels/*.cu` + `host/*.cpp` into a static library `cnn_core`, then links it into the `infer` executable and five unit-test executables. Targets common NVIDIA arches (75/80/86/89). Does **not** build on macOS. |
| `proposal/final-project-proposal_cs5330_adasu.pdf` | The submitted, approved proposal. Frozen artifact — PDF only, the `.tex` is gitignored. |
| `report/report.tex` | The actual final report, starting as a copy of the IEEE conference template. This is the file you will edit. |
| `report/IEEEtran.cls` | IEEE class file. Required by `report.tex` at compile time; do not modify. |
| `report/fig1.png` | Placeholder figure the template references. Gets replaced with real figures or removed before submission. |

## What's intentionally **not** here

- Any `.cu` or `.cpp` source file. Kernels land in later commits, one logical change per commit.
- The weight export tool (`tools/export_weights.py`) — staged as its own commit so the history is legible.
- `IEEEtran_HOWTO.pdf` and the template's sample-rendered PDF — these were documentation for us, not code.
- The proposal `.tex` — it's a frozen, already-submitted document. Keeping only the PDF avoids rebuild drift.

## How to verify this commit

1. **Tree layout** — `ls` at the repo root should show:
   ```
   CLAUDE.md  CMakeLists.txt  docs/  fixtures/  host/  kernels/
   proposal/  readme.md  report/  tests/  tools/  weights/
   ```
   Code subdirs are empty on purpose.

2. **README compiles from rubric** — open `readme.md` and check that it has:
   - your name and group status (solo),
   - the project-description paragraph,
   - a demo-video URL slot (currently `TODO`),
   - build/run instructions.

3. **Report template compiles** — on any machine with `pdflatex`:
   ```
   cd report && pdflatex report.tex && pdflatex report.tex
   ```
   Two passes are required (IEEEtran resolves cross-references on the second
   pass). If it compiles and produces `report.pdf`, the template wiring is
   correct and the build environment is ready for you to start writing.

4. **CMake syntax is valid** — you cannot actually configure on macOS (no
   `nvcc`), but once you're on the RunPod box:
   ```
   cmake -B build -S .
   ```
   should succeed and report the target list. Compile errors are expected
   until kernel source files land in later commits.

5. **Gitignore is doing its job** — after building the proposal or report,
   `git status` should show nothing. If `.aux` / `.log` / `.out` files
   appear as untracked, the gitignore rule is wrong and should be fixed
   before the next commit.

## What this commit unblocks

- **Next commit (02)**: `tools/export_weights.py` — once the scaffold is in,
  the weight export tool has an obvious home (`tools/`) and its output has
  an obvious destination (`weights/`).
- **All CPU reference + kernel work** now has defined source directories
  and a build system expecting specific filenames (see `CMakeLists.txt`
  target list).

## Reading order for reviewers

1. `readme.md` — what the project is.
2. `CLAUDE.md` — how the project is run (standards, tiers, deliverables).
3. `CMakeLists.txt` — how the project builds.
4. `report/report.tex` — where the writeup will live.
