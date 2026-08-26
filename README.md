# cuDSS solver — MATLAB wrapper around NVIDIA cuDSS

A factor-once / solve-many sparse direct solver for large non-symmetric
systems on the GPU.  The factorization is held in device memory across
many right-hand-side blocks, so the analysis + numeric-factorization
cost is paid exactly once per matrix.

The wrapper is used by the thermo-elastic distortion solver
(`solve_elastic_system_decomp_cudss.m`) and by the SmartScan 2.0
simulation runners, where each layer assembles a new `K_full` and then
applies its factor to hundreds–thousands of right-hand sides.

---

## What this directory contains

```
cudss_solver/
├── README.md                   <- this file
├── build_mex.m                 <- compile the MEX (run once after install)
├── test_cudss_solver.m         <- correctness + lifecycle test suite
├── +cudss/                     <- MATLAB-facing package (call these)
│   ├── factor.m                <- handle = cudss.factor(K, opts)
│   ├── solve.m                 <-      W = cudss.solve(handle, L)
│   ├── wait.m                  <- cudss.wait(handle)
│   ├── destroy.m               <- cudss.destroy(handle)
│   ├── internal_handle_registry.m   <- internal; do not call directly
│   └── private/                <- compiled MEX binary lands here
└── mex/                        <- C++/CUDA source for the MEX
    ├── cudss_solver.cu         <- the only .cu compiled; dispatcher entry
    ├── solver_state.hpp        <- per-handle state (RAII over cuDSS objects)
    └── class_handle.hpp        <- uint64 <-> C++ pointer marshalling
```

A user only ever calls things in `+cudss/`.  The MEX binary in
`+cudss/private/` is a single dispatcher and is invoked via a command
string (`'factor'`, `'solve'`, `'wait'`, `'destroy'`) from each
wrapper.

---

## What it does, in one paragraph

NVIDIA's cuDSS library factors a sparse matrix `K` on the GPU and
solves `K * W = L` for dense right-hand-side blocks against the
factor.  This wrapper hides the lifecycle (allocate, upload, analyze,
factor, solve, free) behind four MATLAB functions and a `uint64`
opaque handle.  All numeric work happens on the device; both `L` and
`W` are `gpuArray`s with no implicit host round-trip.  The factor and
all associated cuDSS state are pinned in GPU memory until you call
`cudss.destroy(handle)`.

Internally:
- The MATLAB sparse `K` is transposed in MATLAB (because MATLAB stores
  CSC and cuDSS expects CSR, and `CSC(K') == CSR(K)`).
- The MEX uploads CSR row offsets, column indices, and values into
  device buffers and creates a cuDSS sparse matrix descriptor.
- cuDSS runs ANALYSIS (METIS-style reordering + symbolic factor) and
  FACTORIZATION (numeric LU) on the GPU.  These can run synchronously
  or async on a worker thread.
- Each `cudss.solve` wraps `L` and `W` as cuDSS dense matrices and
  fires `cudssExecute(SOLVE)` on the solver's dedicated CUDA stream.
- `cudss.destroy` releases the matrix descriptor, factor data, cuDSS
  handle, stream, and all device buffers in reverse order.

The matrix type defaults to `CUDSS_MTYPE_GENERAL` (full LU, no symmetry
assumption), which matches the assembled thermo-elastic stiffness
`K_full = blkdiag(K_x,K_y,K_z) - T_op - N_op` used by this repository.
Callers with a symmetric or symmetric-positive-definite `K` can pass
`opts.matrix_type = 'symmetric'` (LDLᵀ) or `'spd'` (Cholesky) to get
cuDSS's faster, lower-memory factorization path — see the Options struct
below.

---

## Requirements

- **MATLAB R2024b or newer** (native single-precision sparse).  Older
  releases are rejected by `build_mex`.
- **CUDA Toolkit 12.x** that matches the installed cuDSS.
- **NVIDIA cuDSS 0.7.x**, installed locally.  The build script looks
  for it in this order:
  1. `'CudssPath'` name/value argument
  2. `CUDSS_PATH` environment variable
  3. Platform default — Linux: `/usr/local/cudss` (or
     `/opt/nvidia/cudss/0.7`); Windows: `C:\Program Files\NVIDIA cuDSS\v0.7`
- A working `gpuDevice()` (or pass `'Arch', 'sm_XY'` to `build_mex`
  explicitly on a build-only host).

---

## Build

From MATLAB, with this directory on the path:

```matlab
build_mex                                 % auto-detects everything
build_mex('CudssPath', '/usr/local/cudss')% override the cuDSS install
build_mex('Arch', 'sm_89')                % override the GPU arch
build_mex('Debug', true)                  % -G -O0 line-level CUDA debug
build_mex('Clean', true)                  % wipe prior binary first
```

The build produces a single MEX binary
`+cudss/private/cudss_solver.<mexext>`.  All four `cudss.*`
functions call into it via the command-string dispatcher.

On Linux, if `libcudss.so` is not on `LD_LIBRARY_PATH` at MATLAB
launch time, the build prints a one-line reminder of the `export`
command needed.

---

## Quick start

```matlab
% K is an n x n sparse, real, square matrix (single or double).
% The default makes no symmetry assumption and factors via general LU;
% pass opts.matrix_type to select symmetric LDL^T or SPD Cholesky.
handle = cudss.factor(K);                        % defaults to FP32, general LU

L_gpu = gpuArray(single(L));                     % n x nrhs RHS block
W_gpu = cudss.solve(handle, L_gpu);              % K * W = L on the GPU
W     = gather(W_gpu);                           % back to host if needed

cudss.destroy(handle);                           % free GPU resources
```

`cudss.solve` requires `L` to already live on the GPU and to match the
factor's precision.  Pass a host array and you get a clean error
asking you to `gpuArray(single(L))` (or `gpuArray(double(L))`)
yourself — that prevents large implicit transfers from sneaking into
benchmarks.

---

## Options struct

```matlab
opts = struct( ...
    'precision',   'single',   ...  % 'single' (default) or 'double'
    'matrix_type', 'general',  ...  % 'general' (default) | 'symmetric' | 'spd'
    'async',       false,      ...  % true: queue analysis + factor and return
    'verbose',     false,      ...  % per-phase ms timing printouts
    'reordering',  'default',  ...  % only 'default' is honored today
    'hybrid',       'false',   ...  % true: run memory mode split between system and VRAM
    'hybrid_memory_limit_gib', '0' ... % specify the VRAM memory limit when in hybrid mode. '0' means no limit and the program uses all the VRAM the program estimates it needs
    'hybrid_memory_report', 'false'); %memory report of reported max and min memory needed before and after factorization. Used to define the hybrid memory limit

handle = cudss.factor(K, opts);
```

- `precision` must match `class(K)`.  Pass a `single` sparse with
  `'single'` and a `double` sparse with `'double'`.  `cudss.factor`
  will auto-cast if the precisions disagree but is faster if you
  hand it a matrix that already has the right class.
- `matrix_type` selects the cuDSS matrix type / factor path:
  - `'general'` — `CUDSS_MTYPE_GENERAL`, full LU.  No symmetry assumed.
  - `'symmetric'` — `CUDSS_MTYPE_SYMMETRIC`, LDLᵀ.  K must be symmetric.
  - `'spd'` — `CUDSS_MTYPE_SPD`, Cholesky.  K must be symmetric positive
    definite (a non-PD K will surface as `cudss:NumericalError` from a
    non-zero `CUDSS_DATA_INFO` during factorization).

  For `'symmetric'` and `'spd'`, the wrapper extracts `triu(K)` before
  transposing, so only the upper triangle is transferred to and stored on
  the device (~half the CSR nnz vs `'general'`).  **Symmetry is NOT
  validated** — if K is not actually symmetric, the solve silently uses
  `triu(K)` and treats the lower triangle as `triu(K)^T`.  The caller is
  responsible for the declaration being true.
- `async=true` is the right choice when you have unrelated GPU or CPU
  work to do between the factor request and the first solve — see
  **Overlapping factor with other work** below.
- `verbose=true` prints a ms breakdown for sparse cast / transpose /
  H2D / cuDSS analysis / cuDSS factorization.  Useful for finding
  where time goes when you first integrate the solver.
- `hybrid=true` uses both system and GPU memory together to reduce VRAM memory requirements with a minimal overhead cost (The overhead cost has not been quantified). Specifically, LU decomposition factors are stored in system ram. cudSS transfer the factor data it currently needs from RAM to VRAM and performs the computation in chunks.
- `hybrid_memory_limit_gib = 0`unless otherwise specified the GPU will always use as much VRAM as it needs. In order to get memory savings you need to specify a memory limit in GiB. This requirement requires the user to figure out how much VRAM cudSS needs, add a safety margin, and specify the limit before any VRAM is saved. cudSS is able to very accurately estimate this value before any factorization using built in commands.
- `hybrid_memory_report = false`gives memory report that prints out in matlab that specifies how much system RAM and VRAM will be used and how much is estimated to be used if switched to hybrid mode.
---

## Async factor + `cudss.wait`

Most cuDSS factor cost is **host-side**: METIS reordering and symbolic
factor are single-threaded inside cuDSS and take O(seconds) at our
mesh sizes.  When you have unrelated work (lumping kernels, sparse
matmul on a different stream, CPU bookkeeping), `opts.async = true`
overlaps the two:

```matlab
handle = cudss.factor(K, struct('precision','single','async',true));

%% ... do unrelated GPU / CPU work here ...

cudss.wait(handle);   % drain anything the unrelated work didn't cover
W = cudss.solve(handle, L_gpu);
```

`cudss.solve` on the same handle is also safe without an explicit
`cudss.wait` — it queues behind the factor on the same CUDA stream
and defensively joins the worker thread before reading the result.
Calling `cudss.wait` first is purely for observability (to time the
*residual* factor cost that overlap didn't hide).

---

## Cleanup pattern with `onCleanup` (used in the SmartScan 2 runners)

Errors, `Ctrl+C`, and `return`s inside a layer loop must not leak the
cuDSS handle — GPU memory is finite and a leaked factor occupies
hundreds of MB of device memory.  The pattern used in
`run_simulation_SmartScan_2_only.m` (and friends) is:

```matlab
function cleanup_cudss(h_K)
    % Bound to an onCleanup so cudss.destroy fires on any scope exit
    % (normal end-of-iteration, error, Ctrl+C).  cudss.destroy is
    % idempotent and warns on a stale handle; suppress that single
    % warning ID so a clean shutdown stays silent.
    prev_warn = warning('off', 'cudss:AlreadyDestroyed');
    restorer  = onCleanup(@() warning(prev_warn)); %#ok<NASGU>
    try
        cudss.destroy(h_K);
    catch ME_K
        warning('cudss:CleanupFailed', ...
                'cudss.destroy(handle_K) failed during cleanup: %s', ...
                ME_K.message);
    end
end
```

Inside the layer loop:

```matlab
for layer = 1:num_layers
    % --- IMPORTANT: clear the previous layer's cleanup BEFORE issuing
    %     the new factor.  Otherwise the assignment of
    %     cudss_handles_cleanup fires the prior destructor AFTER the
    %     new factor has been allocated, peaking at ~2x steady-state
    %     GPU footprint.  Clearing first keeps peak == steady-state.
    if exist('cudss_handles_cleanup', 'var')
        clear cudss_handles_cleanup
    end

    handle_K = cudss.factor(K_layer, struct('precision','single','async',true));
    cudss_handles_cleanup = onCleanup(@() cleanup_cudss(handle_K));

    % ... lumping kernels run concurrently with the async factor ...

    cudss.wait(handle_K);                    % optional, for timing only
    U_gpu = cudss.solve(handle_K, L_th_gpu); % per-RHS solves
    % ... no explicit destroy needed; cleanup_cudss fires next iteration
end
```

Why this works:

- `cudss_handles_cleanup` holds the only reference to the `onCleanup`
  object.  When that variable is overwritten or cleared, MATLAB fires
  the cleanup callback.  Any path out of the loop body (normal
  iteration end, `error`, `Ctrl+C`, `return`) destroys the object.
- The explicit `clear cudss_handles_cleanup` BEFORE the next factor
  is what guarantees the steady-state memory profile.  Without it,
  MATLAB still has the old `onCleanup` alive when the new factor
  allocates, so peak ≈ 2 × steady-state.
- `cudss.destroy` is idempotent, so a clean end-of-script destroy
  followed by the `onCleanup` firing again is a no-op (modulo the
  one warning, which the wrapper suppresses).

There is no obligation to use `onCleanup` — a plain `cudss.destroy`
at the end of a function is fine for one-shot use, as in
`solve_elastic_system_decomp_cudss.m`.  But for any loop where the
body can error, `onCleanup` is what keeps GPU memory honest.

---

## Lifecycle invariants

- A `uint64` handle is valid from the moment `cudss.factor` returns
  until `cudss.destroy(handle)` is called.  Anything else (using a
  destroyed handle, or a `uint64` not produced by `cudss.factor`) is
  rejected by the MATLAB-side registry with `cudss:InvalidHandle`.
- `cudss.destroy` is idempotent.  A second destroy emits
  `cudss:AlreadyDestroyed` and returns.
- If MATLAB unloads the MEX (`clear cudss_solver`, `clear mex`, or
  MATLAB exit) while a factor is still live, the MEX's `mexAtExit`
  hook walks an internal ledger and destroys each handle so cuDSS
  resources are not orphaned.
- The MEX is `mexLock`-ed while at least one handle is live, so
  `clear mex` cannot unload it under your feet between
  `cudss.factor` and `cudss.destroy`.

---

## Errors you might see

| Error ID                          | Meaning |
|-----------------------------------|---------|
| `cudss:BadInput`                  | Input failed shape/type validation (not sparse, not square, wrong dimensions, etc.). |
| `cudss:WrongInputType`            | Class of `K` or `L` does not match `opts.precision` / the handle's precision. |
| `cudss:BadOpts`                   | `opts` struct has an unrecognized value (e.g., bad precision string). |
| `cudss:InvalidHandle`             | `uint64` passed to `solve`/`wait` is not in the live registry. |
| `cudss:AlreadyDestroyed`          | (warning) destroy called on a stale handle; no-op. |
| `cudss:NumericalError`            | cuDSS returned success but reported a non-zero `CUDSS_DATA_INFO` (typically zero/negative pivot) — matrix is likely singular, or `matrix_type='spd'` was used on a non-PD K. Same ID from sync and async factor paths. |
| `cudss:CudssError`                | A cuDSS API call failed; full status name + numeric code are in the message. |
| `cudss:CudaError`                 | A CUDA Runtime call failed; `cudaError` name + numeric code are in the message. |
| `cudss:Error`                     | Catch-all for unexpected exceptions raised during factor construction (CUDA or cuDSS failures inside the exception-safe path that aren't already classified above). |
| `cudss:UnsupportedMATLAB`         | `build_mex` rejected the MATLAB release (need R2024b+). |
| `cudss:BuildPath`                 | `build_mex` could not find `cudss.h` / `libcudss.*` under the resolved cuDSS path. |

---

## Testing

`test_cudss_solver.m` is a self-contained correctness suite:
small smoke tests, multi-RHS correctness, a 3-D 7-point Laplacian
(octree proxy), factor-once / solve-many, precision dispatch,
wrong-precision rejection, handle lifecycle, double-destroy, async
factor, `matrix_type` agreement across general/symmetric/spd, SPD
rejection of indefinite K, lock-counter regression, singular-matrix
detection, and cross-stream interleaving.  Run from MATLAB after
`build_mex`:

```matlab
addpath(genpath('subFunctions'));
run('subFunctions/gpu_solvers/cudss_solver/test_cudss_solver.m');
```

---

## See also

- `reference_docs/cudss.md` — design notes and the cuDSS library
  comparison that motivated this wrapper.
- `subFunctions/solve_elastic_system_decomp_cudss.m` — the one-shot
  caller (factor + multi-RHS solve + destroy) that mirrors the CPU
  baseline `solve_elastic_system_decomp.m`.
- `run_simulation_SmartScan_2_only.m` — the factor-once-per-layer
  caller that uses `onCleanup` + async factor + repeated solves
  inside the layer loop.
