// cudss_solver.cu - single dispatcher MEX for the +cudss MATLAB package.
//
// OVERVIEW
// --------
// This file is the C++/CUDA backend behind the four MATLAB-facing functions
// in +cudss/{factor,solve,destroy,wait}.m.  All four .m wrappers call this
// one MEX binary, passing a command string as the first argument:
//
//     handle = cudss_solver('factor',  Kt, opts)
//     W      = cudss_solver('solve',   handle, L)
//              cudss_solver('destroy', handle)
//              cudss_solver('wait',    handle)
//
// mexFunction at the bottom of this file dispatches on that command string
// to one of the four handle_* functions defined here.
//
// WHY ONE BINARY AND NOT FOUR
// ---------------------------
// MATLAB's mexLock / mexUnlock reference counts are tracked per-MEX-file.
// If 'factor' and 'destroy' lived in separate .mex binaries, the
// mexLock fired during factor would pin the factor binary in memory, and
// the mexUnlock fired during destroy would decrement an unrelated counter
// on the destroy binary.  The factor binary would stay loaded for the
// life of the MATLAB session and its lock count would grow without bound.
// Bundling all four commands into one binary keeps lock/unlock paired on
// the same file.
//
// WHAT A "HANDLE" IS
// ------------------
// 'factor' returns a uint64 scalar to MATLAB.  That uint64 is the address
// of a C++ wrapper object (class_handle<SolverState>) that owns one
// SolverState -- the cuDSS handle, configuration, factor data, CSR matrix
// on the device, and a dedicated CUDA stream.  Subsequent 'solve',
// 'destroy', and 'wait' calls cast the uint64 back to a wrapper pointer
// (see class_handle.hpp) and operate on the owned SolverState.
//
// EXCEPTION SAFETY
// ----------------
// MATLAB's mexErrMsgIdAndTxt raises an error by longjmp-ing out of
// mexFunction.  longjmp does not run C++ destructors on the way out, so a
// raw mexErrMsgIdAndTxt in the middle of building a SolverState would leak
// any CUDA / cuDSS resources allocated so far.  The factor handler holds
// the in-construction state in a std::unique_ptr and uses throwing macros
// (CUDA_THROW / CUDSS_THROW from solver_state.hpp); a try/catch at the
// outer scope converts the C++ exception to a normal MATLAB error after
// the unique_ptr destructor has cleaned up.
//
// ASYNC FACTOR
// ------------
// cuDSS analysis (METIS-based reordering and symbolic factor) runs on the
// host single-threaded and can take O(seconds) on large sparse matrices.
// When opts.async = true, the factor handler spawns a std::thread that
// queues the analysis + factorization onto the SolverState's dedicated
// CUDA stream and returns immediately.  The caller can do unrelated GPU
// or CPU work in the meantime and call cudss.wait (or cudss.solve, which
// flushes implicitly) before reading any result that depends on the
// factor.
//
// STREAM HANDLING
// ---------------
// Each SolverState owns a private CUDA stream.  MATLAB's gpuArrays live
// on the legacy default stream (stream 0).  cudaStreamCreate with default
// flags returns a blocking stream that implicitly serializes with stream
// 0, so kernels MATLAB queued to populate L finish before the SOLVE
// reads it.  As belt-and-suspenders, every SOLVE also records an event
// on stream 0 and makes state->stream wait on it explicitly -- that
// keeps correctness intact if MATLAB ever switches gpuArrays to a
// per-thread default stream, or if this code is later changed to use a
// non-blocking stream.
//
// CLEAN SHUTDOWN
// --------------
// at_exit_cleanup (registered via mexAtExit) walks a static ledger of
// live wrapper pointers and destroys each one if MATLAB unloads the MEX
// (`clear cudss_solver`, `clear mex`, MATLAB exit).  Without this, a
// MATLAB session that exits while a factor is still alive would orphan
// cuDSS resources and device memory.

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "class_handle.hpp"
#include "solver_state.hpp"

#include <cuda_runtime.h>
#include <cudss.h>
#include <library_types.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
//  Verbose timing helpers (factor handler).  When opts.verbose is true the
//  factor handler prints a per-phase ms breakdown using these macros.
// ---------------------------------------------------------------------------
#define VERBOSE_TIC(var) auto var = std::chrono::steady_clock::now()
#define VERBOSE_TOC(var, label)                                            \
    do {                                                                   \
        if (verbose) {                                                     \
            auto _now = std::chrono::steady_clock::now();                  \
            double _ms = std::chrono::duration<double, std::milli>(        \
                _now - var).count();                                       \
            mexPrintf("    [cudss_solver] %-32s %8.2f ms\n",               \
                      label, _ms);                                         \
            mexEvalString("drawnow;");                                     \
        }                                                                  \
    } while (0)

// ===========================================================================
//  Live-handle ledger
//
//  Static vector of every class_handle<SolverState>* this MEX has issued
//  and not yet destroyed.  The MEX boundary is single-threaded, so no
//  mutex is needed -- 'factor', 'destroy', 'solve', 'wait' calls cannot
//  observably race.  mexAtExit fires once when MATLAB unloads the MEX;
//  it walks the ledger and fires each class_handle's destructor, which
//  in turn fires ~SolverState() and releases all CUDA / cuDSS resources.
// ===========================================================================

namespace {

std::vector<class_handle<SolverState> *> &live_ledger() {
    static std::vector<class_handle<SolverState> *> v;
    return v;
}

void ledger_add(class_handle<SolverState> *p) {
    live_ledger().push_back(p);
}

void ledger_remove(class_handle<SolverState> *p) {
    auto &v = live_ledger();
    auto it = std::find(v.begin(), v.end(), p);
    if (it != v.end()) {
        v.erase(it);
    }
}

void at_exit_cleanup() {
    // MATLAB is unloading the MEX.  Destroy every live handle so cuDSS
    // resources are not orphaned.  mexUnlock is intentionally NOT called
    // here -- the MEX is being torn down regardless of the lock counter,
    // and calling mexUnlock during exit emits a warning on some MATLAB
    // releases.
    auto &v = live_ledger();
    for (auto *p : v) {
        if (p != nullptr) {
            // delete fires class_handle dtor -> ~SolverState() ->
            // cudssMatrixDestroy / cudssDestroy / cudaStreamDestroy /
            // cudaFree x3.
            delete p;
        }
    }
    v.clear();
}

bool g_at_exit_registered = false;

void ensure_at_exit_registered() {
    if (!g_at_exit_registered) {
        mexAtExit(at_exit_cleanup);
        g_at_exit_registered = true;
    }
}

// Wrap a freshly-constructed SolverState pointer into a class_handle, pack
// it into a uint64 mxArray (the value handed back to MATLAB), and add the
// wrapper to the ledger so at_exit_cleanup can find it.
mxArray *wrap_and_register(SolverState *raw) {
    mxArray *out = convertPtr2Mat<SolverState>(raw);
    uint64_t encoded = *static_cast<uint64_t *>(mxGetData(out));
    auto *wrap = reinterpret_cast<class_handle<SolverState> *>(encoded);
    ledger_add(wrap);
    return out;
}

}  // namespace

// ===========================================================================
//  handle_factor
//
//  Builds a SolverState from a (transposed) sparse matrix Kt and an opts
//  struct.  The MATLAB wrapper +cudss/factor.m has already transposed K so
//  Kt's CSC arrays (Jc / Ir / values) can be read directly as cuDSS's CSR
//  (rowOffsets / colIndices / values) -- this is the entire CSC->CSR
//  conversion.  On the sync path the call blocks until factorization
//  finishes; on the async path it returns once analysis + factorization
//  are queued on the solver's stream.
// ===========================================================================

static void handle_factor(int nlhs, mxArray *plhs[],
                          int nrhs, const mxArray *prhs[])
{
    if (nrhs != 2) {
        mexErrMsgIdAndTxt("cudss:BadNargin",
            "cudss_solver('factor', ...) expects 2 trailing inputs (Kt, opts).");
    }
    if (nlhs > 1) {
        mexErrMsgIdAndTxt("cudss:BadNargout",
            "cudss_solver('factor', ...) returns at most 1 output.");
    }

    mxInitGPU();

    const mxArray *Kt   = prhs[0];
    const mxArray *opts = prhs[1];

    // ---- shape / type validation on the (already-transposed) input ----
    if (!mxIsSparse(Kt)) {
        mexErrMsgIdAndTxt("cudss:BadInput", "Kt must be sparse.");
    }
    if (mxIsComplex(Kt)) {
        mexErrMsgIdAndTxt("cudss:BadInput", "Kt must be real.");
    }
    if (mxGetM(Kt) != mxGetN(Kt)) {
        mexErrMsgIdAndTxt("cudss:BadInput", "Kt must be square.");
    }
    if (!mxIsStruct(opts)) {
        mexErrMsgIdAndTxt("cudss:BadInput", "opts must be a struct.");
    }

    // ---- parse opts.precision ('single' or 'double') ----
    bool is_single = true;
    {
        const mxArray *p = mxGetField(opts, 0, "precision");
        if (p != nullptr && !mxIsEmpty(p)) {
            if (!mxIsChar(p)) {
                mexErrMsgIdAndTxt("cudss:BadOpts",
                    "opts.precision must be 'single' or 'double'.");
            }
            char buf[16] = {0};
            mxGetString(p, buf, sizeof(buf));
            if (std::strcmp(buf, "single") == 0)        is_single = true;
            else if (std::strcmp(buf, "double") == 0)   is_single = false;
            else mexErrMsgIdAndTxt("cudss:BadOpts",
                "opts.precision must be 'single' or 'double'.");
        }
    }

    // ---- parse opts.async (queue ANALYSIS + FACTORIZATION on a worker thread) ----
    bool async_factor = false;
    {
        const mxArray *p = mxGetField(opts, 0, "async");
        if (p != nullptr && !mxIsEmpty(p)) {
            if (!mxIsLogicalScalar(p) ||
                mxGetNumberOfElements(p) != 1) {
                mexErrMsgIdAndTxt("cudss:BadOpts",
                    "opts.async must be a logical scalar.");
            }
            async_factor = mxIsLogicalScalarTrue(p);
        }
    }

    // ---- parse opts.verbose (per-phase ms timing printout) ----
    bool verbose = false;
    {
        const mxArray *p = mxGetField(opts, 0, "verbose");
        if (p != nullptr && !mxIsEmpty(p)) {
            if (!mxIsLogicalScalar(p) ||
                mxGetNumberOfElements(p) != 1) {
                mexErrMsgIdAndTxt("cudss:BadOpts",
                    "opts.verbose must be a logical scalar.");
            }
            verbose = mxIsLogicalScalarTrue(p);
        }
    }

    // ---- parse opts.matrix_type ('general' | 'symmetric' | 'spd') ----
    // The MATLAB wrapper extracts triu(K) for symmetric/spd before
    // transposing, so when we land here the CSR bytes already represent
    // the upper triangle and we must announce view=UPPER to cuDSS.
    cudssMatrixType_t     mtype = CUDSS_MTYPE_GENERAL;
    cudssMatrixViewType_t mview = CUDSS_MVIEW_FULL;
    {
        const mxArray *p = mxGetField(opts, 0, "matrix_type");
        if (p != nullptr && !mxIsEmpty(p)) {
            if (!mxIsChar(p)) {
                mexErrMsgIdAndTxt("cudss:BadOpts",
                    "opts.matrix_type must be 'general', 'symmetric', or 'spd'.");
            }
            char buf[16] = {0};
            mxGetString(p, buf, sizeof(buf));
            if (std::strcmp(buf, "general") == 0) {
                mtype = CUDSS_MTYPE_GENERAL;
                mview = CUDSS_MVIEW_FULL;
            } else if (std::strcmp(buf, "symmetric") == 0) {
                mtype = CUDSS_MTYPE_SYMMETRIC;
                mview = CUDSS_MVIEW_UPPER;
            } else if (std::strcmp(buf, "spd") == 0) {
                mtype = CUDSS_MTYPE_SPD;
                mview = CUDSS_MVIEW_UPPER;
            } else {
                mexErrMsgIdAndTxt("cudss:BadOpts",
                    "opts.matrix_type must be 'general', 'symmetric', or 'spd'.");
            }
        }
    }

    // ---- the storage class of Kt must match the requested precision ----
    mxClassID kclass = mxGetClassID(Kt);
    if (is_single && kclass != mxSINGLE_CLASS) {
        mexErrMsgIdAndTxt("cudss:WrongInputType",
            "opts.precision='single' requires Kt to be a single sparse "
            "(class(Kt) is '%s'). Cast caller-side.",
            mxGetClassName(Kt));
    }
    if (!is_single && kclass != mxDOUBLE_CLASS) {
        mexErrMsgIdAndTxt("cudss:WrongInputType",
            "opts.precision='double' requires Kt to be a double sparse "
            "(class(Kt) is '%s'). Cast caller-side.",
            mxGetClassName(Kt));
    }

    // ---- exception-safe construction ----
    //
    // The state pointer is held in a unique_ptr so that any throw before
    // we hand ownership to the wrapper (release(), below) runs the
    // SolverState destructor during stack unwinding -- which releases
    // every CUDA / cuDSS resource that had been allocated so far.  The
    // CUDA_THROW / CUDSS_THROW macros (solver_state.hpp) raise a
    // std::runtime_error on failure; the catch at the bottom converts
    // that into a normal MATLAB error.
    try {
        std::unique_ptr<SolverState> state(new SolverState());
        state->is_single  = is_single;
        state->value_type = is_single ? CUDA_R_32F : CUDA_R_64F;
        state->n          = static_cast<int>(mxGetN(Kt));

        // MATLAB sparse uses mwIndex (size_t-ish) for Jc / Ir.  cuDSS
        // expects 32-bit int for CSR row offsets and column indices.
        const mwIndex *Jc = mxGetJc(Kt);
        const mwIndex *Ir = mxGetIr(Kt);

        mwIndex nnz_mw = Jc[state->n];
        if (nnz_mw >= static_cast<mwIndex>(INT_MAX)) {
            throw std::runtime_error(
                std::string("nnz(K) = ") + std::to_string(static_cast<unsigned long long>(nnz_mw))
                + " exceeds INT_MAX; need int64 indexing path (not implemented).");
        }
        state->nnz = static_cast<int>(nnz_mw);

        // ---- downcast mwIndex (8 byte) -> int32 host buffers ----
        VERBOSE_TIC(t_downcast);
        std::vector<int> h_row_off(state->n + 1);
        std::vector<int> h_col_idx(state->nnz);
        for (int i = 0; i <= state->n; ++i) {
            h_row_off[i] = static_cast<int>(Jc[i]);
        }
        for (int k = 0; k < state->nnz; ++k) {
            h_col_idx[k] = static_cast<int>(Ir[k]);
        }
        VERBOSE_TOC(t_downcast, "mwIndex->int downcast");

        const size_t sizeof_value = is_single ? sizeof(float) : sizeof(double);

        // ---- upload CSR + values to device ----
        VERBOSE_TIC(t_malloc);
        CUDA_THROW(cudaMalloc(reinterpret_cast<void **>(&state->d_row_offsets),
                              (state->n + 1) * sizeof(int)));
        CUDA_THROW(cudaMalloc(reinterpret_cast<void **>(&state->d_col_indices),
                              static_cast<size_t>(state->nnz) * sizeof(int)));
        CUDA_THROW(cudaMalloc(&state->d_values,
                              static_cast<size_t>(state->nnz) * sizeof_value));
        VERBOSE_TOC(t_malloc, "cudaMalloc x3 (CSR)");

        VERBOSE_TIC(t_h2d);
        CUDA_THROW(cudaMemcpy(state->d_row_offsets, h_row_off.data(),
                              (state->n + 1) * sizeof(int),
                              cudaMemcpyHostToDevice));
        CUDA_THROW(cudaMemcpy(state->d_col_indices, h_col_idx.data(),
                              static_cast<size_t>(state->nnz) * sizeof(int),
                              cudaMemcpyHostToDevice));

        if (is_single) {
            const float *vals = mxGetSingles(Kt);
            CUDA_THROW(cudaMemcpy(state->d_values, vals,
                                  static_cast<size_t>(state->nnz) * sizeof(float),
                                  cudaMemcpyHostToDevice));
        } else {
            const double *vals = mxGetDoubles(Kt);
            CUDA_THROW(cudaMemcpy(state->d_values, vals,
                                  static_cast<size_t>(state->nnz) * sizeof(double),
                                  cudaMemcpyHostToDevice));
        }
        VERBOSE_TOC(t_h2d, "H2D cudaMemcpy x3 (CSR)");

        // ---- cuDSS lifecycle: handle + stream + config + data + matrix ----
        VERBOSE_TIC(t_cudss_create);
        CUDSS_THROW(cudssCreate(&state->handle));
        // cudaStreamCreate (default flags) returns a *blocking* stream.
        // Blocking streams implicitly serialize with the legacy default
        // stream (stream 0) that MATLAB gpuArrays use, so anything MATLAB
        // queued to populate L_gpu finishes before any work on
        // state->stream begins.  See the explicit event-wait in
        // handle_solve for the belt-and-suspenders.
        CUDA_THROW(cudaStreamCreate(&state->stream));
        CUDSS_THROW(cudssSetStream(state->handle, state->stream));
        CUDSS_THROW(cudssConfigCreate(&state->config));
        CUDSS_THROW(cudssDataCreate(state->handle, &state->data));

        // Pre-create the cross-stream sync event used by handle_solve.
        // cudaEventDisableTiming makes record/wait pure ordering ops
        // (no timing payload), which is all we need.
        CUDA_THROW(cudaEventCreateWithFlags(&state->evt_default,
                                            cudaEventDisableTiming));
        VERBOSE_TOC(t_cudss_create, "cudssCreate + stream + config + event");

        // Config is left at cuDSS defaults: METIS-ND reordering, no
        // iterative refinement, no hybrid memory.

        VERBOSE_TIC(t_csr_desc);
        CUDSS_THROW(cudssMatrixCreateCsr(
            &state->A, state->n, state->n, state->nnz,
            state->d_row_offsets, /*rowEnd=*/NULL, state->d_col_indices,
            state->d_values,
            CUDA_R_32I, state->value_type,
            mtype, mview, CUDSS_BASE_ZERO));
        VERBOSE_TOC(t_csr_desc, "cudssMatrixCreateCsr");

        // cuDSS requires b/x descriptors even for ANALYSIS and
        // FACTORIZATION (it only reads their shape, not their data).
        // We allocate dummy single-column buffers here; the real RHS is
        // wired in during handle_solve.
        VERBOSE_TIC(t_dummy);
        void *d_b_dummy = nullptr;
        void *d_x_dummy = nullptr;
        CUDA_THROW(cudaMalloc(&d_b_dummy,
                              static_cast<size_t>(state->n) * sizeof_value));
        CUDA_THROW(cudaMalloc(&d_x_dummy,
                              static_cast<size_t>(state->n) * sizeof_value));
        CUDA_THROW(cudaMemset(d_b_dummy, 0,
                              static_cast<size_t>(state->n) * sizeof_value));
        CUDA_THROW(cudaMemset(d_x_dummy, 0,
                              static_cast<size_t>(state->n) * sizeof_value));

        cudssMatrix_t b_dummy = nullptr;
        cudssMatrix_t x_dummy = nullptr;
        CUDSS_THROW(cudssMatrixCreateDn(
            &b_dummy, state->n, 1, state->n, d_b_dummy,
            state->value_type, CUDSS_LAYOUT_COL_MAJOR));
        CUDSS_THROW(cudssMatrixCreateDn(
            &x_dummy, state->n, 1, state->n, d_x_dummy,
            state->value_type, CUDSS_LAYOUT_COL_MAJOR));
        VERBOSE_TOC(t_dummy, "dummy b/x alloc + descriptors");

        if (!async_factor) {
            // -------------------- Sync path --------------------
            // Run ANALYSIS + FACTORIZATION inline, drain the stream,
            // then check CUDSS_DATA_INFO for numerical issues before
            // returning the handle.
            VERBOSE_TIC(t_analysis);
            CUDSS_THROW(cudssExecute(state->handle, CUDSS_PHASE_ANALYSIS,
                                     state->config, state->data, state->A,
                                     x_dummy, b_dummy));
            VERBOSE_TOC(t_analysis, "cudssExecute(ANALYSIS)");

            VERBOSE_TIC(t_factorize);
            CUDSS_THROW(cudssExecute(state->handle, CUDSS_PHASE_FACTORIZATION,
                                     state->config, state->data, state->A,
                                     x_dummy, b_dummy));
            VERBOSE_TOC(t_factorize, "cudssExecute(FACTORIZATION) queue");

            // Drain the stream so CUDSS_DATA_INFO is settled before we
            // read it.
            CUDA_THROW(cudaStreamSynchronize(state->stream));

            // cuDSS can return CUDSS_STATUS_SUCCESS while leaving a
            // non-zero info code that signals a numerical issue (a
            // zero or negative pivot at row `info`).  Without this
            // check, a corrupt factor would silently produce garbage
            // on every subsequent solve.
            int info = 0;
            size_t info_written = 0;
            CUDSS_THROW(cudssDataGet(state->handle, state->data,
                                     CUDSS_DATA_INFO,
                                     &info, sizeof(int), &info_written));
            if (info != 0) {
                // Explicitly tear down the dummy descriptors / buffers
                // before throwing -- the unique_ptr destructor will
                // handle the rest of `state` during stack unwinding.
                cudssMatrixDestroy(b_dummy);
                cudssMatrixDestroy(x_dummy);
                cudaFree(d_b_dummy);
                cudaFree(d_x_dummy);
                // Typed exception so the catch below routes this to
                // cudss:NumericalError -- the same MATLAB error ID the
                // async path raises for a DATA_INFO != 0 result.
                throw NumericalError(
                    std::string("cuDSS factorization completed with "
                                "CUDSS_DATA_INFO = ") + std::to_string(info)
                    + " (likely zero/negative pivot -- matrix may be singular).");
            }

            cudssMatrixDestroy(b_dummy);
            cudssMatrixDestroy(x_dummy);
            cudaFree(d_b_dummy);
            cudaFree(d_x_dummy);
        } else {
            // -------------------- Async path --------------------
            //
            // Spawn a worker thread that runs ANALYSIS + FACTORIZATION
            // + the CUDSS_DATA_INFO query.  All three results are
            // recorded on SolverState fields rather than raised from the
            // worker -- longjmp out of a non-MATLAB thread (which
            // mexErrMsgIdAndTxt does) is undefined behavior.  The next
            // join (in handle_solve, handle_wait, or ~SolverState)
            // inspects those fields and raises a MATLAB error if any
            // step failed.
            //
            // The dummy b/x buffers and their cuDSS descriptors stay
            // alive on SolverState until the join, because cuDSS reads
            // them during ANALYSIS / FACTORIZATION; freeing them here
            // would race with the worker.
            state->b_dummy_pending   = b_dummy;
            state->x_dummy_pending   = x_dummy;
            state->d_b_dummy_pending = d_b_dummy;
            state->d_x_dummy_pending = d_x_dummy;
            state->factor_in_flight  = true;
            state->worker_analysis_status      = CUDSS_STATUS_SUCCESS;
            state->worker_factorization_status = CUDSS_STATUS_SUCCESS;
            state->worker_data_info_status     = CUDSS_STATUS_SUCCESS;
            state->worker_factor_info          = 0;

            SolverState *raw = state.get();
            raw->factor_thread = new std::thread([raw]() {
                cudssStatus_t s_a = cudssExecute(
                    raw->handle, CUDSS_PHASE_ANALYSIS,
                    raw->config, raw->data, raw->A,
                    raw->x_dummy_pending, raw->b_dummy_pending);
                raw->worker_analysis_status = s_a;
                if (s_a != CUDSS_STATUS_SUCCESS) {
                    return;
                }
                cudssStatus_t s_f = cudssExecute(
                    raw->handle, CUDSS_PHASE_FACTORIZATION,
                    raw->config, raw->data, raw->A,
                    raw->x_dummy_pending, raw->b_dummy_pending);
                raw->worker_factorization_status = s_f;
                if (s_f != CUDSS_STATUS_SUCCESS) {
                    return;
                }
                // Drain queued kernels before reading DATA_INFO so the
                // value is settled.  Only blocks the worker thread, not
                // MATLAB.
                cudaError_t s_sync = cudaStreamSynchronize(raw->stream);
                if (s_sync != cudaSuccess) {
                    raw->worker_factorization_status =
                        CUDSS_STATUS_EXECUTION_FAILED;
                    return;
                }
                int info = 0;
                size_t written = 0;
                cudssStatus_t s_info = cudssDataGet(
                    raw->handle, raw->data, CUDSS_DATA_INFO,
                    &info, sizeof(int), &written);
                raw->worker_data_info_status = s_info;
                raw->worker_factor_info      = info;
            });
        }

        // Hand ownership of `state` to the wrapper and register the
        // wrapper in the ledger.  After release(), the unique_ptr no
        // longer owns the SolverState -- teardown is now the wrapper's
        // responsibility (and ultimately destroyObject from
        // handle_destroy, or at_exit_cleanup at MEX unload).
        plhs[0] = wrap_and_register(state.release());

    } catch (const NumericalError &e) {
        // Surface zero/negative-pivot info from the sync DATA_INFO check
        // under the same MATLAB error ID the async path uses (see
        // handle_solve / handle_wait), so callers can branch on a stable
        // identifier regardless of whether they took the sync or async
        // factor path.
        mexErrMsgIdAndTxt("cudss:NumericalError", "%s", e.what());
    } catch (const std::exception &e) {
        // Convert C++ exception (from any CUDA_THROW / CUDSS_THROW above)
        // into a normal MATLAB error.  unique_ptr's destructor has
        // already torn down any partially-allocated CUDA / cuDSS state.
        mexErrMsgIdAndTxt("cudss:Error", "%s", e.what());
    }
}

// ===========================================================================
//  handle_solve
//
//  Applies the persisted factorization to a multi-RHS gpuArray L.  Returns
//  W = K^{-1} L as a gpuArray of the same shape and class.  Both L and W
//  stay device-resident -- no host round-trip occurs inside this call.
//
//  Before solving, any in-flight async factor is joined and its status /
//  CUDSS_DATA_INFO codes are surfaced as MATLAB errors.  After solving,
//  the SOLVE-side stream is drained so MATLAB sees a settled W when the
//  call returns.
// ===========================================================================

static void handle_solve(int nlhs, mxArray *plhs[],
                         int nrhs, const mxArray *prhs[])
{
    if (nrhs != 2) {
        mexErrMsgIdAndTxt("cudss:BadNargin",
            "cudss_solver('solve', ...) expects 2 trailing inputs (handle, L).");
    }
    if (nlhs > 1) {
        mexErrMsgIdAndTxt("cudss:BadNargout",
            "cudss_solver('solve', ...) returns at most 1 output.");
    }

    mxInitGPU();

    SolverState *state = convertMat2Ptr<SolverState>(prhs[0]);

    // ---- defensive flush of any pending async factor ----
    // If the caller never invoked cudss.wait between factor and solve,
    // we still need to make sure the worker thread has returned and
    // the factor is numerically valid before applying it.
    if (state->factor_thread != nullptr) {
        if (state->factor_thread->joinable()) {
            state->factor_thread->join();
        }
        delete state->factor_thread;
        state->factor_thread = nullptr;

        if (state->worker_analysis_status != CUDSS_STATUS_SUCCESS) {
            cudssStatus_t s = state->worker_analysis_status;
            mexErrMsgIdAndTxt("cudss:CudssError",
                "Async cudssExecute(CUDSS_PHASE_ANALYSIS) failed with "
                "status %s (%d).",
                cudss_status_name(s), static_cast<int>(s));
        }
        if (state->worker_factorization_status != CUDSS_STATUS_SUCCESS) {
            cudssStatus_t s = state->worker_factorization_status;
            mexErrMsgIdAndTxt("cudss:CudssError",
                "Async cudssExecute(CUDSS_PHASE_FACTORIZATION) failed with "
                "status %s (%d).",
                cudss_status_name(s), static_cast<int>(s));
        }
        if (state->worker_data_info_status != CUDSS_STATUS_SUCCESS) {
            cudssStatus_t s = state->worker_data_info_status;
            mexErrMsgIdAndTxt("cudss:CudssError",
                "Async cudssDataGet(CUDSS_DATA_INFO) failed with "
                "status %s (%d).",
                cudss_status_name(s), static_cast<int>(s));
        }
        if (state->worker_factor_info != 0) {
            int info = state->worker_factor_info;
            mexErrMsgIdAndTxt("cudss:NumericalError",
                "cuDSS factorization completed with CUDSS_DATA_INFO = %d "
                "(likely zero/negative pivot -- matrix may be singular).",
                info);
        }
    }
    if (state->factor_in_flight) {
        // After the worker has returned, the queued cuDSS kernels on
        // state->stream may still be in flight.  Drain before releasing
        // the dummy b/x they read during factorization.
        CUDA_CHECK(cudaStreamSynchronize(state->stream));
        if (state->b_dummy_pending   != nullptr) { cudssMatrixDestroy(state->b_dummy_pending); state->b_dummy_pending = nullptr; }
        if (state->x_dummy_pending   != nullptr) { cudssMatrixDestroy(state->x_dummy_pending); state->x_dummy_pending = nullptr; }
        if (state->d_b_dummy_pending != nullptr) { cudaFree(state->d_b_dummy_pending);         state->d_b_dummy_pending = nullptr; }
        if (state->d_x_dummy_pending != nullptr) { cudaFree(state->d_x_dummy_pending);         state->d_x_dummy_pending = nullptr; }
        state->factor_in_flight = false;
    }

    // ---- L must be a gpuArray ----
    if (!mxIsGPUArray(prhs[1])) {
        mexErrMsgIdAndTxt("cudss:WrongInputType",
            "L must be a gpuArray.  Move it with gpuArray(L) before calling "
            "cudss.solve.");
    }

    const mxGPUArray *L_gpu = mxGPUCreateFromMxArray(prhs[1]);

    // Class of L must match the handle's precision.
    mxClassID L_class = mxGPUGetClassID(L_gpu);
    mxClassID expected = state->is_single ? mxSINGLE_CLASS : mxDOUBLE_CLASS;
    if (L_class != expected) {
        mxGPUDestroyGPUArray(L_gpu);
        mexErrMsgIdAndTxt("cudss:WrongInputType",
            "L precision (class id %d) does not match handle precision "
            "(expected %d).  Cast L to %s.",
            static_cast<int>(L_class), static_cast<int>(expected),
            state->is_single ? "single" : "double");
    }

    if (mxGPUGetNumberOfDimensions(L_gpu) != 2) {
        mxGPUDestroyGPUArray(L_gpu);
        mexErrMsgIdAndTxt("cudss:BadInput", "L must be 2-D.");
    }
    const mwSize *L_dims = mxGPUGetDimensions(L_gpu);
    if (static_cast<int>(L_dims[0]) != state->n) {
        mwSize bad_n = L_dims[0];
        mxGPUDestroyGPUArray(L_gpu);
        mexErrMsgIdAndTxt("cudss:BadInput",
            "size(L,1) = %llu does not match handle n = %d.",
            static_cast<unsigned long long>(bad_n), state->n);
    }
    int nrhs_cols = static_cast<int>(L_dims[1]);
    if (nrhs_cols < 1) {
        mxGPUDestroyGPUArray(L_gpu);
        mexErrMsgIdAndTxt("cudss:BadInput", "size(L,2) must be >= 1.");
    }

    const void *d_L = mxGPUGetDataReadOnly(L_gpu);

    // Allocate the output gpuArray.  W has the same shape and class as L.
    mwSize W_dims[2] = { static_cast<mwSize>(state->n),
                         static_cast<mwSize>(nrhs_cols) };
    mxGPUArray *W_gpu = mxGPUCreateGPUArray(2, W_dims, expected, mxREAL,
                                            MX_GPU_DO_NOT_INITIALIZE);
    void *d_W = mxGPUGetData(W_gpu);

    // Wrap L and W as cuDSS dense matrices.  Column-major and ldb = n
    // exactly match MATLAB's gpuArray storage; no transposes required.
    cudssMatrix_t b_desc = nullptr;
    cudssMatrix_t x_desc = nullptr;
    CUDSS_CHECK(cudssMatrixCreateDn(
        &b_desc, state->n, nrhs_cols, state->n,
        const_cast<void *>(d_L),
        state->value_type, CUDSS_LAYOUT_COL_MAJOR));
    CUDSS_CHECK(cudssMatrixCreateDn(
        &x_desc, state->n, nrhs_cols, state->n,
        d_W, state->value_type, CUDSS_LAYOUT_COL_MAJOR));

    // ---- explicit cross-stream synchronization ----
    //
    // MATLAB's gpuArrays live on the legacy default stream (stream 0).
    // state->stream is a separate (blocking) stream.  Today the blocking
    // flag implicitly serializes the two, so a kernel queued on stream 0
    // that wrote to L_gpu finishes before any kernel on state->stream
    // begins.  But that contract is fragile -- a future MATLAB release
    // that switches gpuArrays to a per-thread default stream, or a
    // refactor here that switches state->stream to non-blocking, would
    // silently let the SOLVE read a partially-initialized L.  Recording
    // an event on stream 0 and waiting on it from state->stream is a
    // belt-and-suspenders that costs ~one event per solve and survives
    // either of those changes.
    CUDA_CHECK(cudaEventRecord(state->evt_default, /*stream=*/0));
    CUDA_CHECK(cudaStreamWaitEvent(state->stream, state->evt_default, 0));

    CUDSS_CHECK(cudssExecute(state->handle, CUDSS_PHASE_SOLVE,
                             state->config, state->data, state->A,
                             x_desc, b_desc));

    // Drain state->stream so W is observably settled when MATLAB next
    // reads it (e.g. via gather or as input to another kernel).
    CUDA_CHECK(cudaStreamSynchronize(state->stream));

    cudssMatrixDestroy(b_desc);
    cudssMatrixDestroy(x_desc);

    plhs[0] = mxGPUCreateMxArrayOnGPU(W_gpu);

    mxGPUDestroyGPUArray(W_gpu);
    mxGPUDestroyGPUArray(L_gpu);
}

// ===========================================================================
//  handle_destroy
//
//  Releases everything owned by one factored handle: cuDSS handle,
//  config, data, sparse matrix descriptor, dedicated stream, device CSR
//  buffers, sync event.  Idempotent at the C++ level: passing a stale
//  uint64 emits a warning instead of erroring (a double-destroy after
//  the MATLAB-side registry already removed the entry is harmless).
// ===========================================================================

static void handle_destroy(int nlhs, mxArray *plhs[],
                           int nrhs, const mxArray *prhs[])
{
    (void)nlhs; (void)plhs;
    if (nrhs != 1) {
        mexErrMsgIdAndTxt("cudss:BadNargin",
            "cudss_solver('destroy', ...) expects 1 trailing input (handle).");
    }

    class_handle<SolverState> *p = tryConvertMat2HandlePtr<SolverState>(prhs[0]);
    if (p == nullptr) {
        mexWarnMsgIdAndTxt("cudss:AlreadyDestroyed",
            "Handle is not valid (already destroyed or never created); "
            "destroy is a no-op.");
        return;
    }

    // Remove from the ledger BEFORE destroying so a partial-destroy
    // failure cannot leave a dangling pointer for at_exit_cleanup to
    // trip over.
    ledger_remove(p);

    // destroyObject (class_handle.hpp) fires the wrapper's destructor,
    // which runs ~SolverState() (all CUDA / cuDSS teardown) and then
    // mexUnlock to balance the mexLock that convertPtr2Mat fired during
    // factor.
    destroyObject<SolverState>(p);
}

// ===========================================================================
//  handle_wait
//
//  Blocks until queued GPU work on the solver's stream has completed.
//  Used after an async factor to make the handle observably equivalent
//  to one produced by a synchronous factor.  Also surfaces any error or
//  zero/negative-pivot info captured by the async worker.  No-op if no
//  factor is in flight.
// ===========================================================================

static void handle_wait(int nlhs, mxArray *plhs[],
                        int nrhs, const mxArray *prhs[])
{
    (void)nlhs; (void)plhs;
    if (nrhs != 1) {
        mexErrMsgIdAndTxt("cudss:BadNargin",
            "cudss_solver('wait', ...) expects 1 trailing input (handle).");
    }

    SolverState *state = convertMat2Ptr<SolverState>(prhs[0]);

    if (state->factor_thread != nullptr) {
        if (state->factor_thread->joinable()) {
            state->factor_thread->join();
        }
        delete state->factor_thread;
        state->factor_thread = nullptr;

        if (state->worker_analysis_status != CUDSS_STATUS_SUCCESS) {
            cudssStatus_t s = state->worker_analysis_status;
            mexErrMsgIdAndTxt("cudss:CudssError",
                "Async cudssExecute(CUDSS_PHASE_ANALYSIS) failed with "
                "status %s (%d).  See cudss_solver factor handler.",
                cudss_status_name(s), static_cast<int>(s));
        }
        if (state->worker_factorization_status != CUDSS_STATUS_SUCCESS) {
            cudssStatus_t s = state->worker_factorization_status;
            mexErrMsgIdAndTxt("cudss:CudssError",
                "Async cudssExecute(CUDSS_PHASE_FACTORIZATION) failed with "
                "status %s (%d).  See cudss_solver factor handler.",
                cudss_status_name(s), static_cast<int>(s));
        }
        if (state->worker_data_info_status != CUDSS_STATUS_SUCCESS) {
            cudssStatus_t s = state->worker_data_info_status;
            mexErrMsgIdAndTxt("cudss:CudssError",
                "Async cudssDataGet(CUDSS_DATA_INFO) failed with "
                "status %s (%d).",
                cudss_status_name(s), static_cast<int>(s));
        }
        if (state->worker_factor_info != 0) {
            int info = state->worker_factor_info;
            mexErrMsgIdAndTxt("cudss:NumericalError",
                "cuDSS factorization completed with CUDSS_DATA_INFO = %d "
                "(likely zero/negative pivot -- matrix may be singular).",
                info);
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(state->stream));

    if (state->factor_in_flight) {
        if (state->b_dummy_pending   != nullptr) { cudssMatrixDestroy(state->b_dummy_pending); state->b_dummy_pending = nullptr; }
        if (state->x_dummy_pending   != nullptr) { cudssMatrixDestroy(state->x_dummy_pending); state->x_dummy_pending = nullptr; }
        if (state->d_b_dummy_pending != nullptr) { cudaFree(state->d_b_dummy_pending);         state->d_b_dummy_pending = nullptr; }
        if (state->d_x_dummy_pending != nullptr) { cudaFree(state->d_x_dummy_pending);         state->d_x_dummy_pending = nullptr; }
        state->factor_in_flight = false;
    }
}

// ===========================================================================
//  mexFunction  -- string-command dispatcher
//
//  Entry point that MATLAB calls.  The first argument is the command
//  string; the remaining arguments are forwarded to the matching
//  handle_* function.  ensure_at_exit_registered installs the
//  at_exit_cleanup hook on the first call.
// ===========================================================================

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    ensure_at_exit_registered();

    if (nrhs < 1 || !mxIsChar(prhs[0])) {
        mexErrMsgIdAndTxt("cudss:BadNargin",
            "First argument must be a command string: "
            "'factor', 'solve', 'destroy', or 'wait'.");
    }

    char cmd[16] = {0};
    if (mxGetString(prhs[0], cmd, sizeof(cmd)) != 0) {
        mexErrMsgIdAndTxt("cudss:BadInput",
            "Command string longer than 15 characters.");
    }

    int sub_nrhs = nrhs - 1;
    const mxArray **sub_prhs = (nrhs > 1) ? prhs + 1 : nullptr;

    if (std::strcmp(cmd, "factor") == 0) {
        handle_factor(nlhs, plhs, sub_nrhs, sub_prhs);
    } else if (std::strcmp(cmd, "solve") == 0) {
        handle_solve(nlhs, plhs, sub_nrhs, sub_prhs);
    } else if (std::strcmp(cmd, "destroy") == 0) {
        handle_destroy(nlhs, plhs, sub_nrhs, sub_prhs);
    } else if (std::strcmp(cmd, "wait") == 0) {
        handle_wait(nlhs, plhs, sub_nrhs, sub_prhs);
    } else {
        mexErrMsgIdAndTxt("cudss:BadCommand",
            "Unknown command '%s' (expected 'factor', 'solve', "
            "'destroy', or 'wait').", cmd);
    }
}
