// solver_state.hpp - persistent cuDSS state for one factored matrix.
//
// One SolverState lives behind every uint64 handle returned by
// cudss.factor.  It owns:
//   - the cuDSS handle, config, data, and sparse matrix descriptor
//   - a dedicated CUDA stream and a reusable sync event
//   - the device-resident CSR arrays (row offsets, column indices,
//     values) for the factored matrix
//   - async-factor bookkeeping: the worker std::thread, captured
//     status codes, and the dummy b/x buffers cuDSS reads during
//     ANALYSIS / FACTORIZATION
//
// All teardown happens in ~SolverState via RAII, so a failure
// mid-construction (in the factor handler's try/catch) or a normal
// destroy / mexAtExit cannot leak device memory or cuDSS handles.
//
// This header also defines the CUDSS_CHECK / CUDA_CHECK and
// CUDSS_THROW / CUDA_THROW error-handling macros used by cudss_solver.cu.
// CHECK macros raise a MATLAB error directly via mexErrMsgIdAndTxt --
// safe on paths with no C++ state to unwind.  THROW macros raise a
// std::runtime_error -- required on the factor construction path where
// the unique_ptr holding the in-progress SolverState must run its
// destructor before the error is surfaced to MATLAB.

#ifndef SOLVER_STATE_HPP
#define SOLVER_STATE_HPP

#include <cuda_runtime.h>
#include <cudss.h>
#include <library_types.h>

#include <thread>

struct SolverState {
    // ---- cuDSS lifecycle objects ----
    cudssHandle_t handle = nullptr;
    cudssConfig_t config = nullptr;
    cudssData_t   data   = nullptr;
    cudssMatrix_t A      = nullptr;
    cudaStream_t  stream = nullptr;

    // Reusable event used at the top of every SOLVE to make state->stream
    // wait on whatever MATLAB queued on the legacy default stream
    // (stream 0) for L_gpu / W_gpu.  Today this is also implicitly
    // enforced because cudaStreamCreate (default flags) returns a
    // *blocking* stream that serializes with stream 0; the event-wait
    // is a belt-and-suspenders that survives a future switch to a
    // non-blocking stream or to per-thread default streams.
    cudaEvent_t   evt_default = nullptr;

    // ---- problem dimensions ----
    int n   = 0;
    int nnz = 0;

    // Value type and a mirrored bool for fast precision checks at solve
    // time.  CUDA_R_32F = float (single), CUDA_R_64F = double.
    cudaDataType_t value_type = CUDA_R_32F;
    bool           is_single  = true;

    // ---- device-resident CSR arrays (owned; freed in destructor) ----
    int  *d_row_offsets = nullptr;  // (n + 1) int32 entries
    int  *d_col_indices = nullptr;  // nnz   int32 entries
    void *d_values      = nullptr;  // nnz * sizeof(value_type) bytes

    // Optional metadata (reserved; not used today).  Will hold
    // iterative-refinement step count and hybrid-memory mode once
    // those config knobs are exposed.
    int  ir_steps    = 0;
    bool hybrid_mode = false;

    // CUDSS_DATA_INFO captured immediately after FACTORIZATION.  cuDSS
    // can return CUDSS_STATUS_SUCCESS while leaving a non-zero info
    // code that signals a numerical issue (e.g., zero/negative pivot at
    // row `info`).  The sync path checks this inline and raises before
    // convertPtr2Mat; the async path captures it on the worker thread
    // (after a stream sync so the value is settled) and the joining
    // caller surfaces it as a MATLAB error before any solve runs.
    int           worker_factor_info        = 0;
    cudssStatus_t worker_data_info_status   = CUDSS_STATUS_SUCCESS;

    // ---- async-factor worker state ----
    //
    // When the factor handler is called with opts.async = true, a
    // std::thread runs cudssExecute(ANALYSIS) + cudssExecute(FACTORIZATION).
    // This hides cuDSS's host-side METIS reordering / symbolic factor
    // (single-threaded inside the library, potentially ~seconds on
    // large sparse matrices) behind unrelated GPU or CPU work the
    // caller queues concurrently.  The numeric factor kernels are
    // queued onto state->stream so they also overlap.
    //
    // Two layers of in-flight state need to outlive the MEX call:
    //   1. The worker thread itself.  Joined by handle_wait,
    //      handle_solve, or ~SolverState before any code observes the
    //      factor result.
    //   2. The dummy b/x descriptors and their device buffers.  cuDSS
    //      reads these during ANALYSIS / FACTORIZATION, so freeing
    //      them eagerly would race with the worker.  Released after
    //      the worker joins.
    //
    // The worker captures cudssStatus_t codes here instead of calling
    // mexErrMsgIdAndTxt directly: longjmp out of a non-MATLAB thread
    // is undefined behavior.  The joining thread inspects these and
    // raises a MATLAB error if either phase failed.
    std::thread   *factor_thread = nullptr;
    cudssStatus_t  worker_analysis_status      = CUDSS_STATUS_SUCCESS;
    cudssStatus_t  worker_factorization_status = CUDSS_STATUS_SUCCESS;

    void          *d_b_dummy_pending = nullptr;
    void          *d_x_dummy_pending = nullptr;
    cudssMatrix_t  b_dummy_pending   = nullptr;
    cudssMatrix_t  x_dummy_pending   = nullptr;
    bool           factor_in_flight  = false;

    SolverState() = default;

    SolverState(const SolverState &) = delete;
    SolverState &operator=(const SolverState &) = delete;

    ~SolverState() {
        // Join the async-factor worker first so it is no longer reading
        // anything we are about to tear down (cuDSS handle, dummy
        // descriptors, device buffers).  join() also acts as a memory
        // barrier so subsequent reads of worker_*_status are visible.
        if (factor_thread != nullptr) {
            if (factor_thread->joinable()) {
                factor_thread->join();
            }
            delete factor_thread;
            factor_thread = nullptr;
        }
        // After join the worker has returned, but the GPU stream may
        // still hold queued numeric-factor kernels.  Drain before
        // releasing the dummy buffers they read.
        if (factor_in_flight && stream != nullptr) {
            cudaStreamSynchronize(stream);
        }
        if (b_dummy_pending   != nullptr) { cudssMatrixDestroy(b_dummy_pending); b_dummy_pending = nullptr; }
        if (x_dummy_pending   != nullptr) { cudssMatrixDestroy(x_dummy_pending); x_dummy_pending = nullptr; }
        if (d_b_dummy_pending != nullptr) { cudaFree(d_b_dummy_pending);         d_b_dummy_pending = nullptr; }
        if (d_x_dummy_pending != nullptr) { cudaFree(d_x_dummy_pending);         d_x_dummy_pending = nullptr; }
        factor_in_flight = false;

        // Reverse order of creation.  Every cuDSS / CUDA handle is
        // checked for null because RAII may run after a partial
        // construction failure.
        if (A      != nullptr) { cudssMatrixDestroy(A);          A = nullptr;      }
        if (data   != nullptr) { cudssDataDestroy(handle, data); data = nullptr;   }
        if (config != nullptr) { cudssConfigDestroy(config);     config = nullptr; }
        if (handle != nullptr) { cudssDestroy(handle);           handle = nullptr; }
        if (evt_default != nullptr) { cudaEventDestroy(evt_default); evt_default = nullptr; }
        if (stream != nullptr) { cudaStreamDestroy(stream);      stream = nullptr; }
        if (d_values      != nullptr) { cudaFree(d_values);      d_values      = nullptr; }
        if (d_col_indices != nullptr) { cudaFree(d_col_indices); d_col_indices = nullptr; }
        if (d_row_offsets != nullptr) { cudaFree(d_row_offsets); d_row_offsets = nullptr; }
    }
};

// Translate cudssStatus_t into a printable name for error messages.
// Inline so each .cu can use it without a separate .cpp.
inline const char *cudss_status_name(cudssStatus_t s) {
    switch (s) {
        case CUDSS_STATUS_SUCCESS:               return "CUDSS_STATUS_SUCCESS";
        case CUDSS_STATUS_NOT_INITIALIZED:       return "CUDSS_STATUS_NOT_INITIALIZED";
        case CUDSS_STATUS_ALLOC_FAILED:          return "CUDSS_STATUS_ALLOC_FAILED";
        case CUDSS_STATUS_INVALID_VALUE:         return "CUDSS_STATUS_INVALID_VALUE";
        case CUDSS_STATUS_NOT_SUPPORTED:         return "CUDSS_STATUS_NOT_SUPPORTED";
        case CUDSS_STATUS_EXECUTION_FAILED:      return "CUDSS_STATUS_EXECUTION_FAILED";
        case CUDSS_STATUS_INTERNAL_ERROR:        return "CUDSS_STATUS_INTERNAL_ERROR";
        default:                                 return "CUDSS_STATUS_UNKNOWN";
    }
}

// ---------------------------------------------------------------------------
//  Error-handling macros
//
//  *_CHECK -- on failure, call mexErrMsgIdAndTxt and never return.
//             Safe to use only where there is no C++ state that needs
//             to be unwound (no live unique_ptr<SolverState>, no owned
//             cuDSS handles outside SolverState).
//
//  *_THROW -- on failure, raise a std::runtime_error.  Used on the
//             factor construction path where unique_ptr<SolverState>
//             owns in-progress state; the surrounding try/catch in the
//             factor handler converts the exception into a MATLAB
//             error AFTER the destructor has cleaned up.
// ---------------------------------------------------------------------------

#define CUDSS_CHECK(call)                                                      \
    do {                                                                       \
        cudssStatus_t _s = (call);                                             \
        if (_s != CUDSS_STATUS_SUCCESS) {                                      \
            mexErrMsgIdAndTxt(                                                 \
                "cudss:CudssError",                                            \
                "%s failed at %s:%d with status %s (%d).",                     \
                #call, __FILE__, __LINE__, cudss_status_name(_s),              \
                static_cast<int>(_s));                                         \
        }                                                                      \
    } while (0)

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            mexErrMsgIdAndTxt(                                                 \
                "cudss:CudaError",                                             \
                "%s failed at %s:%d with cudaError %s (%d).",                  \
                #call, __FILE__, __LINE__, cudaGetErrorString(_e),             \
                static_cast<int>(_e));                                         \
        }                                                                      \
    } while (0)

#include <stdexcept>
#include <string>

// Typed exception for numerical (non-API) factorization failures -- e.g.,
// CUDSS_DATA_INFO != 0 from a zero/negative pivot.  Distinguished from a
// generic std::runtime_error so the factor handler's catch block can route
// it to cudss:NumericalError (the MATLAB error ID the async path also uses
// for the same condition), rather than the catch-all cudss:Error reserved
// for unexpected API / runtime failures.
struct NumericalError : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

#define CUDSS_THROW(call)                                                      \
    do {                                                                       \
        cudssStatus_t _s = (call);                                             \
        if (_s != CUDSS_STATUS_SUCCESS) {                                      \
            throw std::runtime_error(                                          \
                std::string(#call) + " failed at " + __FILE__ + ":" +          \
                std::to_string(__LINE__) + " with status " +                   \
                cudss_status_name(_s) + " (" +                                 \
                std::to_string(static_cast<int>(_s)) + ").");                  \
        }                                                                      \
    } while (0)

#define CUDA_THROW(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            throw std::runtime_error(                                          \
                std::string(#call) + " failed at " + __FILE__ + ":" +          \
                std::to_string(__LINE__) + " with cudaError " +                \
                cudaGetErrorString(_e) + " (" +                                \
                std::to_string(static_cast<int>(_e)) + ").");                  \
        }                                                                      \
    } while (0)

#endif
