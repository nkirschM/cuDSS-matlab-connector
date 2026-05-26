// class_handle.hpp - uint64 <-> C++ pointer marshalling for MEX.
//
// HOW IT WORKS
// ------------
// MATLAB has no way to hold a C++ object directly, so we hand it the
// address of one as a uint64 scalar.  convertPtr2Mat creates a new
// class_handle wrapper around a heap-allocated base* (here:
// SolverState*), packs the wrapper's address into a uint64 mxArray, and
// calls mexLock so MATLAB cannot drop the MEX while live state still
// exists.  convertMat2Ptr does the reverse: it casts the uint64 back to
// a wrapper, validates that it still looks live, and returns the inner
// base*.  destroyObject deletes the wrapper (which deletes the inner
// pointer and runs its destructor) and pairs the mexLock with a
// mexUnlock.
//
// STALE-HANDLE DETECTION
// ----------------------
// class_handle stores a magic signature and the typeid name at
// construction.  isValid() checks both, so a uint64 that no longer
// points at a live wrapper (already destroyed; never came from
// convertPtr2Mat) is rejected before we dereference anything.  This is
// best-effort across address-space reuse; the +cudss MATLAB wrappers
// carry an additional MATLAB-side persistent set as a second-layer
// check (see internal_handle_registry.m).
//
// Pattern is adapted from Oliver Woodford's class_handle pattern on
// MATLAB File Exchange ("A class for managing C++ objects from Matlab").

#ifndef CLASS_HANDLE_HPP
#define CLASS_HANDLE_HPP

#include "mex.h"
#include <cstdint>
#include <cstring>
#include <string>
#include <typeinfo>

#define CLASS_HANDLE_SIGNATURE 0xFF00F0A5

template <class base>
class class_handle {
public:
    explicit class_handle(base *ptr)
        : signature_m(CLASS_HANDLE_SIGNATURE),
          name_m(typeid(base).name()),
          ptr_m(ptr) {}

    ~class_handle() {
        // Zero the signature so any subsequent isValid() on this address
        // fails -- guards against use-after-free if a stale uint64 is
        // passed in after destroy.
        signature_m = 0;
        delete ptr_m;
        ptr_m = nullptr;
    }

    bool isValid() const {
        return signature_m == CLASS_HANDLE_SIGNATURE &&
               std::strcmp(name_m.c_str(), typeid(base).name()) == 0;
    }

    base *ptr() { return ptr_m; }

private:
    uint32_t    signature_m;
    std::string name_m;
    base       *ptr_m;
};

// Wrap a freshly-heap-allocated base* and return its uint64 form to
// MATLAB.  mexLock pins the MEX in memory until the matching
// destroyObject runs.
template <class base>
inline mxArray *convertPtr2Mat(base *ptr) {
    mexLock();
    mxArray *out = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    *static_cast<uint64_t *>(mxGetData(out)) =
        reinterpret_cast<uint64_t>(new class_handle<base>(ptr));
    return out;
}

// Returns the wrapper pointer, or nullptr if the input does not look
// like a live class_handle of the requested type.  Does NOT throw.
// Used by handle_destroy so a double-destroy warns instead of erroring.
template <class base>
inline class_handle<base> *tryConvertMat2HandlePtr(const mxArray *in) {
    if (in == nullptr) return nullptr;
    if (mxGetNumberOfElements(in) != 1) return nullptr;
    if (mxGetClassID(in) != mxUINT64_CLASS) return nullptr;
    if (mxIsComplex(in)) return nullptr;
    uint64_t raw = *static_cast<uint64_t *>(mxGetData(in));
    if (raw == 0) return nullptr;
    auto *p = reinterpret_cast<class_handle<base> *>(raw);
    if (!p->isValid()) return nullptr;
    return p;
}

// Throws cudss:InvalidHandle on any failure.  Used by handle_solve and
// handle_wait, where a stale handle is a programming error and a clean
// MATLAB error message is the right thing to surface.
template <class base>
inline class_handle<base> *convertMat2HandlePtr(const mxArray *in) {
    auto *p = tryConvertMat2HandlePtr<base>(in);
    if (p == nullptr) {
        mexErrMsgIdAndTxt(
            "cudss:InvalidHandle",
            "Handle must be a live uint64 scalar produced by cudss.factor "
            "and not yet passed to cudss.destroy.");
    }
    return p;
}

// Convenience overload that returns the inner pointer.
template <class base>
inline base *convertMat2Ptr(const mxArray *in) {
    return convertMat2HandlePtr<base>(in)->ptr();
}

// Delete the wrapper (which deletes the inner pointer and runs its
// destructor), then pair the mexLock from convertPtr2Mat with a
// mexUnlock so the MEX can eventually be cleared.
template <class base>
inline void destroyObject(class_handle<base> *p) {
    delete p;
    mexUnlock();
}

#endif
