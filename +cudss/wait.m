function wait(handle)
%CUDSS.WAIT  Block until the solver's queued GPU work is complete.
%
%   cudss.wait(handle)
%
%   When cudss.factor was called with opts.async = true, the analysis +
%   factorization phases are queued onto the solver's dedicated CUDA stream
%   and the factor call returns without blocking.  cudss.wait drains that
%   stream and releases the deferred dummy buffers used during analysis /
%   factorization.  After wait returns, the handle is observably equivalent
%   to one produced by a synchronous cudss.factor call.
%
%   It is also safe to call wait on a handle that has no in-flight work --
%   the underlying cudaStreamSynchronize is a no-op against an idle stream.
%
%   INPUTS
%     handle: uint64 scalar from cudss.factor.  Must not have been destroyed.

    arguments
        handle (1,1) uint64
    end

    if ~cudss.internal_handle_registry('contains', handle)
        error('cudss:InvalidHandle', ...
              'Handle is not registered (already destroyed, or never produced by cudss.factor).');
    end

    cudss_solver('wait', handle);
end
