function W = solve(handle, L)
%CUDSS.SOLVE  Apply a persistent cuDSS factorization to a multi-RHS block.
%
%   W = cudss.solve(handle, L)
%
%   Solves K * W = L on the GPU using the factorization computed by
%   cudss.factor(K, ...).  Both L and W are gpuArrays; no host round-trip
%   occurs inside this call.
%
%   INPUTS
%     handle: uint64 scalar from cudss.factor.  Must not have been destroyed.
%     L     : n x nrhs gpuArray of class matching the handle's precision
%             ('single' for an FP32 handle, 'double' for FP64).  Host arrays
%             are rejected -- callers must move L to the GPU explicitly so
%             implicit transfers do not silently dominate the solve cost.
%
%   OUTPUT
%     W     : n x nrhs gpuArray of the same class as L.

    arguments
        handle (1,1) uint64
        L
    end

    % --- handle must be live (MATLAB-side check, before MEX boundary) ---
    if ~cudss.internal_handle_registry('contains', handle)
        error('cudss:InvalidHandle', ...
              'Handle is not registered (already destroyed, or never produced by cudss.factor).');
    end

    % --- L must be a gpuArray ---
    if ~isa(L, 'gpuArray')
        error('cudss:WrongInputType', ...
              'L must be a gpuArray.  Move it with gpuArray(...) before calling cudss.solve.');
    end
    if ~existsOnGPU(L)
        error('cudss:WrongInputType', 'L is a stale gpuArray (existsOnGPU returned false).');
    end

    if ~ismatrix(L)
        error('cudss:BadInput', 'L must be 2-D.');
    end

    % Precision (class(L) vs handle precision) is enforced inside the
    % MEX, which has direct access to is_single on the SolverState.
    W = cudss_solver('solve', handle, L);
end
