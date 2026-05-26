function destroy(handle)
%CUDSS.DESTROY  Release a cuDSS factorization handle and all its GPU memory.
%
%   cudss.destroy(handle)
%
%   Idempotent: calling on an already-destroyed handle issues a warning
%   ('cudss:AlreadyDestroyed') and returns.  This makes defensive
%   double-destroy in user code / onCleanup harmless.
%
%   After destroy returns, gpuDevice().AvailableMemory rises by the
%   factor's footprint (input CSR + LU factors + workspace).

    arguments
        handle (1,1) uint64
    end

    if ~cudss.internal_handle_registry('contains', handle)
        warning('cudss:AlreadyDestroyed', ...
                'Handle is not registered; destroy is a no-op.');
        return
    end

    % Remove from registry FIRST so a subsequent call sees a stale handle
    % even if the MEX itself raises mid-teardown.
    cudss.internal_handle_registry('remove', handle);

    cudss_solver('destroy', handle);
end
