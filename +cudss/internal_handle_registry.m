function out = internal_handle_registry(action, handle)
%CUDSS.INTERNAL_HANDLE_REGISTRY  MATLAB-side persistent set of live handles.
%
%   cudss.internal_handle_registry('add', handle)
%   cudss.internal_handle_registry('remove', handle)
%   tf = cudss.internal_handle_registry('contains', handle)
%   n  = cudss.internal_handle_registry('count')
%   cudss.internal_handle_registry('clear')
%
%   Internal helper for the +cudss package.  Provides a fast MATLAB-side
%   check that a uint64 handle is currently live, layered on top of the
%   signature check inside the MEX (class_handle.hpp).  This makes
%   double-destroy and stale-handle errors surface before the MEX
%   boundary with a clean MATLAB-level stack trace.

    persistent registry
    if isempty(registry)
        registry = uint64.empty(0, 1);
    end

    switch lower(action)
        case 'add'
            assert(isscalar(handle) && isa(handle, 'uint64'));
            registry(end+1, 1) = handle; %#ok<AGROW>
        case 'remove'
            registry(registry == handle) = [];
        case 'contains'
            out = any(registry == handle);
            return
        case 'count'
            out = numel(registry);
            return
        case 'clear'
            registry = uint64.empty(0, 1);
        otherwise
            error('cudss:InternalRegistry', 'Unknown action: %s', action);
    end
end
