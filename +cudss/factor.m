function handle = factor(K, opts)
%CUDSS.FACTOR  Compute and persist a sparse LU factorization of K on the GPU.
%
%   handle = cudss.factor(K)
%   handle = cudss.factor(K, opts)
%
%   Wraps NVIDIA cuDSS.  The factorization is computed once on the GPU and
%   held in device memory until cudss.destroy(handle) is called.  Each
%   subsequent cudss.solve(handle, L) applies the factors to a multi-RHS
%   gpuArray block without re-factoring.
%
%   INPUTS
%     K     : n x n sparse, real, square matrix.  Symmetry / definiteness
%             are NOT validated -- they are declared via opts.matrix_type
%             (see below) and the caller is responsible for the declaration
%             being true.  The default 'general' makes no assumption about K
%             and factors via full LU.
%     opts  : (optional) struct with fields
%               .precision    'single' (default) | 'double'.  Must match the
%                             storage class of K (single sparse K with
%                             precision='single', double sparse with 'double').
%               .matrix_type  'general' (default) | 'symmetric' | 'spd'.
%                             Selects the cuDSS matrix type:
%                               'general'  -> CUDSS_MTYPE_GENERAL,  full LU
%                               'symmetric'-> CUDSS_MTYPE_SYMMETRIC, LDL^T
%                               'spd'      -> CUDSS_MTYPE_SPD,       Cholesky
%                             For 'symmetric' and 'spd' the wrapper extracts
%                             triu(K) before transposing and tells cuDSS the
%                             matrix view is UPPER, so only the upper
%                             triangle is transferred to and stored on the
%                             device.  Symmetry of K is NOT checked; if K is
%                             not actually symmetric the solve silently uses
%                             triu(K) and treats the lower triangle as if it
%                             equaled triu(K)^T.
%               .reordering   'default'.  Other reordering algorithms are
%                             not exposed by the wrapper today.
%               .async        false (default) | true.  When true, analysis
%                             and factorization are queued onto the solver's
%                             CUDA stream and the call returns without
%                             blocking.  Call cudss.wait(handle) before
%                             reading any result that depends on the factor;
%                             cudss.solve on the same handle is also safe
%                             (same stream, ordered) and will defensively
%                             flush any in-flight factor state.
%               .verbose      false (default) | true.  Print per-phase
%                             timing breakdowns from MATLAB and the MEX.
%
%   OUTPUT
%     handle: uint64 scalar opaque token.  Must be passed to cudss.solve
%             and eventually cudss.destroy.
%
%   CSC <-> CSR NOTE
%     MATLAB sparse matrices are stored in CSC; cuDSS expects CSR.  For a
%     non-symmetric matrix the layouts are NOT interchangeable -- passing
%     CSC arrays as CSR silently solves K^T instead of K.  The wrapper
%     uses the identity CSC(K^T) == CSR(K): we transpose K on the MATLAB
%     side and the MEX reads Jc / Ir / values directly as CSR row offsets,
%     column indices, and values.

    arguments
        K      (:,:) {mustBeNumeric}
        opts   (1,1) struct = struct()
    end

    % --- handle input validation up-front so errors surface in MATLAB land ---
    if ~issparse(K)
        error('cudss:BadInput', 'K must be sparse.');
    end
    if ~isreal(K)
        error('cudss:BadInput', 'K must be real.');
    end
    if size(K, 1) ~= size(K, 2)
        error('cudss:BadInput', 'K must be square (got %d x %d).', ...
              size(K, 1), size(K, 2));
    end
    if any(~isfinite(nonzeros(K)))
        error('cudss:BadInput', 'K contains non-finite entries.');
    end

    % --- defaults + validate opts ---
    precision = getfield_default(opts, 'precision', 'single');
    if ~(ischar(precision) || (isstring(precision) && isscalar(precision)))
        error('cudss:BadOpts', 'opts.precision must be a char/string.');
    end
    precision = char(precision);
    if ~(strcmp(precision, 'single') || strcmp(precision, 'double'))
        error('cudss:BadOpts', ...
              "opts.precision must be 'single' or 'double' (got '%s').", ...
              precision);
    end

    reordering = getfield_default(opts, 'reordering', 'default');
    if ~strcmp(reordering, 'default')
        warning('cudss:UnsupportedReordering', ...
                "opts.reordering='%s' is not exposed by this wrapper " + ...
                "(only 'default' is honored).", reordering);
    end

    matrix_type = getfield_default(opts, 'matrix_type', 'general');
    if ~(ischar(matrix_type) || (isstring(matrix_type) && isscalar(matrix_type)))
        error('cudss:BadOpts', 'opts.matrix_type must be a char/string.');
    end
    matrix_type = char(matrix_type);
    if ~ismember(matrix_type, {'general', 'symmetric', 'spd'})
        error('cudss:BadOpts', ...
              "opts.matrix_type must be 'general', 'symmetric', or 'spd' (got '%s').", ...
              matrix_type);
    end

    async_factor = getfield_default(opts, 'async', false);
    if ~(islogical(async_factor) && isscalar(async_factor))
        error('cudss:BadOpts', 'opts.async must be a logical scalar.');
    end

    verbose = getfield_default(opts, 'verbose', false);
    if ~(islogical(verbose) && isscalar(verbose))
        error('cudss:BadOpts', 'opts.verbose must be a logical scalar.');
    end

    % --- cast to requested precision if needed (R2024b+: native single sparse) ---
    if verbose, t_cast = tic; end
    if strcmp(precision, 'single') && ~isa(K, 'single')
        K = single(K);
    elseif strcmp(precision, 'double') && ~isa(K, 'double')
        K = double(K);
    end
    if verbose, fprintf('  [cudss.factor] sparse cast: %.3fs\n', toc(t_cast)); end

    % --- CSC of K' == CSR of K.  This is the entire conversion. ---
    % For symmetric / spd: extract triu(K) first so the device only sees the
    % upper triangle.  The CSC bytes of triu(K).' (== tril(K) for symmetric K)
    % are bit-for-bit the CSR bytes of triu(K), so the MEX then announces
    % view=UPPER and cuDSS reads the matrix correctly while halving the nnz
    % transferred / stored.
    if verbose, t_trans = tic; end
    switch matrix_type
        case 'general'
            Kt = K.';
        case {'symmetric', 'spd'}
            Kt = triu(K).';
    end
    if verbose, fprintf('  [cudss.factor] sparse transpose: %.3fs\n', toc(t_trans)); end

    % --- pack opts into the struct shape the MEX expects ---
    mex_opts = struct('precision',   precision, ...
                      'matrix_type', matrix_type, ...
                      'async',       logical(async_factor), ...
                      'verbose',     logical(verbose));

    if verbose, t_mex = tic; end
    handle = cudss_solver('factor', Kt, mex_opts);
    if verbose, fprintf('  [cudss.factor] MEX (analysis + queue factor): %.3fs\n', toc(t_mex)); end

    % Persistent registry of valid handles -- a MATLAB-side mirror of
    % the C++ signature check inside class_handle.  Used by cudss.solve
    % to reject stale handles before the MEX boundary and by
    % cudss.destroy for idempotent double-destroy semantics.
    cudss.internal_handle_registry('add', handle);
end

% ============================ helpers ====================================

function v = getfield_default(s, name, default_v)
    if isfield(s, name)
        v = s.(name);
        if isempty(v)
            v = default_v;
        end
    else
        v = default_v;
    end
end

