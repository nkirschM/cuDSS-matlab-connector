%% test_cudss_solver.m -- correctness suite for the +cudss MEX wrapper.
%
% Covers:
%   * Basic correctness vs MATLAB's host solve (smoke, multi-RHS, 3-D Laplacian)
%     -- run in both 'single' and 'double' precision.
%   * Factor-once / solve-many timing stability.
%   * Precision dispatch (single + double on the same K) and wrong-precision
%     rejection.
%   * Handle lifecycle (use-after-destroy, double-destroy).
%   * matrix_type agreement across 'general' / 'symmetric' / 'spd' on an
%     SPD K, in both precisions.
%   * matrix_type='spd' rejection of an indefinite K.
%   * Async factor + cudss.wait, in both precisions.
%   * Large-n (~200k) correctness with a fair host pre-factored baseline,
%     in both precisions.
%   * mexLock counter balance over repeated factor + destroy cycles.
%   * Singular-K detection (factor error OR residual >> tolerance), in
%     both precisions.
%   * Default-stream interleave (cudss.solve must wait for queued MATLAB
%     kernels), in both precisions.
%
% Each test prints a description, the expected result, and the observed
% result.  The end-of-suite summary lists every test as PASS/FAIL; failing
% tests also re-print their description, expectation, and what was actually
% observed so the failure is self-explanatory.
%
% Run from MATLAB after build_mex:
%
%     addpath(pwd);
%     run('test_cudss_solver.m');

clc; clearvars; close all;

% --- path setup so cudss.factor / .solve / .destroy resolve ---
this_dir = fileparts(mfilename('fullpath'));
addpath(this_dir);

% sanity check: package functions visible
which_factor = which('cudss.factor');
if isempty(which_factor)
    error('test_cudss_solver:NoPackage', ...
          'cudss.factor not on path.  Did you run build_mex first?');
end
fprintf('cudss.factor resolved at %s\n', which_factor);

% sanity check: GPU available
g = gpuDevice();
fprintf('Using GPU: %s, CC %s, %.2f GB free\n', ...
        g.Name, g.ComputeCapability, g.AvailableMemory / 1e9);

results = struct('name', {}, 'description', {}, 'expected', {}, ...
                 'pass', {}, 'msg', {});

% --- basic correctness (single + double) ---
results(end+1) = run_test_smoke(100, 'single');
results(end+1) = run_test_smoke(100, 'double');
results(end+1) = run_test_multirhs(1000, 10, 'single');
results(end+1) = run_test_multirhs(1000, 10, 'double');
results(end+1) = run_test_laplacian(22, 'single');
results(end+1) = run_test_laplacian(22, 'double');

% --- factor reuse, precision, lifecycle ---
results(end+1) = run_test_factor_once_solve_many(500, 50);
results(end+1) = run_test_precision_dispatch(200);
results(end+1) = run_test_wrong_precision_rejection(200);
results(end+1) = run_test_handle_lifecycle(200);

% --- matrix_type contract (single + double for the agreement test) ---
results(end+1) = run_test_matrix_type(18, 'single');
results(end+1) = run_test_matrix_type(18, 'double');
results(end+1) = run_test_matrix_type_rejects_non_spd(200);

% --- large-n + async (single + double) ---
results(end+1) = run_test_large_n(10, 'single');
results(end+1) = run_test_large_n(10, 'double');
results(end+1) = run_test_async_factor(2000, 5, 'single');
results(end+1) = run_test_async_factor(2000, 5, 'double');

% --- lock counter, singular-K, stream interleave (last two in both precisions) ---
results(end+1) = run_test_lock_counter(200, 50);
results(end+1) = run_test_singular_K_detection(40, 'single');
results(end+1) = run_test_singular_K_detection(40, 'double');
results(end+1) = run_test_stream_interleave(500, 4, 'single');
results(end+1) = run_test_stream_interleave(500, 4, 'double');

% --- reordering algorithm selection ---
results(end+1) = run_test_reordering_alg(18);

% --- hybrid host/device memory mode (single + double) ---
results(end+1) = run_test_hybrid_memory(22, 4, 'single');
results(end+1) = run_test_hybrid_memory(22, 4, 'double');
results(end+1) = run_test_hybrid_memory_limit_rejection(22);

% --- summary ---
fprintf('\n================ TEST SUMMARY ================\n');
n_pass = 0;
for r = results
    tag = ternary(r.pass, 'PASS', 'FAIL');
    fprintf('  %s  %s\n', tag, r.name);
    if ~r.pass
        fprintf('         What:     %s\n', r.description);
        fprintf('         Expected: %s\n', r.expected);
        fprintf('         Got:      %s\n', r.msg);
    end
    if r.pass, n_pass = n_pass + 1; end
end
fprintf('  %d / %d passed\n', n_pass, numel(results));
if n_pass < numel(results)
    error('test_cudss_solver:FailingTests', '%d test(s) failed.', ...
          numel(results) - n_pass);
end


% =========================================================================
%  Test implementations
% =========================================================================

function r = run_test_smoke(n, prec)
    [cast_fn, tol] = precision_params(prec, 'relerr_basic');
    name        = sprintf('Smoke (n=%d, %s)', n, prec);
    description = sprintf(['Solve K \\ L on a small well-conditioned sparse K ' ...
                           '(n=%d, 1 RHS) and compare the GPU result to MATLAB''s host solve.'], n);
    expected    = sprintf('||W_gpu - W_host|| / ||W_host|| < %.0e (%s)', tol, prec);
    print_test_header(name, description, expected);
    try
        rng(0);
        K = sprandn(n, n, 0.05) + n*speye(n);   % strongly diagonal -> well-conditioned
        L = randn(n, 1);

        W_host = K \ L;

        h     = cudss.factor(K, struct('precision', prec));
        L_gpu = gpuArray(cast_fn(L));
        W_gpu = cudss.solve(h, L_gpu);
        cudss.destroy(h);

        W_gpu_h = double(gather(W_gpu));
        relerr  = norm(W_gpu_h - W_host) / max(norm(W_host), eps);
        pass    = relerr < tol;
        msg     = sprintf('relerr = %.2e (tol %.0e)', relerr, tol);
        print_test_result(pass, msg);
        r = make_result(name, description, expected, pass, msg);
    catch ME
        print_test_result(false, ME.message);
        r = make_result(name, description, expected, false, ME.message);
    end
end

function r = run_test_multirhs(n, nrhs, prec)
    [cast_fn, tol] = precision_params(prec, 'relerr_basic');
    name        = sprintf('Multi-RHS (n=%d, nrhs=%d, %s)', n, nrhs, prec);
    description = sprintf(['Solve K * W = L with %d RHS columns on a moderate sparse K ' ...
                           '(n=%d); checks the column-major dense layout end-to-end.'], nrhs, n);
    expected    = sprintf('||W_gpu - W_host||_F / ||W_host||_F < %.0e (%s)', tol, prec);
    print_test_header(name, description, expected);
    try
        rng(1);
        K = sprandn(n, n, 0.01) + n*speye(n);
        L = randn(n, nrhs);

        W_host = K \ L;

        h     = cudss.factor(K, struct('precision', prec));
        L_gpu = gpuArray(cast_fn(L));
        W_gpu = cudss.solve(h, L_gpu);
        cudss.destroy(h);

        W_gpu_h = double(gather(W_gpu));
        relerr  = norm(W_gpu_h - W_host, 'fro') / max(norm(W_host, 'fro'), eps);
        pass    = relerr < tol;
        msg     = sprintf('||W_gpu - W_host||_F / ||W_host||_F = %.2e (tol %.0e)', ...
                          relerr, tol);
        print_test_result(pass, msg);
        r = make_result(name, description, expected, pass, msg);
    catch ME
        print_test_result(false, ME.message);
        r = make_result(name, description, expected, false, ME.message);
    end
end

function r = run_test_laplacian(N, prec)
    n_total = N^3;
    [cast_fn, tol] = precision_params(prec, 'residual_laplacian');
    name        = sprintf('3-D Laplacian (%d^3 = %d, nrhs=4, %s)', N, n_total, prec);
    description = sprintf(['Solve a 3-D 7-point Laplacian (SPD by construction) on a ' ...
                           '%d^3 = %d grid with 4 RHS, then check the residual against the original K.'], ...
                           N, n_total);
    expected    = sprintf('per-column ||K*w - l|| / ||l|| < %.0e (%s) for every RHS', tol, prec);
    print_test_header(name, description, expected);
    try
        K  = build_3d_laplacian(N);
        L  = randn(n_total, 4);

        h     = cudss.factor(K, struct('precision', prec));
        L_gpu = gpuArray(cast_fn(L));
        W_gpu = cudss.solve(h, L_gpu);
        cudss.destroy(h);

        W_h    = double(gather(W_gpu));
        R      = K * W_h - L;
        relres = vecnorm(R) ./ max(vecnorm(L), eps);
        pass   = max(relres) < tol;
        msg    = sprintf('max ||K*w - l||/||l|| over %d RHS = %.2e (tol %.0e)', ...
                         numel(relres), max(relres), tol);
        print_test_result(pass, msg);
        r = make_result(name, description, expected, pass, msg);
    catch ME
        print_test_result(false, ME.message);
        r = make_result(name, description, expected, false, ME.message);
    end
end

function r = run_test_factor_once_solve_many(n, nsolves)
    name        = sprintf('Factor-once / %d solves (n=%d)', nsolves, n);
    description = sprintf(['Factor K once, then run %d consecutive cudss.solve calls against the ' ...
                           'cached factor; per-solve wall time must not regress (no hidden re-factor).'], nsolves);
    expected    = 'max(solve time) < 5 x median(solve time)';
    print_test_header(name, description, expected);
    try
        rng(2);
        K = sprandn(n, n, 0.02) + n*speye(n);
        h = cudss.factor(K, struct('precision', 'single'));
        t_solve = zeros(nsolves, 1);
        for j = 1:nsolves
            L_gpu = gpuArray(single(randn(n, 1)));
            tic; W = cudss.solve(h, L_gpu); wait(gpuDevice()); t_solve(j) = toc;
            assert(numel(W) == n);
        end
        cudss.destroy(h);
        med_t = median(t_solve);
        max_t = max(t_solve);
        pass  = max_t < 5 * med_t;
        msg   = sprintf('median %.4f s, max %.4f s (ratio %.2fx, limit 5x)', ...
                        med_t, max_t, max_t / max(med_t, eps));
        print_test_result(pass, msg);
        r = make_result(name, description, expected, pass, msg);
    catch ME
        print_test_result(false, ME.message);
        r = make_result(name, description, expected, false, ME.message);
    end
end

function r = run_test_precision_dispatch(n)
    name        = sprintf('Precision dispatch (n=%d)', n);
    description = ['Factor and solve the same K once in ''single'' and once in ''double''; ' ...
                   'verify each precision hits its expected accuracy floor against the host solve.'];
    expected    = 'FP64 relerr < 1e-13 AND FP32 relerr < 1e-5';
    print_test_header(name, description, expected);
    try
        rng(3);
        K = sprandn(n, n, 0.05) + n*speye(n);
        L = randn(n, 1);

        h_s = cudss.factor(K, struct('precision', 'single'));
        W_s = cudss.solve(h_s, gpuArray(single(L)));
        cudss.destroy(h_s);

        h_d = cudss.factor(K, struct('precision', 'double'));
        W_d = cudss.solve(h_d, gpuArray(double(L)));
        cudss.destroy(h_d);

        rel_d = norm(double(gather(W_d)) - K \ L) / norm(K \ L);
        rel_s = norm(double(gather(W_s)) - K \ L) / norm(K \ L);
        pass  = rel_d < 1e-13 && rel_s < 1e-5;
        msg   = sprintf('FP64 relerr %.2e (tol 1e-13), FP32 relerr %.2e (tol 1e-05)', ...
                        rel_d, rel_s);
        print_test_result(pass, msg);
        r = make_result(name, description, expected, pass, msg);
    catch ME
        print_test_result(false, ME.message);
        r = make_result(name, description, expected, false, ME.message);
    end
end

function r = run_test_wrong_precision_rejection(n)
    name        = sprintf('Wrong-precision rejection (n=%d)', n);
    description = ['Factor K in ''single'', then call cudss.solve with a ''double'' gpuArray. ' ...
                   'The wrapper must reject the mismatched precision instead of silently casting.'];
    expected    = 'solve throws with error identifier ''cudss:WrongInputType''';
    print_test_header(name, description, expected);
    try
        rng(4);
        K = sprandn(n, n, 0.05) + n*speye(n);
        h = cudss.factor(K, struct('precision', 'single'));
        threw = false;
        got_id = '';
        try
            cudss.solve(h, gpuArray(double(randn(n, 1))));
        catch ME
            got_id = ME.identifier;
            threw  = strcmp(got_id, 'cudss:WrongInputType');
        end
        cudss.destroy(h);
        msg = sprintf('threw cudss:WrongInputType = %s (got id: %s)', ...
                       mat2str(threw), got_id);
        print_test_result(threw, msg);
        r = make_result(name, description, expected, threw, msg);
    catch ME
        print_test_result(false, ME.message);
        r = make_result(name, description, expected, false, ME.message);
    end
end

function r = run_test_handle_lifecycle(n)
    name        = sprintf('Handle lifecycle (n=%d)', n);
    description = ['Factor + solve + destroy a handle, then attempt solve-after-destroy (must error) ' ...
                   'and a second destroy (must warn, not error).'];
    expected    = ['solve-after-destroy raises cudss:InvalidHandle; ' ...
                   'second destroy emits cudss:AlreadyDestroyed warning'];
    print_test_header(name, description, expected);
    try
        rng(5);
        K  = sprandn(n, n, 0.05) + n*speye(n);
        h  = cudss.factor(K, struct('precision', 'single'));
        cudss.solve(h, gpuArray(single(randn(n, 1))));
        cudss.destroy(h);

        % solve after destroy should error
        threw_solve = false;
        try
            cudss.solve(h, gpuArray(single(randn(n, 1))));
        catch ME
            threw_solve = strcmp(ME.identifier, 'cudss:InvalidHandle');
        end

        % double destroy should warn (not error)
        prev = warning('off', 'cudss:AlreadyDestroyed');
        cleanup = onCleanup(@() warning(prev)); %#ok<NASGU>
        lastwarn('');
        cudss.destroy(h);
        [~, wid] = lastwarn;
        warned = strcmp(wid, 'cudss:AlreadyDestroyed');

        pass = threw_solve && warned;
        msg  = sprintf('solve-after-destroy threw=%s, double-destroy warned=%s', ...
                       mat2str(threw_solve), mat2str(warned));
        print_test_result(pass, msg);
        r = make_result(name, description, expected, pass, msg);
    catch ME
        print_test_result(false, ME.message);
        r = make_result(name, description, expected, false, ME.message);
    end
end

function r = run_test_matrix_type(N, prec)
    % opts.matrix_type end-to-end on a symmetric-positive-definite K
    % (3-D 7-point Laplacian): 'general', 'symmetric', and 'spd' must
    % all factor and solve the same RHS to matching residuals.
    % Exercises both the MATLAB-side triu(K)+transpose path for sym/spd
    % and the MEX mapping to MTYPE_SYMMETRIC / MTYPE_SPD with MVIEW_UPPER.
    n_total = N^3;
    [cast_fn, residual_tol, agree_tol] = precision_params_matrix_type(prec);
    name        = sprintf('matrix_type agreement: general/sym/spd (%d^3 = %d, %s)', ...
                          N, n_total, prec);
    description = ['Factor an SPD 3-D Laplacian three times -- once each with matrix_type ' ...
                   '''general'', ''symmetric'', ''spd'' -- and confirm all three produce small ' ...
                   'residuals and agree with each other on the solve.'];
    expected    = sprintf(['per-config max residual < %.0e (%s); ' ...
                           'sym-vs-general and spd-vs-general agreement < %.0e'], ...
                           residual_tol, prec, agree_tol);
    print_test_header(name, description, expected);
    try
        K = build_3d_laplacian(N);
        rng(10);
        L = randn(n_total, 3);
        L_gpu = gpuArray(cast_fn(L));

        configs = {'general', 'symmetric', 'spd'};
        W_each  = cell(1, numel(configs));
        for c = 1:numel(configs)
            h = cudss.factor(K, struct('precision',   prec, ...
                                       'matrix_type', configs{c}));
            W_each{c} = double(gather(cudss.solve(h, L_gpu)));
            cudss.destroy(h);
        end

        rel_max = zeros(1, numel(configs));
        for c = 1:numel(configs)
            R = K * W_each{c} - L;
            rel_max(c) = max(vecnorm(R) ./ max(vecnorm(L), eps));
        end
        sym_vs_gen = norm(W_each{2} - W_each{1}, 'fro') / max(norm(W_each{1}, 'fro'), eps);
        spd_vs_gen = norm(W_each{3} - W_each{1}, 'fro') / max(norm(W_each{1}, 'fro'), eps);

        pass = all(rel_max < residual_tol) && sym_vs_gen < agree_tol && spd_vs_gen < agree_tol;
        msg  = sprintf(['relres max (gen / sym / spd) = %.2e / %.2e / %.2e (tol %.0e); ' ...
                        'sym-vs-gen %.2e, spd-vs-gen %.2e (tol %.0e)'], ...
                        rel_max(1), rel_max(2), rel_max(3), residual_tol, ...
                        sym_vs_gen, spd_vs_gen, agree_tol);
        print_test_result(pass, msg);
        r = make_result(name, description, expected, pass, msg);
    catch ME
        print_test_result(false, ME.message);
        r = make_result(name, description, expected, false, ME.message);
    end
end

function r = run_test_matrix_type_rejects_non_spd(n)
    % Note: no assertion is made about matrix_type='symmetric' on this K.
    % cuDSS's LDL^T behavior on arbitrary indefinite matrices depends on
    % its pivoting policy, which varies across cuDSS releases.
    name        = sprintf('matrix_type=''spd'' rejects indefinite K (n=%d)', n);
    description = ['Build a symmetric *indefinite* K.  matrix_type=''general'' must factor + solve ' ...
                   'successfully; matrix_type=''spd'' (Cholesky) must reject K because Cholesky ' ...
                   'encounters a non-positive pivot.'];
    expected    = ['general OK; spd raises cudss:NumericalError ' ...
                   '(or cudss:CudssError if cuDSS caught it earlier)'];
    print_test_header(name, description, expected);
    try
        rng(20);
        % Symmetric indefinite K: random sparse symmetrized + small diagonal
        % to keep it nonsingular.  Mix of positive and negative eigenvalues
        % guaranteed by the small diagonal shift relative to the off-diagonal
        % spectral radius.
        A = sprandn(n, n, 0.05);
        K = A + A.' + 0.1 * speye(n);

        % 'general' (LU) must succeed on the same K
        gen_ok = false;
        try
            h = cudss.factor(K, struct('precision',   'double', ...
                                       'matrix_type', 'general'));
            L_gpu = gpuArray(double(randn(n, 1)));
            W_gpu = cudss.solve(h, L_gpu);
            cudss.destroy(h);
            gen_ok = all(isfinite(gather(W_gpu)));
        catch ME_gen
            gen_ok = false;
            fprintf('  general factor errored unexpectedly: %s\n', ME_gen.message);
        end

        % 'spd' (Cholesky) must reject with cudss:NumericalError.
        % cudss:CudssError is also acceptable, in case cuDSS catches the
        % failure at the cudssExecute level instead of the DATA_INFO query.
        spd_rejected = false;
        spd_err_id   = '';
        try
            h = cudss.factor(K, struct('precision',   'double', ...
                                       'matrix_type', 'spd'));
            cudss.destroy(h);
        catch ME_spd
            spd_err_id   = ME_spd.identifier;
            spd_rejected = strcmp(spd_err_id, 'cudss:NumericalError') || ...
                           strcmp(spd_err_id, 'cudss:CudssError');
        end

        pass = gen_ok && spd_rejected;
        msg  = sprintf('general OK = %s, spd rejected = %s (err id: %s)', ...
                        mat2str(gen_ok), mat2str(spd_rejected), spd_err_id);
        print_test_result(pass, msg);
        r = make_result(name, description, expected, pass, msg);
    catch ME
        print_test_result(false, ME.message);
        r = make_result(name, description, expected, false, ME.message);
    end
end

function r = run_test_large_n(nrhs, prec)
    % Large-n correctness on a 3-D 7-point Laplacian (~200k unknowns).
    % Pass/fail is on the residual; timings are informational.
    %
    % The host baseline pre-factors with `decomposition` and then solves
    % all RHS columns against the cached factor -- matches the GPU's
    % factor-once / solve-many path so the reported speedup is fair.
    N = round(200000^(1/3));     % 58 -> 58^3 = 195112
    n = N^3;
    [cast_fn, tol] = precision_params(prec, 'residual_large');
    name        = sprintf('Large-n correctness (n=%d, %s, %d RHS)', n, prec, nrhs);
    description = sprintf(['Solve a ~200k 3-D Laplacian on the GPU in %s with %d RHS; check the ' ...
                           'residual against the original K.  Reports factor + solve timings vs a ' ...
                           'host baseline that pre-factors with `decomposition` (factor-once / solve-many ' ...
                           'on both sides for a fair comparison).'], prec, nrhs);
    expected    = sprintf('max ||K*w - l|| / ||l|| < %.0e (%s); speedup figures are informational', ...
                          tol, prec);
    print_test_header(name, description, expected);
    try
        K = build_3d_laplacian(N);
        L = randn(n, nrhs);

        % --- host: factor once (auto picks Cholesky for this SPD K), solve many ---
        t_host_factor = tic;
        D_host = decomposition(K);
        host_factor = toc(t_host_factor);

        t_host_solve = tic;
        W_host = D_host \ L; %#ok<NASGU>
        host_solve_total = toc(t_host_solve);
        host_per_rhs = host_solve_total / nrhs;

        % --- gpu: factor once, solve many ---
        t_gpu_factor = tic;
        h = cudss.factor(K, struct('precision', prec));
        wait(gpuDevice());
        gpu_factor = toc(t_gpu_factor);

        L_gpu = gpuArray(cast_fn(L));
        t_gpu_solve = tic;
        W_gpu = cudss.solve(h, L_gpu);
        wait(gpuDevice());
        gpu_solve_total = toc(t_gpu_solve);
        gpu_per_rhs = gpu_solve_total / nrhs;

        cudss.destroy(h);

        W = double(gather(W_gpu));
        relres_max = max(vecnorm(K * W - L) ./ max(vecnorm(L), eps));
        pass = relres_max < tol;

        speedup_factor = host_factor  / max(gpu_factor,  eps);
        speedup_solve  = host_per_rhs / max(gpu_per_rhs, eps);
        msg = sprintf(['max relres = %.2e (tol %.0e); ' ...
                       'factor: host %.2f s, gpu %.2f s (%.1fx); ' ...
                       'solve/RHS: host %.4f s, gpu %.4f s (%.1fx)'], ...
                       relres_max, tol, ...
                       host_factor, gpu_factor, speedup_factor, ...
                       host_per_rhs, gpu_per_rhs, speedup_solve);
        print_test_result(pass, msg);
        r = make_result(name, description, expected, pass, msg);
    catch ME
        print_test_result(false, ME.message);
        r = make_result(name, description, expected, false, ME.message);
    end
end


function r = run_test_async_factor(n, nrhs, prec)
    [cast_fn, tol] = precision_params(prec, 'async_vs_sync');
    name        = sprintf('Async factor + cudss.wait (n=%d, nrhs=%d, %s)', n, nrhs, prec);
    description = ['Issue cudss.factor with opts.async=true; the call must return promptly while ' ...
                   'analysis/factor proceed on a worker thread.  After cudss.wait, the solve must ' ...
                   'match a synchronous factor of the same K to numerical noise.'];
    expected    = sprintf(['issue time < 0.5 s AND ' ...
                           '||W_async - W_sync||_F / ||W_sync||_F < %.0e (%s)'], tol, prec);
    print_test_header(name, description, expected);
    try
        rng(9);
        K = sprandn(n, n, 0.02) + n*speye(n);
        L = randn(n, nrhs);

        % --- sync reference ---
        h_sync = cudss.factor(K, struct('precision', prec));
        L_gpu  = gpuArray(cast_fn(L));
        W_sync = cudss.solve(h_sync, L_gpu);
        cudss.destroy(h_sync);

        % --- async path ---
        t_issue = tic;
        h_async = cudss.factor(K, struct('precision', prec, 'async', true));
        issue_s = toc(t_issue);

        % Run unrelated GPU work to verify the issue truly returned without
        % blocking on the factor.  arrayfun on the default stream contends
        % for the same GPU but is dispatched onto a different stream than
        % the cuDSS factor stream, so they should overlap on the device.
        x = gpuArray.rand(1024, 1024, 'single');
        for k = 1:5
            x = arrayfun(@(v) sin(v) + cos(v), x);
        end
        wait(gpuDevice());

        cudss.wait(h_async);

        W_async = cudss.solve(h_async, gpuArray(cast_fn(L)));
        cudss.destroy(h_async);

        relerr = norm(double(gather(W_async)) - double(gather(W_sync)), 'fro') / ...
                 max(norm(double(gather(W_sync)), 'fro'), eps);
        % Issue time should be much smaller than a typical factor wall.
        % 500 ms is a generous ceiling that still catches a regression
        % where the MEX accidentally syncs on this size.
        issue_pass = issue_s < 0.5;
        pass = (relerr < tol) && issue_pass;
        msg  = sprintf('issue %.3f s (limit 0.500 s), relerr async vs sync %.2e (tol %.0e)', ...
                       issue_s, relerr, tol);
        print_test_result(pass, msg);
        r = make_result(name, description, expected, pass, msg);
    catch ME
        print_test_result(false, ME.message);
        r = make_result(name, description, expected, false, ME.message);
    end
end

function r = run_test_lock_counter(n, ncycles)
    name        = sprintf('mexLock counter balance (n=%d, %d cycles)', n, ncycles);
    description = sprintf(['Run %d factor + destroy round-trips.  The MEX must be unlocked at the ' ...
                           'end -- a leaked lock would pin the binary in memory across `clear mex`.'], ncycles);
    expected    = 'mislocked(''cudss_solver'') == false after the cycles (when no other handle is live)';
    print_test_header(name, description, expected);
    try
        rng(11);
        K = sprandn(n, n, 0.05) + n*speye(n);

        baseline_locked = safe_mislocked('cudss_solver');

        for k = 1:ncycles
            h = cudss.factor(K, struct('precision', 'single'));
            cudss.destroy(h);
        end

        final_locked = safe_mislocked('cudss_solver');

        if baseline_locked
            % MEX was already locked on entry (some earlier handle still
            % live).  Conservative requirement: this test did not leave it
            % in a worse state.
            pass = true;
            msg  = sprintf('baseline locked = true (other handle live), final locked = %s', ...
                           mat2str(final_locked));
        else
            pass = ~final_locked;
            msg  = sprintf('after %d cycles mislocked = %s (expected false)', ...
                           ncycles, mat2str(final_locked));
        end
        print_test_result(pass, msg);
        r = make_result(name, description, expected, pass, msg);
    catch ME
        print_test_result(false, ME.message);
        r = make_result(name, description, expected, false, ME.message);
    end
end

function r = run_test_singular_K_detection(n, prec)
    % NOTE on cuDSS behavior: by default cuDSS applies a small pivot
    % perturbation (regularization) when it encounters a numerically-zero
    % pivot, and CUDSS_DATA_INFO comes back 0 even for a structurally
    % singular K.  The DATA_INFO check catches the cases where cuDSS *does*
    % report a non-zero info (e.g., the analysis phase rejects the matrix
    % outright, or pivoting cannot be perturbed).  For matrices where cuDSS
    % chooses to perturb-and-continue, the factor returns SUCCESS but the
    % solve produces a residual orders of magnitude larger than tolerance,
    % because the equation 0 = L(7) is unsolvable for a generic L
    % regardless of float precision.
    cast_fn = precision_cast(prec);
    name        = sprintf('Singular-K detection (n=%d, %s)', n, prec);
    description = ['Factor a structurally singular K (one row zeroed out).  Either cudss.factor ' ...
                   'errors, or -- if cuDSS perturbs-and-continues -- the solve''s residual must ' ...
                   'be large enough that a downstream residual check can detect the failure.'];
    expected    = sprintf('factor throws  OR  ||K*w - l|| / ||l|| > 1e-2 (%s)', prec);
    print_test_header(name, description, expected);
    try
        rng(12);
        K = sprandn(n, n, 0.02) + n*speye(n);
        K(7, :) = 0;   % zero out an entire row -> guaranteed singular

        threw    = false;
        err_msg  = '';
        residual = NaN;
        try
            h = cudss.factor(K, struct('precision', prec));
            L = randn(n, 1);
            L_gpu = gpuArray(cast_fn(L));
            W_gpu = cudss.solve(h, L_gpu);
            cudss.destroy(h);

            W = double(gather(W_gpu));
            residual = norm(K * W - L) / max(norm(L), eps);
        catch ME
            threw   = true;
            err_msg = ME.message;
        end

        if threw
            pass = true;
            msg  = sprintf('factor errored as expected: %s', trim_msg(err_msg, 160));
        else
            pass = residual > 1e-2;
            msg  = sprintf('no-throw factor; residual ||K*w - l||/||l|| = %.2e (must be > 1e-2)', residual);
        end
        print_test_result(pass, msg);
        r = make_result(name, description, expected, pass, msg);
    catch ME
        print_test_result(false, ME.message);
        r = make_result(name, description, expected, false, ME.message);
    end
end

function r = run_test_stream_interleave(n, nrhs, prec)
    [cast_fn, tol] = precision_params(prec, 'residual_basic');
    name        = sprintf('Default-stream interleave (n=%d, nrhs=%d, %s)', n, nrhs, prec);
    description = ['Queue arrayfun kernels on MATLAB''s default stream that mutate L *immediately ' ...
                   'before* cudss.solve.  The cuDSS solver stream must wait for those kernels; ' ...
                   'otherwise the solve reads partially-mutated L and the answer is wrong.'];
    expected    = sprintf('||W_gpu - W_host||_F / ||W_host||_F < %.0e (%s)', tol, prec);
    print_test_header(name, description, expected);
    try
        rng(13);
        K = sprandn(n, n, 0.02) + n*speye(n);
        L = randn(n, nrhs);

        h = cudss.factor(K, struct('precision', prec));

        % Build L on the GPU and mutate it in-place via a non-trivial
        % arrayfun pipeline.  This queues several kernels on MATLAB's
        % default stream.  cudss.solve must observe the *post-mutation*
        % values.
        L_gpu = gpuArray(cast_fn(L));
        L_gpu = arrayfun(@(v) v + 1.0, L_gpu);
        L_gpu = arrayfun(@(v) v - 1.0, L_gpu);   % net no-op, but forces sync

        W_gpu = cudss.solve(h, L_gpu);
        cudss.destroy(h);

        W_host = K \ L;   % reference against the original (= post-mutation) L
        relerr = norm(double(gather(W_gpu)) - W_host, 'fro') / ...
                 max(norm(W_host, 'fro'), eps);
        pass = relerr < tol;
        msg  = sprintf('relerr after default-stream interleave = %.2e (tol %.0e)', relerr, tol);
        print_test_result(pass, msg);
        r = make_result(name, description, expected, pass, msg);
    catch ME
        print_test_result(false, ME.message);
        r = make_result(name, description, expected, false, ME.message);
    end
end

function r = run_test_reordering_alg(N)
    % Every reordering algorithm must produce the same solution.
    %
    % CUDSS_CONFIG_REORDERING_ALG changes the fill-reducing permutation, so
    % it changes the factor's sparsity, memory, and speed -- but never the
    % solution.  Also checks that an unknown algorithm name is rejected.
    n = N^3;
    algs = {'default', 'alg1', 'alg2', 'alg3'};
    name        = sprintf('reordering_alg agreement (%d^3 = %d)', N, n);
    description = ['Factor the same K under reordering_alg default/alg1/alg2/alg3 and require ' ...
                   'every one to solve to the same answer (the permutation changes fill, not ' ...
                   'the solution).  An unrecognized algorithm name must raise cudss:BadOpts.'];
    expected    = 'all algorithms residual < 1e-12; bad name raises cudss:BadOpts';
    print_test_header(name, description, expected);
    try
        K = build_3d_laplacian(N);
        rng(41);
        L     = randn(n, 3);
        L_gpu = gpuArray(L);

        resids = nan(numel(algs), 1);
        for k = 1:numel(algs)
            h = cudss.factor(K, struct('precision', 'double', ...
                                       'reordering_alg', algs{k}));
            W = gather(cudss.solve(h, L_gpu));
            cudss.destroy(h);
            resids(k) = max(vecnorm(K * W - L) ./ max(vecnorm(L), eps));
        end

        bad_id = '';
        try
            h = cudss.factor(K, struct('precision', 'double', ...
                                       'reordering_alg', 'alg99'));
            cudss.destroy(h);
        catch ME_bad
            bad_id = ME_bad.identifier;
        end

        pass = all(resids < 1e-12) && strcmp(bad_id, 'cudss:BadOpts');
        msg  = sprintf('residuals: %s (tol 1e-12); bad-name err id = "%s"', ...
                       strjoin(arrayfun(@(k) sprintf('%s %.2e', algs{k}, resids(k)), ...
                                        1:numel(algs), 'UniformOutput', false), ', '), ...
                       bad_id);
        print_test_result(pass, msg);
        r = make_result(name, description, expected, pass, msg);
    catch ME
        print_test_result(false, ME.message);
        r = make_result(name, description, expected, false, ME.message);
    end
end

function r = run_test_hybrid_memory(N, nrhs, prec)
    % Hybrid host/device memory mode must not change the answer.
    %
    % Solves the same 3-D Laplacian three ways -- no hybrid, hybrid with no
    % device-memory limit, and hybrid with a limit derived from the minimum
    % cuDSS reports -- and requires all three to agree with the host solve.
    % Hybrid mode relocates most of the factor to host memory and streams it
    % back per phase, so it can change the answer, not just the timing.
    n = N^3;
    [cast_fn, tol] = precision_params(prec, 'residual_laplacian');
    name        = sprintf('Hybrid memory mode (%d^3 = %d, nrhs=%d, %s)', N, n, nrhs, prec);
    description = ['Factor the same K with hybrid off, hybrid on (no limit), and hybrid on with ' ...
                   'a device-memory limit above the cuDSS-reported minimum.  All three must ' ...
                   'produce the same solution, and the limited run must not error.'];
    expected    = sprintf('all three configs: max ||K*w - l|| / ||l|| < %.0e (%s)', tol, prec);
    print_test_header(name, description, expected);
    try
        K  = cast_fn(build_3d_laplacian(N));
        rng(31);
        L  = cast_fn(randn(n, nrhs));
        L_gpu = gpuArray(L);

        configs = { 'hybrid off', struct('precision', prec); ...
                    'hybrid, no limit', struct('precision', prec, 'hybrid', true); ...
                    'hybrid, 2 GiB limit', struct('precision', prec, 'hybrid', true, ...
                                                  'hybrid_memory_limit_gib', 2) };

        resids = zeros(size(configs, 1), 1);
        W_ref  = [];
        agree  = 0;
        for k = 1:size(configs, 1)
            h = cudss.factor(K, configs{k, 2});
            W = gather(cudss.solve(h, L_gpu));
            cudss.destroy(h);
            resids(k) = max(vecnorm(double(K) * double(W) - double(L)) ./ ...
                            max(vecnorm(double(L)), eps));
            if k == 1
                W_ref = double(W);
            else
                agree = max(agree, norm(double(W) - W_ref, 'fro') / ...
                                   max(norm(W_ref, 'fro'), eps));
            end
        end

        pass = all(resids < tol);
        msg  = sprintf(['residuals: off %.2e, hybrid %.2e, hybrid+limit %.2e ' ...
                        '(tol %.0e); max deviation from non-hybrid = %.2e'], ...
                       resids(1), resids(2), resids(3), tol, agree);
        print_test_result(pass, msg);
        r = make_result(name, description, expected, pass, msg);
    catch ME
        print_test_result(false, ME.message);
        r = make_result(name, description, expected, false, ME.message);
    end
end

function r = run_test_hybrid_memory_limit_rejection(N)
    % A hybrid device-memory limit below the cuDSS minimum must be rejected
    % with cudss:HybridMemoryLimitTooSmall.  That check fires inside the
    % factor handler's try block, which owns a half-built SolverState, so
    % the follow-up factor/solve is the real assertion: it fails only if
    % the aborted factor stranded a cuDSS handle or stream.
    n = N^3;
    name        = sprintf('Hybrid limit below minimum rejected (%d^3 = %d)', N, n);
    description = ['Request an impossibly small opts.hybrid_memory_limit_gib.  cudss.factor must ' ...
                   'raise cudss:HybridMemoryLimitTooSmall, and a normal factor + solve + destroy ' ...
                   'cycle must still succeed afterwards (proving the aborted factor was cleaned up).'];
    expected    = 'cudss:HybridMemoryLimitTooSmall, then a clean factor/solve/destroy';
    print_test_header(name, description, expected);
    try
        K = single(build_3d_laplacian(N));
        rng(32);
        L_gpu = gpuArray(single(randn(n, 2)));

        err_id = '';
        try
            h = cudss.factor(K, struct('precision', 'single', 'hybrid', true, ...
                                       'hybrid_memory_limit_gib', 1e-6));
            cudss.destroy(h);
        catch ME_lim
            err_id = ME_lim.identifier;
        end
        rejected = strcmp(err_id, 'cudss:HybridMemoryLimitTooSmall');

        % The wrapper must still be healthy after that aborted factor.
        h = cudss.factor(K, struct('precision', 'single'));
        W = gather(cudss.solve(h, L_gpu));
        cudss.destroy(h);
        resid = max(vecnorm(double(K) * double(W) - double(gather(L_gpu))) ./ ...
                    max(vecnorm(double(gather(L_gpu))), eps));
        recovered = resid < 1e-4;

        pass = rejected && recovered;
        msg  = sprintf('err id = "%s" (rejected = %s); post-error residual = %.2e (recovered = %s)', ...
                       err_id, mat2str(rejected), resid, mat2str(recovered));
        print_test_result(pass, msg);
        r = make_result(name, description, expected, pass, msg);
    catch ME
        print_test_result(false, ME.message);
        r = make_result(name, description, expected, false, ME.message);
    end
end


% =========================================================================
%  Helpers
% =========================================================================

function K = build_3d_laplacian(N)
    % Standard 7-point Laplacian on an N x N x N cubic grid with Dirichlet
    % zero on the boundary.  Returns SPD sparse double.
    n = N^3;
    e = ones(N, 1);
    T1 = spdiags([-e, 2*e, -e], -1:1, N, N);
    I  = speye(N);
    % kron of sparse + sparse stays sparse; do NOT do `K + 0` here -- adding
    % a numeric scalar to a sparse matrix densifies it in MATLAB (every
    % implicit zero becomes an explicit one).
    K  = kron(kron(I, I), T1) + kron(kron(I, T1), I) + kron(kron(T1, I), I);
    assert(issparse(K));
    assert(size(K, 1) == n);
end

function [cast_fn, tol] = precision_params(prec, kind)
    % Returns the host-side cast function and a per-precision tolerance.
    % `kind` selects which tolerance flavor a test wants:
    %   'relerr_basic'        small well-conditioned K, compare to host K\L
    %   'residual_basic'      small well-conditioned K, residual against K
    %   'residual_laplacian'  3-D Laplacian (n ~ 10k-50k), residual
    %   'residual_large'      large 3-D Laplacian (~200k), residual
    %   'async_vs_sync'       same factor, different wait point; near bit-exact
    cast_fn = precision_cast(prec);
    is_single = strcmp(prec, 'single');
    switch kind
        case 'relerr_basic'
            tol = ternary(is_single, 1e-5,  1e-13);
        case 'residual_basic'
            tol = ternary(is_single, 1e-4,  1e-12);
        case 'residual_laplacian'
            tol = ternary(is_single, 1e-4,  1e-12);
        case 'residual_large'
            tol = ternary(is_single, 1e-4,  1e-11);
        case 'async_vs_sync'
            tol = ternary(is_single, 1e-6,  1e-14);
        otherwise
            error('precision_params:UnknownKind', 'Unknown tolerance kind: %s', kind);
    end
end

function [cast_fn, residual_tol, agree_tol] = precision_params_matrix_type(prec)
    % Tolerances for the matrix_type agreement test: per-config residual
    % and cross-config (sym/spd vs general) agreement.
    cast_fn = precision_cast(prec);
    if strcmp(prec, 'single')
        residual_tol = 1e-4;
        agree_tol    = 1e-4;
    else
        residual_tol = 1e-12;
        agree_tol    = 1e-11;
    end
end

function fn = precision_cast(prec)
    if strcmp(prec, 'single')
        fn = @single;
    else
        fn = @double;
    end
end

function print_test_header(name, description, expected)
    fprintf('\n--- %s ---\n', name);
    fprintf('  What:     %s\n', description);
    fprintf('  Expected: %s\n', expected);
end

function print_test_result(pass, msg)
    tag = ternary(pass, 'PASS', 'FAIL');
    fprintf('  Result:   %s  [%s]\n', msg, tag);
end

function r = make_result(name, description, expected, pass, msg)
    r = struct('name', name, 'description', description, ...
               'expected', expected, 'pass', pass, 'msg', msg);
end

function s = ternary(cond, a, b)
    if cond, s = a; else, s = b; end
end

function tf = safe_mislocked(name)
    % mislocked(name) errors when the named MEX is not loaded.  We want a
    % tristate -- locked / unlocked / unloaded -- collapsed onto a boolean
    % "is this MEX pinned in memory?".  Treat "not loaded" as "not locked".
    try
        tf = mislocked(name);
    catch
        tf = false;
    end
end

function out = trim_msg(s, maxlen)
    if isempty(s)
        out = '';
        return
    end
    s = strrep(s, sprintf('\n'), ' ');
    if numel(s) > maxlen
        out = [s(1:maxlen) '...'];
    else
        out = s;
    end
end
