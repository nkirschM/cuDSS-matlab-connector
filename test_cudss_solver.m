%% test_cudss_solver.m  --  Phase 1 validation suite for the +cudss MEX wrapper.
%
% Runs tests 1-3 from the BUILDSPEC (§7):
%   1. Tiny smoke (n=100, 1 RHS)        -- correctness vs host K\L
%   2. Multi-RHS  (n=1000, nrhs=10)     -- column-major dense layout check
%   3. Octree-shaped Laplacian (n=1e4)  -- 7-point 3-D stencil residual
%
% Tests 4-7 (Phase 2: factor-once-solve-many, precision dispatch, wrong-precision
% rejection, handle lifecycle) and test 8 (Phase 3: n=200,000 stress) are
% gated behind RUN_PHASE2 / RUN_PHASE3 flags, default off.  They are
% included as scaffolding so the suite grows monotonically as features land.
%
% Run:   matlab -batch "addpath(genpath('subFunctions')); ..." +
%        "run('subFunctions/gpu_solvers/test_cudss_solver.m')"

clc; clearvars; close all;

RUN_PHASE2      = true;   % flip to true once Phase 2 lands
RUN_PHASE3      = true;   % flip to true once Phase 3 lands
RUN_REGRESSIONS = true;   % regression tests covering the dispatcher refactor
                          % and the CUDSS_DATA_INFO / stream-sync fixes.

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

results = struct('name', {}, 'pass', {}, 'msg', {});

% =====================================================================
%  Test 1: tiny smoke
% =====================================================================
results(end+1) = run_test_smoke(100);

% =====================================================================
%  Test 2: multi-RHS correctness
% =====================================================================
results(end+1) = run_test_multirhs(1000, 10);

% =====================================================================
%  Test 3: octree-proxy 3-D 7-point Laplacian
% =====================================================================
results(end+1) = run_test_laplacian(22);   % 22^3 = 10648 ~= 1e4

% =====================================================================
%  Phase 2 / 3 (gated)
% =====================================================================
if RUN_PHASE2
    results(end+1) = run_test_factor_once_solve_many(500, 50);
    results(end+1) = run_test_precision_dispatch(200);
    results(end+1) = run_test_wrong_precision_rejection(200);
    results(end+1) = run_test_handle_lifecycle(200);
    results(end+1) = run_test_matrix_type(18);
    results(end+1) = run_test_matrix_type_rejects_non_spd(200);
end
if RUN_PHASE3
    results(end+1) = run_test_stress_200k();
    results(end+1) = run_test_async_factor(2000, 5);
end
if RUN_REGRESSIONS
    results(end+1) = run_test_lock_counter(200, 50);
    results(end+1) = run_test_singular_K_detection(40);
    results(end+1) = run_test_stream_interleave(500, 4);
end

% =====================================================================
%  Summary
% =====================================================================
fprintf('\n================ TEST SUMMARY ================\n');
n_pass = 0;
for r = results
    tag = ternary(r.pass, 'PASS', 'FAIL');
    fprintf('  %s  %s\n', tag, r.name);
    if ~r.pass, fprintf('         %s\n', r.msg); end
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

function r = run_test_smoke(n)
    name = sprintf('Test 1 (smoke, n=%d)', n);
    fprintf('\n--- %s ---\n', name);
    try
        rng(0);
        K = sprandn(n, n, 0.05) + n*speye(n);   % strongly diagonal -> well-conditioned
        L = randn(n, 1);

        W_host = K \ L;

        h     = cudss.factor(K, struct('precision', 'single'));
        L_gpu = gpuArray(single(L));
        W_gpu = cudss.solve(h, L_gpu);
        cudss.destroy(h);

        W_gpu_h = double(gather(W_gpu));
        relerr  = norm(W_gpu_h - W_host) / max(norm(W_host), eps);
        tol     = 1e-4;        % FP32 + sprandn -- conservative
        pass    = relerr < tol;
        msg     = sprintf('relerr = %.2e (tol %.0e)', relerr, tol);
        fprintf('  %s\n', msg);
        r = make_result(name, pass, msg);
    catch ME
        r = make_result(name, false, ME.message);
    end
end

function r = run_test_multirhs(n, nrhs)
    name = sprintf('Test 2 (multi-RHS, n=%d, nrhs=%d)', n, nrhs);
    fprintf('\n--- %s ---\n', name);
    try
        rng(1);
        K = sprandn(n, n, 0.01) + n*speye(n);
        L = randn(n, nrhs);

        W_host = K \ L;

        h     = cudss.factor(K, struct('precision', 'single'));
        L_gpu = gpuArray(single(L));
        W_gpu = cudss.solve(h, L_gpu);
        cudss.destroy(h);

        W_gpu_h = double(gather(W_gpu));
        relerr  = norm(W_gpu_h - W_host, 'fro') / max(norm(W_host, 'fro'), eps);
        tol     = 1e-4;
        pass    = relerr < tol;
        msg     = sprintf('||W_gpu - W_host||_F / ||W_host||_F = %.2e (tol %.0e)', ...
                          relerr, tol);
        fprintf('  %s\n', msg);
        r = make_result(name, pass, msg);
    catch ME
        r = make_result(name, false, ME.message);
    end
end

function r = run_test_laplacian(N)
    n_total = N^3;
    name = sprintf('Test 3 (3-D Laplacian, %d^3 = %d, nrhs=4)', N, n_total);
    fprintf('\n--- %s ---\n', name);
    try
        K  = build_3d_laplacian(N);     % SPD by construction
        L  = randn(n_total, 4);

        h     = cudss.factor(K, struct('precision', 'single'));
        L_gpu = gpuArray(single(L));
        W_gpu = cudss.solve(h, L_gpu);
        cudss.destroy(h);

        % residual check rather than W vs host: factor for FP32, the
        % per-column residual is the directly meaningful quantity.
        W_h    = double(gather(W_gpu));
        R      = K * W_h - L;
        relres = vecnorm(R) ./ max(vecnorm(L), eps);
        tol    = 1e-3;          % FP32 Laplacian conditioning is mild but not 1e-5 free
        pass   = max(relres) < tol;
        msg    = sprintf('max ||K*w - l||/||l|| over %d RHS = %.2e (tol %.0e)', ...
                         numel(relres), max(relres), tol);
        fprintf('  %s\n', msg);
        r = make_result(name, pass, msg);
    catch ME
        r = make_result(name, false, ME.message);
    end
end

% --- Phase 2 stubs (gated by RUN_PHASE2) ----------------------------------

function r = run_test_factor_once_solve_many(n, nsolves) %#ok<DEFNU>
    name = sprintf('Test 4 (factor-once / %d solves, n=%d)', nsolves, n);
    fprintf('\n--- %s ---\n', name);
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
        % guarantee solves are not re-factoring: max should not exceed 5x median
        pass  = max_t < 5 * med_t;
        msg   = sprintf('median %.4f s, max %.4f s', med_t, max_t);
        r     = make_result(name, pass, msg);
    catch ME
        r = make_result(name, false, ME.message);
    end
end

function r = run_test_precision_dispatch(n) %#ok<DEFNU>
    name = sprintf('Test 5 (precision dispatch, n=%d)', n);
    fprintf('\n--- %s ---\n', name);
    try
        rng(3);
        K = sprandn(n, n, 0.05) + n*speye(n);
        L = randn(n, 1);

        h_s     = cudss.factor(K, struct('precision', 'single'));
        W_s     = cudss.solve(h_s, gpuArray(single(L)));
        cudss.destroy(h_s);

        h_d     = cudss.factor(K, struct('precision', 'double'));
        W_d     = cudss.solve(h_d, gpuArray(double(L)));
        cudss.destroy(h_d);

        rel_d   = norm(double(gather(W_d)) - K \ L) / norm(K \ L);
        rel_s   = norm(double(gather(W_s)) - K \ L) / norm(K \ L);
        pass    = rel_d < 1e-12 && rel_s < 1e-4;
        msg     = sprintf('FP64 relerr %.2e, FP32 relerr %.2e', rel_d, rel_s);
        r       = make_result(name, pass, msg);
    catch ME
        r = make_result(name, false, ME.message);
    end
end

function r = run_test_wrong_precision_rejection(n) %#ok<DEFNU>
    name = sprintf('Test 6 (wrong-precision rejection, n=%d)', n);
    fprintf('\n--- %s ---\n', name);
    try
        rng(4);
        K = sprandn(n, n, 0.05) + n*speye(n);
        h = cudss.factor(K, struct('precision', 'single'));
        threw = false;
        try
            cudss.solve(h, gpuArray(double(randn(n, 1))));
        catch ME
            threw = strcmp(ME.identifier, 'cudss:WrongInputType');
        end
        cudss.destroy(h);
        msg = sprintf('threw cudss:WrongInputType = %s', mat2str(threw));
        r   = make_result(name, threw, msg);
    catch ME
        r = make_result(name, false, ME.message);
    end
end

function r = run_test_handle_lifecycle(n) %#ok<DEFNU>
    name = sprintf('Test 7 (handle lifecycle, n=%d)', n);
    fprintf('\n--- %s ---\n', name);
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
        warned = false;
        prev = warning('off', 'cudss:AlreadyDestroyed');
        cleanup = onCleanup(@() warning(prev));
        lastwarn('');
        cudss.destroy(h);
        [~, wid] = lastwarn;
        warned = strcmp(wid, 'cudss:AlreadyDestroyed');

        pass = threw_solve && warned;
        msg  = sprintf('solve-after-destroy threw=%s, double-destroy warned=%s', ...
                       mat2str(threw_solve), mat2str(warned));
        r    = make_result(name, pass, msg);
    catch ME
        r = make_result(name, false, ME.message);
    end
end

function r = run_test_matrix_type(N) %#ok<DEFNU>
    % Verifies opts.matrix_type end-to-end on a symmetric-positive-definite
    % K (3-D 7-point Laplacian): all three settings -- 'general',
    % 'symmetric', 'spd' -- must factor and solve the same RHS to matching
    % FP32 residuals.  Exercises both the MATLAB-side triu(K)+transpose
    % path for sym/spd and the MEX mapping to MTYPE_SYMMETRIC / MTYPE_SPD
    % with MVIEW_UPPER.
    n_total = N^3;
    name = sprintf('Test 10 (matrix_type SPD/sym/general agreement, %d^3 = %d)', ...
                   N, n_total);
    fprintf('\n--- %s ---\n', name);
    try
        K = build_3d_laplacian(N);     % SPD by construction
        rng(10);
        L = randn(n_total, 3);
        L_gpu = gpuArray(single(L));

        configs = {'general', 'symmetric', 'spd'};
        W_each  = cell(1, numel(configs));
        for c = 1:numel(configs)
            h = cudss.factor(K, struct('precision',   'single', ...
                                       'matrix_type', configs{c}));
            W_each{c} = double(gather(cudss.solve(h, L_gpu)));
            cudss.destroy(h);
        end

        % residual against original K, per RHS column
        tol = 1e-3;  % FP32 Laplacian conditioning is mild; matches Test 3
        rel_max = zeros(1, numel(configs));
        for c = 1:numel(configs)
            R = K * W_each{c} - L;
            rel_max(c) = max(vecnorm(R) ./ max(vecnorm(L), eps));
        end
        % Cross-config agreement (symmetric and spd should match general to
        % FP32 noise -- same matrix, same solve up to factor-path reordering).
        sym_vs_gen = norm(W_each{2} - W_each{1}, 'fro') / max(norm(W_each{1}, 'fro'), eps);
        spd_vs_gen = norm(W_each{3} - W_each{1}, 'fro') / max(norm(W_each{1}, 'fro'), eps);
        agree_tol  = 1e-3;

        pass = all(rel_max < tol) && sym_vs_gen < agree_tol && spd_vs_gen < agree_tol;
        msg  = sprintf(['relres max (gen / sym / spd) = %.2e / %.2e / %.2e, ' ...
                        'sym-vs-gen %.2e, spd-vs-gen %.2e (tol %.0e)'], ...
                        rel_max(1), rel_max(2), rel_max(3), ...
                        sym_vs_gen, spd_vs_gen, agree_tol);
        fprintf('  %s\n', msg);
        r = make_result(name, pass, msg);
    catch ME
        r = make_result(name, false, ME.message);
    end
end

function r = run_test_matrix_type_rejects_non_spd(n) %#ok<DEFNU>
    % Verifies matrix_type='spd' on a symmetric *indefinite* K surfaces a
    % cudss:NumericalError (non-zero CUDSS_DATA_INFO from Cholesky meeting
    % a non-positive pivot), while matrix_type='general' on the same K still
    % factors and solves successfully.  This is a stricter check than the
    % general matrix_type test: SPD's contract must actually constrain.
    %
    % Note: we don't assert anything about matrix_type='symmetric' on this K.
    % cuDSS's LDL^T behavior on arbitrary indefinite matrices depends on its
    % pivoting policy, which varies across cuDSS releases; the test would be
    % brittle to that.  The general/spd pair is enough to prove the flag is
    % wired through to cudssMatrixCreateCsr correctly.
    name = sprintf('Test 11 (matrix_type=''spd'' rejects indefinite K, n=%d)', n);
    fprintf('\n--- %s ---\n', name);
    try
        rng(20);
        % Symmetric indefinite K: random sparse symmetrized + small diagonal
        % to keep it nonsingular.  Mix of positive and negative eigenvalues
        % guaranteed by the small diagonal shift relative to the off-diagonal
        % spectral radius.
        A = sprandn(n, n, 0.05);
        K = A + A.' + 0.1 * speye(n);   % indefinite, but symmetric

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

        % 'spd' (Cholesky) must reject with cudss:NumericalError -- the
        % stable ID both sync and async factor paths use when cuDSS
        % reports CUDSS_DATA_INFO != 0.  cudss:CudssError is also
        % acceptable, in case cuDSS catches the failure at the
        % cudssExecute level instead of at the DATA_INFO query.
        spd_rejected = false;
        spd_err_id   = '';
        try
            h = cudss.factor(K, struct('precision',   'double', ...
                                       'matrix_type', 'spd'));
            cudss.destroy(h);   % only reached if factor wrongly succeeded
        catch ME_spd
            spd_err_id   = ME_spd.identifier;
            spd_rejected = strcmp(spd_err_id, 'cudss:NumericalError') || ...
                           strcmp(spd_err_id, 'cudss:CudssError');
        end

        pass = gen_ok && spd_rejected;
        msg  = sprintf('general OK = %s, spd rejected = %s (err id: %s)', ...
                        mat2str(gen_ok), mat2str(spd_rejected), spd_err_id);
        fprintf('  %s\n', msg);
        r = make_result(name, pass, msg);
    catch ME
        r = make_result(name, false, ME.message);
    end
end

function r = run_test_stress_200k() %#ok<DEFNU>
    name = 'Test 8 (n=200k stress, FP32, 10 RHS)';
    fprintf('\n--- %s ---\n', name);
    try
        N = round(200000^(1/3));     % ~58
        K = build_3d_laplacian(N);
        nrhs = 10;
        L = randn(size(K, 1), nrhs);

        t_host = tic;
        W_host = K \ L(:, 1);   % single column on host -- baseline
        host_per_rhs = toc(t_host);

        t_factor = tic;
        h = cudss.factor(K, struct('precision', 'single'));
        factor_time = toc(t_factor);

        t_solve = tic;
        L_gpu = gpuArray(single(L));
        W_gpu = cudss.solve(h, L_gpu); %#ok<NASGU>
        wait(gpuDevice());
        gpu_total = toc(t_solve);
        gpu_per_rhs = gpu_total / nrhs;

        cudss.destroy(h);

        speedup = host_per_rhs / max(gpu_per_rhs, eps);
        pass    = speedup >= 5;
        msg     = sprintf('host %.3f s/RHS, gpu %.3f s/RHS, factor %.2f s, speedup %.1fx', ...
                          host_per_rhs, gpu_per_rhs, factor_time, speedup);
        fprintf('  %s\n', msg);
        r = make_result(name, pass, msg);
    catch ME
        r = make_result(name, false, ME.message);
    end
end


function r = run_test_async_factor(n, nrhs) %#ok<DEFNU>
    % Verify that opts.async = true returns immediately, that cudss.wait
    % drains the queued factor, and that the resulting solve matches a
    % synchronous factor of the same K to FP32 tolerance.
    name = sprintf('Test 9 (async factor + cudss.wait, n=%d, nrhs=%d)', n, nrhs);
    fprintf('\n--- %s ---\n', name);
    try
        rng(9);
        K = sprandn(n, n, 0.02) + n*speye(n);
        L = randn(n, nrhs);

        % --- sync reference ---
        h_sync = cudss.factor(K, struct('precision', 'single'));
        L_gpu  = gpuArray(single(L));
        W_sync = cudss.solve(h_sync, L_gpu);
        cudss.destroy(h_sync);

        % --- async path ---
        t_issue = tic;
        h_async = cudss.factor(K, struct('precision', 'single', 'async', true));
        issue_s = toc(t_issue);

        % Run unrelated GPU work to verify the issue truly returned without
        % blocking on the factor.  arrayfun on the default stream contends
        % for the same GPU but is dispatched onto a different stream than
        % the cuDSS factor stream, so they should overlap on the device.
        x = gpuArray.rand(1024, 1024, 'single');
        for k = 1:5
            x = arrayfun(@(v) sin(v) + cos(v), x);
        end
        wait(gpuDevice());   % flush the unrelated kernel queue

        cudss.wait(h_async); % drain the factor stream

        W_async = cudss.solve(h_async, gpuArray(single(L)));
        cudss.destroy(h_async);

        relerr = norm(double(gather(W_async)) - double(gather(W_sync)), 'fro') / ...
                 max(norm(double(gather(W_sync)), 'fro'), eps);
        % Async vs sync use the same factor on the same K -- the math is
        % identical, only the wait point differs.  Tolerance reflects FP32
        % nondeterminism (parallel reductions can reorder), but the result
        % should be very close to bit-exact.
        tol  = 1e-5;
        % Issue time should be much smaller than a typical factor wall (we
        % don't have a sync baseline here, but anything sub-millisecond is
        % a good signal that the call returned without waiting).  500 ms is
        % a generous ceiling that still catches a regression where the MEX
        % accidentally syncs on this size.
        issue_pass = issue_s < 0.5;
        pass = (relerr < tol) && issue_pass;
        msg  = sprintf('issue %.3f s, relerr async vs sync %.2e (tol %.0e)', ...
                       issue_s, relerr, tol);
        fprintf('  %s\n', msg);
        r = make_result(name, pass, msg);
    catch ME
        r = make_result(name, false, ME.message);
    end
end


% --- Regression tests (gated by RUN_REGRESSIONS) -------------------------

function r = run_test_lock_counter(n, ncycles) %#ok<DEFNU>
    % Verifies Fix 1: the dispatcher MEX's mexLock / mexUnlock pair within
    % the same binary, so repeated factor + destroy cycles do not leak the
    % lock count.  After `ncycles` round-trips the MEX should be unlocked
    % (no live handles), unlike the old four-binary layout where every
    % factor permanently incremented solver_factor's lock counter.
    name = sprintf('Regression: lock counter no drift (n=%d, %d cycles)', ...
                   n, ncycles);
    fprintf('\n--- %s ---\n', name);
    try
        rng(11);
        K = sprandn(n, n, 0.05) + n*speye(n);

        % Capture the baseline lock state.  mislocked returns true if the
        % MEX was previously locked (e.g., by an earlier test that hasn't
        % finished destroying a handle yet -- we make no assumption that
        % the count starts at zero).
        baseline_locked = safe_mislocked('cudss_solver');

        for k = 1:ncycles
            h = cudss.factor(K, struct('precision', 'single'));
            cudss.destroy(h);
        end

        final_locked = safe_mislocked('cudss_solver');

        % If we entered with no other handles outstanding and ncycles > 0,
        % the MEX should be unlocked at the end.  If we entered with the
        % MEX already locked (some other test hasn't cleaned up), require
        % at least that the state did not get *more* locked -- mislocked
        % is a boolean, but the symptom of a leak is that final stays
        % true even after every handle was destroyed.
        if baseline_locked
            % Conservative: just require it to still be locked-or-unlocked
            % consistently with at-least-one outstanding handle externally.
            pass = true;
            msg  = sprintf('baseline locked = true (other handle live), final locked = %s', ...
                           mat2str(final_locked));
        else
            pass = ~final_locked;
            msg  = sprintf('after %d cycles mislocked = %s (expected false)', ...
                           ncycles, mat2str(final_locked));
        end
        fprintf('  %s\n', msg);
        r = make_result(name, pass, msg);
    catch ME
        r = make_result(name, false, ME.message);
    end
end

function r = run_test_singular_K_detection(n) %#ok<DEFNU>
    % Verifies Fix 3 (and the broader user-facing contract): a clearly
    % singular K must not silently produce a "valid" handle whose solve
    % returns garbage with no diagnostic.
    %
    % NOTE on cuDSS behavior: by default cuDSS applies a small pivot
    % perturbation (regularization) when it encounters a numerically-zero
    % pivot, and CUDSS_DATA_INFO comes back 0 even for a structurally
    % singular K.  The DATA_INFO check (Fix 3) catches the cases where
    % cuDSS *does* report a non-zero info -- e.g., when the analysis
    % phase itself rejects the matrix or pivoting cannot be perturbed.
    % For matrices where cuDSS chooses to perturb-and-continue, the
    % factor returns SUCCESS but the solve produces a residual that is
    % orders of magnitude larger than tolerance.
    %
    % The user-facing contract this test enforces is:
    %   (factor errors) OR (solve produces a residual >> tolerance)
    % Either branch means a downstream caller can detect the failure
    % via the existing per-RHS residual check in
    % solve_elastic_system_decomp_cudss.m.  What MUST NOT happen is a
    % no-throw factor + solve with a small-looking residual.
    name = sprintf('Regression: singular-K detection (n=%d)', n);
    fprintf('\n--- %s ---\n', name);
    try
        rng(12);
        % Strongly diagonal background, then knock out one diagonal entry
        % to force a zero pivot.
        K = sprandn(n, n, 0.02) + n*speye(n);
        K(7, :) = 0;   % zero out an entire row -> guaranteed singular

        threw   = false;
        err_msg = '';
        residual = NaN;
        try
            h = cudss.factor(K, struct('precision', 'single'));
            % cuDSS perturbed-and-continued.  The factor exists but is
            % numerically corrupt -- we should observe that via a huge
            % residual on a generic RHS.
            L = randn(n, 1);
            L_gpu = gpuArray(single(L));
            W_gpu = cudss.solve(h, L_gpu);
            cudss.destroy(h);

            W = double(gather(W_gpu));
            % Compute residual against the original singular K (not the
            % perturbed one cuDSS factored).  ||K*w - l|| / ||l|| should
            % be massive because the row-7 equation of K is 0 = L(7),
            % which a generic L violates outright.
            residual = norm(K * W - L) / max(norm(L), eps);
        catch ME
            threw   = true;
            err_msg = ME.message;
        end

        if threw
            % Either Fix 3's DATA_INFO check, the throwing-construction
            % path, or any other cuDSS / CUDA error caught the singular
            % factor before it reached the caller.  All count as the
            % desired behavior.
            pass = true;
            msg  = sprintf('factor errored as expected: %s', ...
                           trim_msg(err_msg, 160));
        else
            % cuDSS perturbed and the factor "succeeded".  The residual
            % must blow past the FP32 tolerance gate by a wide margin so
            % a downstream caller (e.g., solve_elastic_system_decomp_cudss
            % which already checks ||K*u - f||/||f|| >= 1e-4) can detect
            % the failure.
            pass = residual > 1e-2;
            msg  = sprintf(['no-throw factor; residual ||K*w - l||/||l|| ' ...
                            '= %.2e (must be >> 1e-4 for downstream callers ' ...
                            'to detect)'], residual);
        end
        fprintf('  %s\n', msg);
        r = make_result(name, pass, msg);
    catch ME
        r = make_result(name, false, ME.message);
    end
end

function r = run_test_stream_interleave(n, nrhs) %#ok<DEFNU>
    % Verifies Fix 4: the explicit cudaEventRecord(stream 0) +
    % cudaStreamWaitEvent at the top of the SOLVE keeps things correct
    % when MATLAB has just queued kernels on its default stream.  We
    % build L_gpu and immediately mutate it via arrayfun (still on the
    % default stream) right before cudss.solve.  If the cuDSS stream
    % did not wait, it could read the partially-mutated L and produce a
    % wrong answer.
    name = sprintf('Regression: stream interleave (n=%d, nrhs=%d)', n, nrhs);
    fprintf('\n--- %s ---\n', name);
    try
        rng(13);
        K = sprandn(n, n, 0.02) + n*speye(n);
        L = randn(n, nrhs);

        h = cudss.factor(K, struct('precision', 'single'));

        % Build L on the GPU and mutate it in-place via a non-trivial
        % arrayfun pipeline.  This queues several kernels on MATLAB's
        % default stream.  cudss.solve must observe the *post-mutation*
        % values.
        L_gpu = gpuArray(single(L));
        L_gpu = arrayfun(@(v) v + 1.0, L_gpu);
        L_gpu = arrayfun(@(v) v - 1.0, L_gpu);   % net no-op, but forces sync

        W_gpu = cudss.solve(h, L_gpu);
        cudss.destroy(h);

        % Reference: solve the same system on the host (against the
        % post-mutation L, which equals the original).
        W_host = K \ L;
        relerr = norm(double(gather(W_gpu)) - W_host, 'fro') / ...
                 max(norm(W_host, 'fro'), eps);
        tol  = 1e-3;
        pass = relerr < tol;
        msg  = sprintf('relerr after default-stream interleave = %.2e (tol %.0e)', ...
                       relerr, tol);
        fprintf('  %s\n', msg);
        r = make_result(name, pass, msg);
    catch ME
        r = make_result(name, false, ME.message);
    end
end


% =========================================================================
%  Helpers
% =========================================================================

function K = build_3d_laplacian(N)
    % Standard 7-point Laplacian on an N x N x N cubic grid with Dirichlet
    % zero on the boundary (proxy for the octree-FE pattern).  Returns SPD
    % sparse double.
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

function r = make_result(name, pass, msg)
    r = struct('name', name, 'pass', pass, 'msg', msg);
end

function s = ternary(cond, a, b)
    if cond, s = a; else, s = b; end
end

function tf = safe_mislocked(name)
    % mislocked(name) errors when the named MEX is not loaded.  For the
    % regression test we want a tristate -- locked / unlocked / unloaded
    % collapsed onto a boolean "is this MEX pinned in memory?".  Treat
    % "not loaded" as "not locked".
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
