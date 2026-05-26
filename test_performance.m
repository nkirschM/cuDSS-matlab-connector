%% test_performance.m -- speed benchmark: cuDSS (GPU) vs MATLAB host solver.
%
% Builds a 3-D 7-point Laplacian K = K(N) on an N-by-N-by-N grid and
% times factor and per-RHS solve in both 'single' and 'double'
% precision.  K is SPD by construction.
%
% cuDSS is benchmarked under all three matrix_type configurations the
% wrapper exposes; each selects a different factorization algorithm:
%
%     matrix_type = 'general'    -> LU       (no symmetry assumption)
%     matrix_type = 'symmetric'  -> LDL^T    (K declared symmetric)
%     matrix_type = 'spd'        -> Cholesky (K declared SPD)
%
% For 'symmetric' and 'spd' the wrapper transfers only the upper
% triangle, so device memory and factor cost can drop substantially
% relative to 'general'.
%
% Side-by-side figure layout (single precision | double precision):
%
%     Figure 1 (assets/perf_solve.png)
%         Solve time per RHS vs number of unknowns.
%         Each cell times one multi-RHS solve with an n-by-50 dense L
%         and reports the result divided by 50.  This exercises the
%         multi-RHS path: one cudss.solve call performs forward/back
%         substitution across all 50 columns in a single launch.
%
%     Figure 2 (assets/perf_factor.png)
%         Factor time vs number of unknowns.
%
% Each figure draws four lines per subplot (CPU baseline + three GPU
% matrix_type variants).  Each timing cell is the median of n_runs
% trials; cells that error (e.g. cuDSS rejecting a config, VRAM
% exhaustion) are caught and left as NaN, which the log-log plot
% renders as a gap.
%
% Raw timings are saved to assets/perf_results.mat.
%
% Run from MATLAB after build_mex:
%
%     addpath(pwd);
%     run('test_performance.m');

clc; clearvars; close all;

this_dir = fileparts(mfilename('fullpath'));
addpath(this_dir);

which_factor = which('cudss.factor');
if isempty(which_factor)
    error('test_performance:NoPackage', ...
          'cudss.factor not on path.  Did you run build_mex first?');
end
fprintf('cudss.factor resolved at %s\n', which_factor);

[cpu_label, gpu_label] = identify_hardware();
fprintf('CPU: %s\n', cpu_label);
fprintf('GPU: %s\n', gpu_label);

% --- problem sizes: log-spaced N spanning n = 10^3 .. 10^6 unknowns ---
% Sparse-direct fill for a 3-D Laplacian is O(N^4), so each step up in
% N grows the host CHOLMOD working set sharply.  Increase the upper
% bound only if your host has the memory headroom to factor it.
N_list      = [10, 14, 19, 26, 36, 50, 70, 100];
n_list      = N_list.^3;
precisions  = {'single', 'double'};
gpu_configs = {'general', 'symmetric', 'spd'};   % cuDSS matrix_type
n_runs      = 3;                                 % timed trials per cell
nrhs        = 50;                                % multi-RHS solve width

n_N    = numel(N_list);
times  = struct();
for pp = 1:numel(precisions)
    p = precisions{pp};
    times.(p).cpu.factor = nan(n_N, 1);
    times.(p).cpu.solve  = nan(n_N, 1);
    for gc = 1:numel(gpu_configs)
        c = gpu_configs{gc};
        times.(p).gpu.(c).factor = nan(n_N, 1);
        times.(p).gpu.(c).solve  = nan(n_N, 1);
    end
end

% --- driver init warmup so the first timed factor isn't paying for it ---
fprintf('\n[warmup] cuDSS driver init on small K...\n');
K_warm = build_3d_laplacian(6);
h_warm = cudss.factor(K_warm, struct('precision', 'single'));
W_warm = cudss.solve(h_warm, gpuArray.zeros(size(K_warm,1), 1, 'single')); %#ok<NASGU>
wait(gpuDevice());
cudss.destroy(h_warm);
clear K_warm h_warm W_warm

% --- main sweep ---
for pp = 1:numel(precisions)
    prec    = precisions{pp};
    cast_fn = str2func(prec);
    for kk = 1:n_N
        N = N_list(kk);
        n = N^3;
        fprintf('\n[%s] N = %d (n = %d)\n', prec, N, n);

        K_dbl = build_3d_laplacian(N);
        rng(0);
        L_dbl = randn(n, nrhs);

        % ---- CPU baseline.  decomposition() auto-picks Cholesky for
        % an SPD sparse K; the call has no first-touch state, so no
        % warmup is needed.
        try
            K_cpu = cast_to_precision(K_dbl, prec);
            L_cpu = cast_to_precision(L_dbl, prec);

            t_fact = zeros(n_runs, 1);
            t_solv = zeros(n_runs, 1);
            for r = 1:n_runs
                tic; D = decomposition(K_cpu); t_fact(r) = toc;
                tic; W = D \ L_cpu;            t_solv(r) = toc / nrhs; %#ok<NASGU>
                clear D W
            end
            times.(prec).cpu.factor(kk) = median(t_fact);
            times.(prec).cpu.solve(kk)  = median(t_solv);
            fprintf('  CPU              factor %.4f s, solve %.4f s/RHS (nrhs=%d)\n', ...
                    times.(prec).cpu.factor(kk), times.(prec).cpu.solve(kk), nrhs);
        catch err_cpu
            fprintf('  CPU skipped: %s\n', trim_msg(err_cpu.message, 200));
        end
        clear K_cpu L_cpu

        % ---- GPU: loop over the three cuDSS matrix_type configs ----
        K_gpu_in = cast_to_precision(K_dbl, prec);
        L_gpu    = gpuArray(cast_fn(L_dbl));

        for gc = 1:numel(gpu_configs)
            cfg = gpu_configs{gc};
            try
                opts = struct('precision', prec, 'matrix_type', cfg);

                % per-(N, cfg) warmup so first-touch allocations are amortized
                h_w = cudss.factor(K_gpu_in, opts);
                W_w = cudss.solve(h_w, L_gpu); wait(gpuDevice()); %#ok<NASGU>
                cudss.destroy(h_w);

                t_fact_g = zeros(n_runs, 1);
                t_solv_g = zeros(n_runs, 1);
                for r = 1:n_runs
                    tic;
                    h = cudss.factor(K_gpu_in, opts);
                    wait(gpuDevice());
                    t_fact_g(r) = toc;

                    tic;
                    W = cudss.solve(h, L_gpu);
                    wait(gpuDevice());
                    t_solv_g(r) = toc / nrhs; %#ok<NASGU>

                    cudss.destroy(h);
                end
                times.(prec).gpu.(cfg).factor(kk) = median(t_fact_g);
                times.(prec).gpu.(cfg).solve(kk)  = median(t_solv_g);
                fprintf('  GPU %-10s factor %.4f s, solve %.4f s/RHS\n', ...
                        cfg, times.(prec).gpu.(cfg).factor(kk), ...
                        times.(prec).gpu.(cfg).solve(kk));
            catch err_gpu
                fprintf('  GPU %-10s skipped: %s\n', cfg, ...
                        trim_msg(err_gpu.message, 200));
            end
        end

        % free per-N memory before the next size
        clear K_dbl L_dbl K_gpu_in L_gpu W W_w h h_w
        wait(gpuDevice());
    end
end

% --- emit plots + raw timings ---
assets_dir = fullfile(this_dir, 'assets');
if ~exist(assets_dir, 'dir'), mkdir(assets_dir); end

solve_path  = fullfile(assets_dir, 'perf_solve.png');
factor_path = fullfile(assets_dir, 'perf_factor.png');
mat_path    = fullfile(assets_dir, 'perf_results.mat');

make_side_by_side(n_list, times, 'solve',  cpu_label, gpu_label, ...
                  gpu_configs, solve_path);
make_side_by_side(n_list, times, 'factor', cpu_label, gpu_label, ...
                  gpu_configs, factor_path);

save(mat_path, 'N_list', 'n_list', 'times', 'cpu_label', 'gpu_label', ...
     'n_runs', 'nrhs', 'gpu_configs');

fprintf('\nWrote:\n  %s\n  %s\n  %s\n', solve_path, factor_path, mat_path);

% --- per-precision summary tables to stdout ---
print_summary_table(N_list, n_list, times, precisions, gpu_configs);


% =========================================================================
%  Helpers
% =========================================================================

function s = ternary(cond, a, b)
    if cond, s = a; else, s = b; end
end

function out = trim_msg(s, maxlen)
    if isempty(s), out = ''; return; end
    s = strrep(s, sprintf('\n'), ' ');
    if numel(s) > maxlen
        out = [s(1:maxlen) '...'];
    else
        out = s;
    end
end

function K = build_3d_laplacian(N)
    % Standard 7-point Laplacian on an N-by-N-by-N grid, Dirichlet zero
    % on the boundary.  SPD sparse double.
    e  = ones(N, 1);
    T1 = spdiags([-e, 2*e, -e], -1:1, N, N);
    I  = speye(N);
    K  = kron(kron(I, I), T1) + kron(kron(I, T1), I) + kron(kron(T1, I), I);
end

function X = cast_to_precision(X, prec)
    if strcmp(prec, 'single') && ~isa(X, 'single')
        X = single(X);
    elseif strcmp(prec, 'double') && ~isa(X, 'double')
        X = double(X);
    end
end

function [cpu_label, gpu_label] = identify_hardware()
    cpu_label = 'CPU';
    if isunix && ~ismac
        [s, out] = system('lscpu | sed -n ''s/^Model name:[[:space:]]*//p''');
        if s == 0
            t = regexprep(strtrim(out), '\s+', ' ');
            if ~isempty(t), cpu_label = t; end
        end
    elseif ismac
        [s, out] = system('sysctl -n machdep.cpu.brand_string');
        if s == 0
            t = strtrim(out);
            if ~isempty(t), cpu_label = t; end
        end
    elseif ispc
        [s, out] = system('wmic cpu get Name /value');
        if s == 0
            tok = regexp(out, 'Name=([^\r\n]+)', 'tokens', 'once');
            if ~isempty(tok), cpu_label = strtrim(tok{1}); end
        end
    end

    gpu_label = 'GPU';
    try
        g = gpuDevice();
        gpu_label = g.Name;
    catch
    end
end

function make_side_by_side(n_list, times, kind, cpu_label, gpu_label, ...
                           gpu_configs, out_path)
    % Wong colorblind-safe palette
    wong = {'#D55E00', '#0072B2', '#009E73', '#CC79A7', ...
            '#E69F00', '#56B4E9', '#F0E442'};

    % Per-line styling: { display_label, source, cfg-or-empty, marker, color }
    %   source 'cpu' -> times.(prec).cpu.(field)
    %   source 'gpu' -> times.(prec).gpu.(cfg).(field)
    cfg_display = struct('general',   'cudss general (LU)', ...
                         'symmetric', 'cudss symmetric (LDL^{T})', ...
                         'spd',       'cudss SPD (Cholesky)');
    cfg_marker  = struct('general', 's', 'symmetric', '^', 'spd', 'd');

    series = { {'CPU (decomposition)', 'cpu', '', 'o', wong{1}} };
    for gc = 1:numel(gpu_configs)
        c = gpu_configs{gc};
        series{end+1} = { cfg_display.(c), 'gpu', c, cfg_marker.(c), wong{1+gc} };  %#ok<AGROW>
    end

    is_solve   = strcmp(kind, 'solve');
    field      = ternary(is_solve, 'solve', 'factor');
    y_label    = ternary(is_solve, ...
                         'time per RHS [s]   (one nrhs = 50 solve / 50)', ...
                         'factor time [s]');
    title_lead = ternary(is_solve, 'Solve time per RHS', 'Factor time');
    title_sub  = ternary(is_solve, ...
                         '  multi-RHS solve (nrhs = 50), median of 3 trials', ...
                         '  median of 3 trials');

    % Shared y-limits across both subplots so the left/right panels are
    % visually comparable.
    precisions = {'single', 'double'};
    all_t = [];
    for pp = 1:2
        p = precisions{pp};
        all_t = [all_t; times.(p).cpu.(field)]; %#ok<AGROW>
        for gc = 1:numel(gpu_configs)
            c = gpu_configs{gc};
            all_t = [all_t; times.(p).gpu.(c).(field)]; %#ok<AGROW>
        end
    end
    all_t = all_t(isfinite(all_t) & all_t > 0);
    ylim_pad = [10^floor(log10(min(all_t))), 10^ceil(log10(max(all_t)))];

    fig = figure('Position', [100, 100, 1300, 540], 'Color', 'w', ...
                 'Visible', 'off');

    LW = 2.8;
    MS = 9;

    for pp = 1:2
        prec = precisions{pp};
        subplot(1, 2, pp); hold on;
        for sk = 1:numel(series)
            s = series{sk};
            label  = s{1}; src = s{2}; cfg = s{3};
            marker = s{4}; color = s{5};
            if isempty(cfg)
                y = times.(prec).(src).(field);
            else
                y = times.(prec).(src).(cfg).(field);
            end
            loglog(n_list, y, ['-' marker], ...
                   'LineWidth',       LW, ...
                   'MarkerSize',      MS, ...
                   'Color',           color, ...
                   'MarkerFaceColor', color, ...
                   'MarkerEdgeColor', 'none', ...
                   'DisplayName',     label);
        end
        set(gca, 'XScale', 'log', 'YScale', 'log');
        grid on; grid minor;
        xlabel('number of unknowns');
        ylabel(y_label);
        title(sprintf('%s precision', prec));
        ylim(ylim_pad);
        legend('Location', 'northwest');
        set(gca, 'FontSize', 11, 'LineWidth', 1.0);
    end

    sgtitle(sprintf(['%s vs number of unknowns  --  3-D 7-point Laplacian,%s' ...
                     '\nCPU: %s  |  GPU: %s'], ...
                     title_lead, title_sub, cpu_label, gpu_label), ...
            'FontSize', 12, 'FontWeight', 'bold');

    exportgraphics(fig, out_path, 'Resolution', 150);
    close(fig);
end

function print_summary_table(N_list, n_list, times, precisions, gpu_configs)
    for pp = 1:numel(precisions)
        prec = precisions{pp};
        fprintf('\n=== %s precision (median seconds; solve = per-RHS) ===\n', prec);
        hdr = sprintf('  %4s %10s | %8s %8s', 'N', 'unknowns', 'cpu_f', 'cpu_s');
        for gc = 1:numel(gpu_configs)
            c = gpu_configs{gc};
            hdr = [hdr sprintf(' | %8s %8s', [c(1:min(3,end)) '_f'], ...
                                              [c(1:min(3,end)) '_s'])]; %#ok<AGROW>
        end
        fprintf('%s\n', hdr);
        for kk = 1:numel(N_list)
            row = sprintf('  %4d %10d | %s %s', ...
                          N_list(kk), n_list(kk), ...
                          fmt_t(times.(prec).cpu.factor(kk)), ...
                          fmt_t(times.(prec).cpu.solve(kk)));
            for gc = 1:numel(gpu_configs)
                c = gpu_configs{gc};
                row = [row sprintf(' | %s %s', ...
                                   fmt_t(times.(prec).gpu.(c).factor(kk)), ...
                                   fmt_t(times.(prec).gpu.(c).solve(kk)))]; %#ok<AGROW>
            end
            fprintf('%s\n', row);
        end
    end
end

function s = fmt_t(t)
    if isnan(t) || ~isfinite(t)
        s = sprintf('%8s', '--');
    else
        s = sprintf('%8.4f', t);
    end
end
