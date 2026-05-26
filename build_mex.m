function build_mex(varargin)
%BUILD_MEX  Compile the cuDSS-backed MEX function for MATLAB.
%
%   build_mex                       % auto-detect cuDSS path + GPU arch
%   build_mex('CudssPath', P)       % override cuDSS install location
%   build_mex('Arch', 'sm_89')      % override the auto-detected GPU arch
%   build_mex('Debug', true)        % -G -O0 (line-level CUDA debug info)
%   build_mex('Clean', true)        % delete prior binaries first
%
%   REQUIREMENTS
%     * MATLAB R2024b or newer.  The +cudss wrappers assume native
%       single-precision sparse support (introduced in R2024b).
%     * CUDA Toolkit compatible with the installed cuDSS and MATLAB
%       (CUDA 12 was used in testing).
%     * NVIDIA cuDSS 0.7.1 installed locally (the only tested version);
%       install path resolved as described below.
%
%   Resolution order for the cuDSS install path:
%     1. 'CudssPath' name/value argument
%     2. CUDSS_PATH environment variable
%     3. Platform default:
%          Windows: C:\Program Files\NVIDIA cuDSS\v0.7
%          Linux  : /usr/local/cudss  (falls back to /opt/nvidia/cudss/0.7)
%
%   Resolution order for the target GPU compute capability (-arch=sm_XY):
%     1. 'Arch' name/value argument (must already be in 'sm_XY' form)
%     2. gpuDevice().ComputeCapability  (e.g. '8.9' -> 'sm_89')
%     3. Fallback: 'sm_89' (RTX 4080 / RTX 4090 / Hopper-era Ada)
%
%   Output binary lands in +cudss/private/, where MATLAB's package
%   dispatch finds it but it is not directly callable by users (the
%   +cudss/{factor,solve,destroy,wait}.m wrappers call it via the
%   string-command dispatcher in cudss_solver.cu).

    % --- enforce MATLAB R2024b or newer (native single sparse) ---
    if isMATLABReleaseOlderThan('R2024b')
        error('cudss:UnsupportedMATLAB', ...
              ['This wrapper requires MATLAB R2024b or newer (native ', ...
               'single-precision sparse).  Detected release: %s.'], ...
              matlabRelease().Release);
    end

    p = inputParser;
    addParameter(p, 'CudssPath', '', @(s) ischar(s) || isstring(s));
    addParameter(p, 'Arch',      '', @(s) ischar(s) || isstring(s));
    addParameter(p, 'Debug',     false, @islogical);
    addParameter(p, 'Clean',     false, @islogical);
    parse(p, varargin{:});

    % --- locate cuDSS ---
    cudss_path = char(p.Results.CudssPath);
    if isempty(cudss_path)
        cudss_path = getenv('CUDSS_PATH');
    end
    if isempty(cudss_path)
        if ispc
            cudss_path = 'C:\Program Files\NVIDIA cuDSS\v0.7';
        else
            candidates = {'/usr', '/usr/local/cudss', '/opt/nvidia/cudss/0.7'};
            cudss_path = '';
            for k = 1:numel(candidates)
                if isfolder(candidates{k})
                    cudss_path = candidates{k};
                    break
                end
            end
            if isempty(cudss_path)
                cudss_path = candidates{1};   % surface the missing-path error below
            end
        end
    end

    inc_dir = fullfile(cudss_path, 'include');
    if ~isfolder(inc_dir)
        error('cudss:BuildPath', ...
              'cuDSS include directory not found at %s. Set CUDSS_PATH or pass CudssPath.', ...
              inc_dir);
    end
    if ~isfile(fullfile(inc_dir, 'cudss.h'))
        error('cudss:BuildPath', ...
              'cudss.h not found at %s. Verify cuDSS version (tested with 0.7.1).', ...
              inc_dir);
    end

    % --- paths ---
    here    = fileparts(mfilename('fullpath'));
    src_dir = fullfile(here, 'mex');
    out_dir = fullfile(here, '+cudss', 'private');
    if ~isfolder(out_dir)
        mkdir(out_dir);
    end

    if p.Results.Clean
        old = dir(fullfile(out_dir, ['cudss_solver.' mexext]));
        for k = 1:numel(old)
            f = fullfile(out_dir, old(k).name);
            fprintf('  removing %s\n', f);
            delete(f);
        end
    end

    % --- mexcuda flags ---
    inc = ['-I"' inc_dir '"'];
    if ispc
        libdir_arg = ['-L"' fullfile(cudss_path, 'lib', 'x64') '"'];
    else
        libdir_arg = ['-L"' fullfile(cudss_path, 'lib64') '"'];
    end
    lib = '-lcudss';
    % The async-factor worker uses std::thread, which on Linux requires
    % linking against libpthread.  -lpthread is a no-op on Windows
    % (std::thread there uses Win32 primitives).
    pthread_lib = '-lpthread';

    % --- resolve target GPU architecture ---
    arch = char(p.Results.Arch);
    if isempty(arch)
        arch = detect_arch();
    end
    fprintf('Targeting GPU architecture: -arch=%s\n', arch);

    if p.Results.Debug
        nvccflags = sprintf('NVCCFLAGS=-arch=%s -G -O0 --allow-unsupported-compiler', arch);
    else
        nvccflags = sprintf('NVCCFLAGS=-arch=%s -O3 --use_fast_math --allow-unsupported-compiler', arch);
    end

    % Single dispatcher MEX -- one binary handles factor/solve/destroy/wait
    % via the command-string dispatcher in cudss_solver.cu.
    targets = {'cudss_solver'};

    for k = 1:numel(targets)
        src = fullfile(src_dir, [targets{k} '.cu']);
        if ~isfile(src)
            error('cudss:BuildPath', 'Source not found: %s', src);
        end
        fprintf('Building %s ...\n', targets{k});
        % -R2018a opts into the interleaved-complex MEX API so the typed
        % accessors (mxGetSingles / mxGetDoubles / mxGetInt32s) are
        % available.  -R2018a implies -largeArrayDims, so we do NOT
        % pass both -- mex rejects the combination.
        mexcuda('-v', '-R2018a', nvccflags, ...
                inc, libdir_arg, lib, pthread_lib, ...
                '-outdir', out_dir, src);
    end

    fprintf('\nBuild complete.  Binaries in %s\n', out_dir);

    % --- on Linux, hint at LD_LIBRARY_PATH if it doesn't include cuDSS ----
    if ~ispc
        ld = getenv('LD_LIBRARY_PATH');
        cudss_lib = fullfile(cudss_path, 'lib64');
        if ~contains(ld, cudss_lib)
            fprintf(['\nNote: %s is not on LD_LIBRARY_PATH.\n', ...
                     'If the MEX fails to dlopen libcudss.so, run before launching MATLAB:\n', ...
                     '    export LD_LIBRARY_PATH=%s:$LD_LIBRARY_PATH\n'], ...
                    cudss_lib, cudss_lib);
        end
    end
end

function arch = detect_arch()
%DETECT_ARCH  Best-effort -arch=sm_XY string from the active GPU.
%   Falls back to sm_89 (Ada / RTX 4080 / RTX 4090) if gpuDevice() is
%   unavailable or returns a malformed compute-capability string.  This
%   keeps the build flow working on machines without an attached GPU
%   (e.g., a build-only host).
    fallback = 'sm_89';
    try
        g = gpuDevice();
        cc = char(g.ComputeCapability);   % e.g. '8.9' on Ada, '7.5' on Turing
        digits = cc(isstrprop(cc, 'digit'));
        if isempty(digits)
            arch = fallback;
            return
        end
        arch = ['sm_' digits];
    catch
        arch = fallback;
    end
end
