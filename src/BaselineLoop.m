clc; clear; close all;

baseDir = pwd;
outDir  = fullfile(baseDir, 'model_outputs');
if ~exist(outDir,'dir'); mkdir(outDir); end

spectra   = {'abs','real','img'};
freqs_kHz = [80 85 90 95];
prefix    = 'CNN_input_5ms_';

rows = [];
k = 0;

for s = 1:numel(spectra)
    spec = spectra{s};

    for ff = 1:numel(freqs_kHz)
        fk = freqs_kHz(ff);

        fileA = fullfile(baseDir, sprintf('%s%s_%03dkHz.mat', prefix, spec, fk));
        fileB = fullfile(baseDir, sprintf('%s%s_%dkHz.mat',   prefix, spec, fk));

        if exist(fileA,'file')
            matfile = fileA;
        elseif exist(fileB,'file')
            matfile = fileB;
        else
            warning('Missing: %s %dkHz', spec, fk);
            continue;
        end

        try
            results = run_all_baselines(matfile);   % <-- IMPORTANT
        catch ME
            if contains(ME.message,'CORRUPTED_MAT_FILE')
                warning('Skipping corrupted file: %s', matfile);
                continue;
            else
                rethrow(ME);
            end
        end

        [~,name,~] = fileparts(matfile);

        % Record 3 rows per dataset (linear/dense/cnn1d)
        models = {'linear','dense','cnn1d'};
        modelNames = {'Linear','Dense','CNN1D'};

        for mi = 1:numel(models)
            mkey = models{mi};

            if ~isfield(results, mkey) || ~isfield(results.(mkey), 'mse')
                warning('Missing model output: %s for %s', mkey, matfile);
                continue;
            end

            k = k + 1;
            rows(k).model    = string(modelNames{mi});
            rows(k).spectrum = string(spec);
            rows(k).freq_kHz = fk;
            rows(k).mse      = results.(mkey).mse;
            rows(k).file     = string(name);
        end
    end
end

T = struct2table(rows);
T = sortrows(T, {'model','spectrum','freq_kHz'});

disp(T(:,{'model','spectrum','freq_kHz','mse','file'}));

csvPath = fullfile(outDir, 'summary_all_baselines.csv');
writetable(T, csvPath);
fprintf('\nSaved: %s\n', csvPath);

% ----------------------------
% MSE OVERVIEW plots (one per model, same style as before)
% ----------------------------
models = unique(T.model);

for mi = 1:numel(models)
    mdl = models(mi);

    figure('Color','w'); hold on; grid on;
    for s = 1:numel(spectra)
        spec = spectra{s};

        mask = (T.model == mdl) & (T.spectrum == spec);
        Ts = T(mask,:);
        if isempty(Ts); continue; end

        Ts = sortrows(Ts, 'freq_kHz');
        plot(Ts.freq_kHz, Ts.mse, '-o', 'LineWidth', 1.5);
    end

    xlabel('Frequency (kHz)');
    ylabel('MSE');
    title(sprintf('MSE overview | %s', mdl));
    legend(spectra, 'Location', 'best');

    overviewPath = fullfile(outDir, sprintf('MSE_overview_%s.png', lower(char(mdl))));
    exportgraphics(gcf, overviewPath, 'Resolution', 200);
    close(gcf);

    fprintf('Saved MSE overview: %s\n', overviewPath);
end
