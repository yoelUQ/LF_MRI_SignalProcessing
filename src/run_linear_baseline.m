function results = run_linear_baseline(matfile)
% Linear regression baseline for MRI noise synthesis
% Per-frequency independent linear models
%
% INPUT:
%   matfile : path to .mat file containing
%       src_data : [N x 51 x 15]
%       tgt_data : [N x 51]
%       xfreq    : [51 x 1]
%
% OUTPUT:
%   results struct with fields:
%       .mse
%       .xfreq
%       .y_mean_test
%       .yhat_mean_test

    rng(42);   % fixed seed for reproducibility (UNCHANGED)

    % ------------------
    % Load data (robust)
    % ------------------
    try
        whos('-file', matfile);
        S = load(matfile);

        X = S.src_data;     % [N x 51 x 15]
        Y = S.tgt_data;     % [N x 51]
        xfreq = S.xfreq(:); % [51 x 1]

        [N, F, D] = size(X);
        assert(F == 51, 'Expected 51 frequency bins');
        assert(D == 15, 'Expected 15 input features');
        assert(all(size(Y) == [N F]), 'Expected tgt_data to be [N x 51]');
    catch ME
        error('CORRUPTED_MAT_FILE: %s', ME.message);
    end

    % ------------------
    % Train / test split (UNCHANGED)
    % ------------------
    idx = randperm(N);
    ntrain = round(0.8 * N);

    train_idx = idx(1:ntrain);
    test_idx  = idx(ntrain+1:end);

    % ------------------
    % Storage
    % ------------------
    Yhat_test = zeros(numel(test_idx), F);
    mse_f_fit = zeros(F,1);

    % ------------------
    % Per-frequency linear regression (UNCHANGED)
    % ------------------
    for f = 1:F
        Xf_train = squeeze(X(train_idx, f, :));  % [Ntrain x 15]
        yf_train = Y(train_idx, f);              % [Ntrain x 1]

        Xf_test  = squeeze(X(test_idx, f, :));   % [Ntest x 15]
        yf_test  = Y(test_idx, f);               % [Ntest x 1]

        beta = Xf_train \ yf_train;              % OLS
        yhat = Xf_test * beta;

        Yhat_test(:, f) = yhat;
        mse_f_fit(f)    = mean((yf_test - yhat).^2);
    end

    % ------------------
    % Define test matrices once
    % ------------------
    Yte  = Y(test_idx, :);
    Yhat = Yhat_test;

    mse_test = mean((Yhat(:) - Yte(:)).^2);  % used in titles/plots only

    % ------------------
    % Outputs (UNCHANGED)
    % ------------------
    results.mse   = mean(mse_f_fit);
    results.xfreq = xfreq;
    results.y_mean_test    = mean(Yte, 1);
    results.yhat_mean_test = mean(Yhat, 1);

    % ============================
    % PLOTS (CHANGED ONLY)
    % Saved: samples / mse_hist / freq_iqr
    % ============================

    [matdir, matname, ~] = fileparts(matfile);

    % Method output folder
    outDir = fullfile(matdir, 'accuracy_plots_linear');
    if ~exist(outDir, 'dir'); mkdir(outDir); end

    % Shared index folder so ALL methods use same example samples
    sharedIdxDir = fullfile(matdir, 'accuracy_plots_shared');
    if ~exist(sharedIdxDir, 'dir'); mkdir(sharedIdxDir); end

    % ---- x-axis ----
    if ~isempty(xfreq) && numel(xfreq) == size(Yte,2)
        xax = xfreq(:).';
        xlab = 'Frequency (Hz)';
    else
        xax = 1:size(Yte,2);
        xlab = 'Frequency bin';
    end

    % ---- short label (no underscores) ----
    shortLabel = makeShortLabel(matname);

    % ---- fixed overlay indices (shared across methods) ----
    K = 6;
    seedPlot = 123;
    idxFile = fullfile(sharedIdxDir, sprintf('%s_fixedPlotIdx.mat', matname));

    if exist(idxFile, 'file')
        Sidx = load(idxFile, 'idx_plot');
        idx_plot = Sidx.idx_plot;
        idx_plot = idx_plot(idx_plot >= 1 & idx_plot <= size(Yte,1));
        if isempty(idx_plot)
            rng(seedPlot);
            idx_plot = randperm(size(Yte,1), min(K, size(Yte,1)));
            save(idxFile, 'idx_plot');
        end
    else
        rng(seedPlot);
        idx_plot = randperm(size(Yte,1), min(K, size(Yte,1)));
        save(idxFile, 'idx_plot');
    end

    % ---- 1) Samples overlay (3x2) ----
    fig1 = figure('Color','w');
    try
        tl = tiledlayout(3, 2, 'TileSpacing','compact', 'Padding','compact');
        hTarget = []; hPred = [];

        for k = 1:min(6, numel(idx_plot))
            i = idx_plot(k);
            ax = nexttile; hold(ax, 'on');
            p1 = plot(ax, xax, Yte(i,:),  'k',  'LineWidth', 1.2);
            p2 = plot(ax, xax, Yhat(i,:), 'r--','LineWidth', 1.2);
            if k == 1, hTarget = p1; hPred = p2; end
            grid(ax, 'on');
            title(ax, sprintf('Sample %d', i), 'Interpreter','none');
            xlabel(ax, xlab); ylabel(ax, 'Noise amplitude');
        end

        for k = numel(idx_plot)+1 : 6
            ax = nexttile; axis(ax, 'off');
        end

        if ~isempty(hTarget) && ~isempty(hPred)
            legend([hTarget hPred], {'Target','Predicted'}, ...
                'Location','southoutside', 'Orientation','horizontal');
        end

        sgtitle(tl, sprintf('%s | Linear | Test MSE=%.4g', shortLabel, mse_test), 'Interpreter','none');
    catch
        clf(fig1);
        i = idx_plot(1);
        plot(xax, Yte(i,:), 'k', 'LineWidth', 1.2); hold on;
        plot(xax, Yhat(i,:), 'r--', 'LineWidth', 1.2);
        grid on; xlabel(xlab); ylabel('Noise amplitude');
        legend('Target','Predicted','Location','best');
        title(sprintf('%s | Linear | Test MSE=%.4g', shortLabel, mse_test), 'Interpreter','none');
    end
    exportgraphics(fig1, fullfile(outDir, sprintf('%s_linear_samples.png', matname)), 'Resolution', 200);
    close(fig1);

    % ---- 2) Per-sample MSE histogram ----
    mse_per_sample = mean((Yhat - Yte).^2, 2);

    fig2 = figure('Color','w');
    histogram(mse_per_sample, 60);
    grid on;
    xlabel('Per-sample MSE'); ylabel('Count');
    title(sprintf('%s | Linear | Per-sample MSE (median=%.3g, 90%%=%.3g)', ...
        shortLabel, median(mse_per_sample), prctile(mse_per_sample, 90)), 'Interpreter','none');
    exportgraphics(fig2, fullfile(outDir, sprintf('%s_linear_mse_hist.png', matname)), 'Resolution', 200);
    close(fig2);

    % ---- 3) Frequency-wise error (mean + IQR) ----
    E2 = (Yhat - Yte).^2;
    mse_f = mean(E2, 1);
    q25  = prctile(E2, 25, 1);
    q75  = prctile(E2, 75, 1);

    fig3 = figure('Color','w'); hold on; grid on;
    plot(xax, mse_f, 'LineWidth', 1.8);
    plot(xax, q25, '--', 'LineWidth', 1.0);
    plot(xax, q75, '--', 'LineWidth', 1.0);
    xlabel(xlab); ylabel('Squared error');
    legend('Mean SE','25th pct','75th pct', 'Location','best');
    title(sprintf('%s | Linear | Frequency-wise error (mean + IQR)', shortLabel), 'Interpreter','none');
    exportgraphics(fig3, fullfile(outDir, sprintf('%s_linear_freq_iqr.png', matname)), 'Resolution', 200);
    close(fig3);

end

% -------- helper: short label without underscores ----------
function s = makeShortLabel(matname)
    m = upper(matname);
    if contains(m,'_ABS_'), spec = 'ABS';
    elseif contains(m,'_REAL_'), spec = 'REAL';
    elseif contains(m,'_IMG_') || contains(m,'_IMAG_'), spec = 'IMAG';
    else, spec = 'DATA';
    end

    tok = regexp(matname, '_(\d{2,3})kHz', 'tokens', 'once');
    if isempty(tok), fk = '';
    else, fk = [tok{1} 'kHz'];
    end

    if isempty(fk), s = spec;
    else, s = [spec ' ' fk];
    end
end
