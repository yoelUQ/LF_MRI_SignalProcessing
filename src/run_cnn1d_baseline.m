function results = run_cnn1d_paperstyle(matfile)
% CNN (DAY 9 paper-style): repeated Conv1D blocks + final 1x1 conv
% Input X: [N x 51 x 15], Target Y: [N x 51]

    rng(1);  % match DAY 9 split behavior

    % ---- robust load ----
    try
        whos('-file', matfile);
        S = load(matfile);
    catch ME
        error('CORRUPTED_MAT_FILE: %s', ME.message);
    end

    X = S.src_data;     % [N x 51 x 15]
    Y = S.tgt_data;     % [N x 51]
    xfreq = S.xfreq(:); % [51 x 1]

    [N,T,F] = size(X);
    assert(T==51 && F==15, 'Expected X: [N x 51 x 15]');
    assert(all(size(Y)==[N T]), 'Expected Y: [N x 51]');

    % ---- split: 80/10/10 (DAY 9) ----
    idx = randperm(N);
    nTrain = round(0.8*N);
    nVal   = round(0.1*N);
    idxTrain = idx(1:nTrain);
    idxVal   = idx(nTrain+1:nTrain+nVal);
    idxTest  = idx(nTrain+nVal+1:end);

    % ---- build cell sequences ----
    XTrain = cell(numel(idxTrain),1);
    YTrain = cell(numel(idxTrain),1);
    for k = 1:numel(idxTrain)
        i = idxTrain(k);
        xi = squeeze(X(i,:,:));     % 51 x 15
        XTrain{k} = xi.';           % 15 x 51
        YTrain{k} = Y(i,:);         % 1 x 51
    end

    XVal = cell(numel(idxVal),1);
    YVal = cell(numel(idxVal),1);
    for k = 1:numel(idxVal)
        i = idxVal(k);
        xi = squeeze(X(i,:,:));
        XVal{k} = xi.';            % 15 x 51
        YVal{k} = Y(i,:);          % 1 x 51
    end

    XTest = cell(numel(idxTest),1);
    YTest = cell(numel(idxTest),1);
    for k = 1:numel(idxTest)
        i = idxTest(k);
        xi = squeeze(X(i,:,:));
        XTest{k} = xi.';           % 15 x 51
        YTest{k} = Y(i,:);         % 1 x 51
    end

    % ---- DAY 9 paper CNN parameters (defaults) ----
    numFeatures = 15;
    numFiltersCNN = 16;
    kernelSizeCNN = 64;
    numLayersCNN  = 7;

    layers_cnn = buildNoiseCNN(numFeatures, numFiltersCNN, kernelSizeCNN, numLayersCNN);

    options = trainingOptions('adam', ...
        'MaxEpochs', 30, ...
        'MiniBatchSize', 16, ...
        'InitialLearnRate', 1e-3, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', {XVal, YVal}, ...
        'ValidationFrequency', 50, ...
        'Verbose', true);

    % ============================
    % MODEL CACHE (CNN1D)
    % ============================
    [matdir, matname, ~] = fileparts(matfile);
    modelDir = fullfile(matdir, 'models_cnn1d');
    if ~exist(modelDir,'dir'); mkdir(modelDir); end
    modelPath = fullfile(modelDir, sprintf('%s_cnn1d_net.mat', matname));
    
    if exist(modelPath, 'file')
        tmp = load(modelPath, 'net');
        net = tmp.net;
    else
        net = trainNetwork(XTrain, YTrain, layers_cnn, options);
    
        % Save trained net (v7.3 is safer for big objects)
        try
            save(modelPath, 'net', '-v7.3');
        catch
            save(modelPath, 'net');
        end
    end


    % ---- predict on test (cell) ----
    YhatCell = predict(net, XTest, 'MiniBatchSize', 128);

    % ---- stack to matrices ----
    Ntest = numel(idxTest);
    Yhat = zeros(Ntest, 51);
    Yte  = zeros(Ntest, 51);
    for i = 1:Ntest
        Yhat(i,:) = reshape(YhatCell{i}, 1, []);
        Yte(i,:)  = reshape(YTest{i},    1, []);
    end

    mse_test = mean((Yte(:) - Yhat(:)).^2);

    results.mse = mse_test;
    results.xfreq = xfreq;
    results.y_mean_test = mean(Yte,1);
    results.yhat_mean_test = mean(Yhat,1);

    % ============================
    % SAVE 3 FIGURES (same style)
    % ============================

    [matdir, matname, ~] = fileparts(matfile);
    outDir = fullfile(matdir, 'accuracy_plots_cnn1d');
    if ~exist(outDir, 'dir'); mkdir(outDir); end

    sharedIdxDir = fullfile(matdir, 'accuracy_plots_shared');
    if ~exist(sharedIdxDir, 'dir'); mkdir(sharedIdxDir); end

    if ~isempty(xfreq) && numel(xfreq) == size(Yte,2)
        xax = xfreq(:).';
        xlab = 'Frequency (Hz)';
    else
        xax = 1:size(Yte,2);
        xlab = 'Frequency bin';
    end

    shortLabel = makeShortLabel(matname);

    % fixed overlay indices shared across methods
    K = 6; seedPlot = 123;
    idxFile = fullfile(sharedIdxDir, sprintf('%s_fixedPlotIdx.mat', matname));
    if exist(idxFile, 'file')
        Sidx = load(idxFile, 'idx_plot');
        idx_plot = Sidx.idx_plot;
        idx_plot = idx_plot(idx_plot >= 1 & idx_plot <= size(Yte,1));
        if isempty(idx_plot)
            rng(seedPlot); idx_plot = randperm(size(Yte,1), min(K, size(Yte,1)));
            save(idxFile, 'idx_plot');
        end
    else
        rng(seedPlot); idx_plot = randperm(size(Yte,1), min(K, size(Yte,1)));
        save(idxFile, 'idx_plot');
    end

    % 1) samples
    fig1 = figure('Color','w');
    try
        tl = tiledlayout(3,2,'TileSpacing','compact','Padding','compact');
        hTarget=[]; hPred=[];
        for k = 1:min(6, numel(idx_plot))
            ii = idx_plot(k);
            ax = nexttile; hold(ax,'on');
            p1 = plot(ax, xax, Yte(ii,:), 'k', 'LineWidth', 1.2);
            p2 = plot(ax, xax, Yhat(ii,:), 'r--', 'LineWidth', 1.2);
            if k==1, hTarget=p1; hPred=p2; end
            grid(ax,'on');
            title(ax, sprintf('Sample %d', ii), 'Interpreter','none');
            xlabel(ax, xlab); ylabel(ax,'Noise amplitude');
        end
        for k = numel(idx_plot)+1:6
            ax = nexttile; axis(ax,'off');
        end
        if ~isempty(hTarget) && ~isempty(hPred)
            legend([hTarget hPred], {'Target','Predicted'}, 'Location','southoutside', 'Orientation','horizontal');
        end
        sgtitle(tl, sprintf('%s | CNN1D | Test MSE=%.4g', shortLabel, mse_test), 'Interpreter','none');
    catch
        clf(fig1);
        ii = idx_plot(1);
        plot(xax, Yte(ii,:), 'k', 'LineWidth', 1.2); hold on;
        plot(xax, Yhat(ii,:), 'r--', 'LineWidth', 1.2);
        grid on; xlabel(xlab); ylabel('Noise amplitude');
        legend('Target','Predicted','Location','best');
        title(sprintf('%s | CNN1D | Test MSE=%.4g', shortLabel, mse_test), 'Interpreter','none');
    end
    exportgraphics(fig1, fullfile(outDir, sprintf('%s_cnn1d_samples.png', matname)), 'Resolution', 200);
    close(fig1);

    % 2) per-sample MSE hist
    mse_per_sample = mean((Yhat - Yte).^2, 2);
    fig2 = figure('Color','w');
    histogram(mse_per_sample, 60); grid on;
    xlabel('Per-sample MSE'); ylabel('Count');
    title(sprintf('%s | CNN1D | Per-sample MSE (median=%.3g, 90%%=%.3g)', ...
        shortLabel, median(mse_per_sample), prctile(mse_per_sample,90)), 'Interpreter','none');
    exportgraphics(fig2, fullfile(outDir, sprintf('%s_cnn1d_mse_hist.png', matname)), 'Resolution', 200);
    close(fig2);

    % 3) freq error mean + IQR
    E2 = (Yhat - Yte).^2;
    mse_f = mean(E2,1); q25=prctile(E2,25,1); q75=prctile(E2,75,1);
    fig3 = figure('Color','w'); hold on; grid on;
    plot(xax, mse_f, 'LineWidth', 1.8);
    plot(xax, q25, '--', 'LineWidth', 1.0);
    plot(xax, q75, '--', 'LineWidth', 1.0);
    xlabel(xlab); ylabel('Squared error');
    legend('Mean SE','25th pct','75th pct','Location','best');
    title(sprintf('%s | CNN1D | Frequency-wise error (mean + IQR)', shortLabel), 'Interpreter','none');
    exportgraphics(fig3, fullfile(outDir, sprintf('%s_cnn1d_freq_iqr.png', matname)), 'Resolution', 200);
    close(fig3);

end

% ---- DAY 9 paper-style CNN builder ----
function layers = buildNoiseCNN(numFeatures, numFilters, kernelSize, numLayers)
    layers = [
        sequenceInputLayer(numFeatures, "Name","input", "Normalization","zscore")
    ];

    for L = 1:numLayers
        layers = [
            layers
            convolution1dLayer(kernelSize, numFilters, ...
                "Padding","same", ...
                "Name", sprintf("conv_%d",L))
            reluLayer("Name", sprintf("relu_%d",L))
        ];
    end

    layers = [
        layers
        convolution1dLayer(1, 1, "Padding","same", "Name","conv_final")
        regressionLayer("Name","regression_output")
    ];
end

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
    if isempty(fk), s = spec; else, s = [spec ' ' fk]; end
end
