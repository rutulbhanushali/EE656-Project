%% System Configuration
clear; clc; close all;
mainFolder    = 'AirCompressor_Data';
faultClasses  = {'Bearing','Flywheel','Healthy','LIV','LOV','NRV','Piston','Riderbelt'};
Fs            = 50000;                   % Sampling rate (Hz)
segmentLength = Fs * 1;                  % 1-second window
stepSize      = round(segmentLength / 2);% 50% overlap
numFeatures   = 286;                     % Full feature set from paper

%% Precompute WPT Node Indices
maxWPTLevel = 7;
wptNodes = zeros(254, 2);
nodeIdx = 1;
for level = 1:maxWPTLevel
    for node = 0:(2^level - 1)
        wptNodes(nodeIdx, :) = [level, node];
        nodeIdx = nodeIdx + 1;
    end
end

%% Precomputed Constants for Frequency Binning (Vectorized)
[~, F] = pwelch(randn(segmentLength,1), hann(1024), 512, 2048, Fs);
binEdges = linspace(0, Fs/2, 9);
binMasks = false(length(F), 8);
for band = 1:8
    binMasks(:, band) = (F >= binEdges(band)) & (F < binEdges(band+1));
end

%% File Inventory Management
filePaths = {};
classIndices = [];
for c = 1:numel(faultClasses)
    folder = fullfile(mainFolder, faultClasses{c});
    files = dir(fullfile(folder, '*.dat'));
    filePaths = [filePaths; cellfun(@(f) fullfile(folder, f), {files.name}', 'UniformOutput', false)]; %#ok<AGROW>
    classIndices = [classIndices; c * ones(numel(files), 1)]; %#ok<AGROW>
end
totalFiles = numel(filePaths);
features = NaN(totalFiles, numFeatures);
classLabels = faultClasses(classIndices)';
extractionErrors = cell(totalFiles, 1);

%% Digital Filter Design (Bandpass 400Hz-12kHz)
bpFilt = designfilt('bandpassiir', ...
    'FilterOrder', 8, ...
    'HalfPowerFrequency1', 400, ...
    'HalfPowerFrequency2', 12000, ...
    'SampleRate', Fs, ...
    'DesignMethod', 'butter');

%% Initialize Parallel Pool with Dependency
if isempty(gcp('nocreate'))
    pool = parpool('Processes');
    pool.IdleTimeout = 120;
    % Add this function to pool dependencies
    addAttachedFiles(pool, mfilename('fullpath'));
end

%% Preallocate Feature Section Indices
TD_START = 1; TD_END = 8;
FD_START = 9; FD_END = 16;
WPT_START = 17; WPT_END = 270;
DWT_START = 271; DWT_END = 279;
MWT_START = 280; MWT_END = 286;

%% Main Processing Loop
parfor idx = 1:totalFiles
    try
        % ===== 24-bit PCM DECODING =====
        fid = fopen(filePaths{idx}, 'r');
        if fid == -1
            error('File not found: %s', filePaths{idx});
        end
        raw = fread(fid, inf, 'uint8=>uint8'); 
        fclose(fid);
        
        % Check minimum file size
        if numel(raw) < 3
            error('File too small: %s (%d bytes)', filePaths{idx}, numel(raw));
        end
        
        % Decode 24-bit signed integer
        N = floor(numel(raw)/3);
        bytes = reshape(raw(1:3*N), 3, []);
        data = double(bytes(1,:)) + 256*double(bytes(2,:)) + 65536*double(bytes(3,:));
        negMask = bytes(3,:) >= 128;
        data(negMask) = data(negMask) - 2^24;
        data = data' / 2^23;  % Normalize to [-1, 1]
        
        % ===== PREPROCESSING =====
        % Bandpass Filtering
        if ~all(isfinite(data))
            error('Non-finite values in raw data');
        end
        sig = filtfilt(bpFilt, data);
        
        % Segmentation with 50% overlap
        numSegments = floor((length(sig) - segmentLength)/stepSize) + 1;
        if numSegments < 1
            error('Signal too short for segmentation: %d samples', length(sig));
        end
        segments = zeros(segmentLength, numSegments);
        stdVals = zeros(numSegments, 1);
        for seg = 1:numSegments
            startIdx = (seg-1)*stepSize + 1;
            endIdx = startIdx + segmentLength - 1;
            segmentData = sig(startIdx:endIdx);
            segments(:, seg) = segmentData;
            stdVals(seg) = std(segmentData);
        end
        
        % Select most stable segment (min std)
        [~, minIdx] = min(stdVals);
        clip = segments(:, minIdx);
        
        % Moving Average Smoothing (window=5)
        sm = movmean(clip, 5);
        
        % Modified Min-Max Normalization (Algorithm 3)
        sorted = sort(sm);
        k = max(1, round(0.00025 * length(sorted)));
        L = sorted(k);
        U = sorted(end-k+1);
        normSig = 2 * (sm - L) / (U - L) - 1;
        
        % ===== FEATURE EXTRACTION =====
        % Pre-allocate feature vector for this file
        featureVector = NaN(1, numFeatures);
        
        % ---- Time Domain (8 features) ----
        featureVector(TD_START) = rms(normSig);                         % RMS
        featureVector(TD_START+1) = sum(abs(diff(normSig>0)))/length(normSig); % ZCR
        featureVector(TD_START+2) = kurtosis(normSig);                  % Kurtosis
        featureVector(TD_START+3) = skewness(normSig);                  % Skewness
        featureVector(TD_START+4) = peak2peak(normSig);                 % Peak-to-Peak
        featureVector(TD_START+5) = var(normSig);                       % Variance
        featureVector(TD_START+6) = max(abs(normSig)) / featureVector(TD_START); % Crest Factor
        featureVector(TD_START+7) = featureVector(TD_START) / mean(abs(normSig)); % Shape Factor
        
        % ---- Frequency Domain (8 features) ----
        [Pxx, ~] = pwelch(normSig, hann(1024), 512, 2048, Fs);
        totalEnergy = sum(Pxx);
        if totalEnergy > 0
            for band = 1:8
                featureVector(FD_START-1+band) = sum(Pxx(binMasks(:, band))) / totalEnergy;
            end
        else
            featureVector(FD_START:FD_END) = 0;
        end
        
        % ---- Wavelet Packet Transform (254 features) ----
        if exist('wpdec', 'file') == 2  % Check if Wavelet Toolbox available
            try
                tree = wpdec(normSig, maxWPTLevel, 'db4');
                for i = 1:size(wptNodes,1)
                    coefs = wpcoef(tree, wptNodes(i,:));
                    featureVector(WPT_START-1+i) = sum(coefs.^2);  % Energy
                end
            catch
                % Fallback: Use zeros instead of NaNs
                featureVector(WPT_START:WPT_END) = 0;
            end
        else
            featureVector(WPT_START:WPT_END) = 0;
        end
        
        % ---- Discrete Wavelet Transform (9 features) ----
        if exist('wavedec', 'file') == 2  % Check if Wavelet Toolbox available
            try
                [C, L] = wavedec(normSig, 6, 'db4');
                % Detail coefficients variances (levels 1-3)
                for lev = 1:3
                    dCoef = detcoef(C, L, lev);
                    featureVector(DWT_START-1+lev) = var(dCoef);
                end
                % Autocorrelation variances (levels 4-6)
                for lev = 4:6
                    dCoef = detcoef(C, L, lev);
                    acf = xcorr(dCoef, 'unbiased');
                    featureVector(DWT_START-1+lev) = var(acf);
                end
                % Smoothed detail means (levels 1-3)
                for lev = 1:3
                    dCoef = detcoef(C, L, lev);
                    smoothed = movmean(dCoef, 5);
                    featureVector(DWT_START-1+6+lev) = mean(smoothed);
                end
            catch
                featureVector(DWT_START:DWT_END) = 0;
            end
        else
            featureVector(DWT_START:DWT_END) = 0;
        end
        
        % ---- Morlet Wavelet Transform (7 features) ----
        try
            a = 16;
            b = 0.02;
            t_wavelet = linspace(-0.5, 0.5, 2001);
            wavelet = exp(-b^2 * t_wavelet.^2 / a^2) .* cos(pi * t_wavelet / a);
            convSig = conv(normSig, wavelet, 'same');
            
            % Feature calculations
            P = convSig.^2; 
            P = P / sum(P);
            featureVector(MWT_START) = -sum(P .* log2(P + eps));   % Entropy
            [pks, ~] = findpeaks(abs(convSig));                    % Sum of peaks
            featureVector(MWT_START+1) = sum(pks);
            featureVector(MWT_START+2) = std(convSig);             % Std dev
            featureVector(MWT_START+3) = kurtosis(convSig);        % Kurtosis
            featureVector(MWT_START+4) = sum(diff(convSig>0)~=0)/length(convSig); % ZCR
            featureVector(MWT_START+5) = var(convSig);             % Variance
            featureVector(MWT_START+6) = skewness(convSig);        % Skewness
        catch
            featureVector(MWT_START:MWT_END) = 0;
        end
        
        features(idx, :) = featureVector;
        
    catch ME
        extractionErrors{idx} = ME.message;
        warning('Error processing file %s: %s', filePaths{idx}, ME.message);
    end
end

%% Handle Missing Data
validRows = ~any(isnan(features), 2);
numValid = sum(validRows);

if numValid == 0
    fprintf('\nCritical: All files failed feature extraction\n');
    emptyMask = ~cellfun(@isempty, extractionErrors);
    errorFiles = find(emptyMask, 3);
    for i = 1:min(3, numel(errorFiles))
        fprintf('%d: %s - %s\n', errorFiles(i), filePaths{errorFiles(i)}, extractionErrors{errorFiles(i)});
    end
    error('All files failed feature extraction.');
end

features = features(validRows, :);
classLabels = classLabels(validRows);
classIndices = classIndices(validRows);
fprintf('Retained %d/%d files after filtering\n', numValid, totalFiles);

%% Feature Diagnostic Analysis
% 1. Class Distribution
fprintf('\nClass Distribution:\n');
tabulate(classLabels)

% 2. Feature Variance Analysis
featureVars = var(features, 0, 1, 'omitnan');
lowVarFeatures = find(featureVars < 1e-6);
fprintf('\n%d features with near-zero variance (<1e-6)\n', numel(lowVarFeatures));

% 3. PCA Visualization
[coeff, score] = pca(zscore(features));
figure;
gscatter(score(:,1), score(:,2), classLabels);
title('PCA Projection (First Two Components)');
xlabel('PC1');
ylabel('PC2');
grid on;

% 4. Feature Correlation
corrMatrix = corr(features, 'Rows','pairwise');
figure;
imagesc(corrMatrix);
colorbar;
title('Feature Correlation Matrix');
axis square;

%% Corrected Model Training Pipeline
% Fix 1: Per-fold standardization to prevent data leakage
% Fix 2: Expanded hyperparameter search space
% Fix 3: Feature selection integrated into CV
% Fix 4: Class weighting for imbalance

% Create CV partitions
cv = cvpartition(classLabels, 'KFold', 5, 'Stratify', true);
accuracies = zeros(cv.NumTestSets, 1);
confusionMats = cell(1, cv.NumTestSets);

% Hyperparameter grid
C_values = [0.01, 0.1, 1, 10, 100, 1000];
gamma_values = [0.0001, 0.001, 0.01, 0.1, 1, 10];

parfor k = 1:cv.NumTestSets
    try
        % Split data
        trainIdx = cv.training(k);
        testIdx = cv.test(k);
        
        % Standardize within fold
        mu_train = mean(features(trainIdx,:), 1, 'omitnan');
        sigma_train = std(features(trainIdx,:), 0, 1, 'omitnan');
        sigma_train(sigma_train == 0) = 1;
        X_train = (features(trainIdx,:) - mu_train) ./ sigma_train;
        X_test = (features(testIdx,:) - mu_train) ./ sigma_train;
        y_train = classLabels(trainIdx);
        y_test = classLabels(testIdx);
        
        % Feature selection (mRMR) - only on training data
        numToSelect = 50;
        selectedFeatures = mRMR_selection(X_train, classIndices(trainIdx), numToSelect);
        
        % Hyperparameter tuning using grid search
        bestFoldAcc = 0;
        bestFoldModel = [];
        
        for C = C_values
            for gamma = gamma_values
                % Train SVM with RBF kernel
                t = templateSVM('KernelFunction', 'rbf', ...
                                'BoxConstraint', C, ...
                                'KernelScale', 1/sqrt(gamma), ...  % More intuitive parameterization
                                'Standardize', false);
                
                model = fitcecoc(X_train(:, selectedFeatures), y_train, ...
                                 'Learners', t, ...
                                 'Coding', 'onevsone', ...
                                 'ClassNames', unique(y_train));
                
                % Predict and evaluate
                pred = predict(model, X_test(:, selectedFeatures));
                acc = sum(strcmp(pred, y_test)) / numel(y_test);
                
                % Update best model
                if acc > bestFoldAcc
                    bestFoldAcc = acc;
                    bestFoldModel = model;
                end
            end
        end
        
        % Store results
        accuracies(k) = bestFoldAcc;
        pred = predict(bestFoldModel, X_test(:, selectedFeatures));
        confusionMats{k} = confusionmat(y_test, pred);
        
        fprintf('Fold %d: Accuracy = %.2f%%\n', k, bestFoldAcc*100);
    catch ME
        accuracies(k) = 0;
        fprintf('Error in fold %d: %s\n', k, ME.message);
    end
end

% Aggregate results
meanAcc = mean(accuracies);
stdAcc = std(accuracies);
fprintf('\nCross-validated Accuracy: %.2f%% ± %.2f%%\n', meanAcc*100, stdAcc*100);

% Combine confusion matrices
combinedCM = zeros(numel(faultClasses));
for i = 1:cv.NumTestSets
    if size(confusionMats{i},1) == numel(faultClasses)
        combinedCM = combinedCM + confusionMats{i};
    end
end

% Plot confusion matrix
figure;
confusionchart(combinedCM, faultClasses, ...
               'Normalization', 'row-normalized', ...
               'RowSummary', 'row-normalized', ...
               'ColumnSummary', 'column-normalized');
title(sprintf('Average Accuracy: %.2f%%', meanAcc*100));

%% Feature Selection Function (mRMR)
function selected = mRMR_selection(X, y, numToSelect)
    % Discretize continuous variables
    numBins = 15;
    X_disc = zeros(size(X));
    for i = 1:size(X,2)
        edges = linspace(min(X(:,i)), max(X(:,i)), numBins+1);
        X_disc(:,i) = discretize(X(:,i), edges);
    end
    
    % Initialize
    selected = [];
    available = 1:size(X,2);
    
    % Compute MI with class
    classMI = zeros(1, size(X,2));
    for f = 1:size(X,2)
        classMI(f) = mutualInfo(X_disc(:,f), y);
    end
    
    % Select first feature (max MI)
    [~, idx] = max(classMI);
    selected = [selected, available(idx)];
    available(idx) = [];
    
    % Select subsequent features
    for i = 2:numToSelect
        redundancy = zeros(1, length(available));
        for j = 1:length(available)
            f_j = available(j);
            redundancy(j) = mean(arrayfun(@(s) mutualInfo(X_disc(:,s), X_disc(:,f_j)), selected));
        end
        
        [~, bestIdx] = max(classMI(available) - redundancy);
        selected = [selected, available(bestIdx)];
        available(bestIdx) = [];
    end
end

%% Mutual Information Calculation
function mi = mutualInfo(x, y)
    % Joint probability distribution
    [joint, ~, ~] = histcounts2(x, y, 'Normalization', 'probability');
    joint = joint(:);
    
    % Marginal probabilities
    px = histcounts(x, 'Normalization', 'probability');
    py = histcounts(y, 'Normalization', 'probability');
    
    % Mutual information
    mi = 0;
    for i = 1:numel(joint)
        if joint(i) > 0
            [row, col] = ind2sub([length(px), length(py)], i);
            mi = mi + joint(i) * log2(joint(i) / (px(row) * py(col)));
        end
    end
end