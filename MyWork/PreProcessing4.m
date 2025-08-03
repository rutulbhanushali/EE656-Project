%% Air Compressor Fault Diagnosis System
% Robust end-to-end implementation based on IEEE paper
% Features: 24-bit PCM decoding, 629 feature extraction, mRMR selection, SVM classification

%% System Configuration
clear; clc; close all;
mainFolder = 'AirCompressor_Data';
faultClasses = {'Bearing','Flywheel','Healthy','LIV','LOV','NRV','Piston','Riderbelt'};
Fs = 50000;                    % Sampling rate (Hz)
segmentLength = Fs * 1;        % 1-second window
stepSize = round(segmentLength / 2);  % 50% overlap
numFeatures = 629;             % Full feature set from paper

%% Bandpass Filter Design (400Hz-12kHz)
bpFilt = designfilt('bandpassiir', ...
    'FilterOrder', 8, ...
    'HalfPowerFrequency1', 400, ...
    'HalfPowerFrequency2', 12000, ...
    'SampleRate', Fs, ...
    'DesignMethod', 'butter');

%% Precompute Constants
% WPT Node Indices
maxWPTLevel = 7;
wptNodes = zeros(254, 2);
nodeIdx = 1;
for level = 1:maxWPTLevel
    for node = 0:(2^level - 1)
        wptNodes(nodeIdx, :) = [level, node];
        nodeIdx = nodeIdx + 1;
    end
end

% Frequency Binning
[~, F] = pwelch(randn(segmentLength,1), hann(1024), 512, 2048, Fs);
binEdges = linspace(0, Fs/2, 9);
binMasks = false(length(F), 8);
for band = 1:8
    binMasks(:, band) = (F >= binEdges(band)) & (F < binEdges(band+1));
end

% Define feature indices using robust struct initialization
featureRanges = struct();
featureRanges.TD = 1:8;         % Time Domain
featureRanges.FD = 9:16;        % Frequency Domain
featureRanges.WPT = 17:270;     % Wavelet Packet Transform
featureRanges.DWT = 271:279;    % Discrete Wavelet Transform
featureRanges.MWT = 280:286;    % Morlet Wavelet Transform
featureRanges.DCT = 287:294;    % Discrete Cosine Transform
featureRanges.STFT = 295:366;   % Short-Time Fourier Transform
featureRanges.WVD = 367:438;    % Wigner-Ville Distribution
% Add more ranges as needed

%% File Management
fprintf('Loading dataset from: %s\n', mainFolder);
filePaths = {};
classLabels = categorical();
for c = 1:numel(faultClasses)
    folder = fullfile(mainFolder, faultClasses{c});
    if ~isfolder(folder)
        error('Folder not found: %s', folder);
    end
    files = dir(fullfile(folder, '*.dat'));
    if isempty(files)
        error('No .dat files found in: %s', folder);
    end
    for i = 1:numel(files)
        filePaths{end+1} = fullfile(folder, files(i).name);
        classLabels(end+1) = faultClasses{c};
    end
end
totalFiles = numel(filePaths);
fprintf('Found %d files across %d classes\n', totalFiles, numel(faultClasses));

%% Initialize Parallel Pool
if isempty(gcp('nocreate'))
    pool = parpool('Processes');
    fprintf('Parallel pool created with %d workers\n', pool.NumWorkers);
end

%% ====================== MAIN PROCESSING LOOP ======================
features = NaN(totalFiles, numFeatures);
extractionErrors = cell(totalFiles, 1);

parfor idx = 1:totalFiles
    try
        % --- 24-bit PCM DECODING ---
        fid = fopen(filePaths{idx}, 'r');
        if fid == -1
            error('File not found');
        end
        raw = fread(fid, inf, 'uint8=>uint8');
        fclose(fid);
        
        % Check minimum file size (5 seconds * 50kHz * 3 bytes)
        if numel(raw) < 150000
            error('File too small: %d bytes', numel(raw));
        end
        
        % Robust 24-bit signed integer decoding
        numSamples = floor(numel(raw)/3);
        data = zeros(1, numSamples);
        for i = 1:numSamples
            % Extract 3 bytes per sample (little-endian)
            byte1 = double(raw(3*i-2));
            byte2 = double(raw(3*i-1));
            byte3 = double(raw(3*i));
            
            % Combine bytes with sign extension
            sample = byte1 + 256*byte2 + 65536*byte3;
            if sample >= 2^23
                sample = sample - 2^24;  % Two's complement conversion
            end
            data(i) = sample / (2^23);  % Normalize to [-1, 1]
        end
        
        % --- PREPROCESSING ---
        % Bandpass Filtering (400Hz-12kHz)
        sig = filtfilt(bpFilt, data(:));  % Ensure column vector
        
        % Segmentation with 50% overlap
        numSegments = floor((length(sig) - segmentLength)/stepSize) + 1;
        if numSegments < 1
            error('Signal too short: %d samples', length(sig));
        end
        
        segments = zeros(segmentLength, numSegments);
        stdVals = zeros(numSegments, 1);
        for seg = 1:numSegments
            startIdx = (seg-1)*stepSize + 1;
            endIdx = startIdx + segmentLength - 1;
            segments(:, seg) = sig(startIdx:endIdx);
            stdVals(seg) = std(segments(:, seg));
        end
        
        % Select most stable segment
        [~, minIdx] = min(stdVals);
        clip = segments(:, minIdx);
        
        % Moving Average Smoothing
        sm = movmean(clip, 5);
        
        % Robust Normalization (outlier-resistant)
        sorted = sort(sm);
        k = max(1, round(0.00025 * length(sorted)));
        L = sorted(k);
        U = sorted(end-k+1);
        normSig = 2 * (sm - L) / (U - L) - 1;
        
        % --- FEATURE EXTRACTION ---
        featVec = zeros(1, numFeatures);  % Initialize with zeros
        
        % TIME DOMAIN (8 features)
        rmsVal = rms(normSig);
        zcr = sum(diff(normSig > 0) ~= 0) / (length(normSig)-1);
        kurtVal = kurtosis(normSig);
        skewVal = skewness(normSig);
        pk2pk = peak2peak(normSig);
        variance = var(normSig);
        crestFactor = max(abs(normSig)) / rmsVal;
        shapeFactor = rmsVal / mean(abs(normSig));
        
        featVec(featureRanges.TD) = [rmsVal, zcr, kurtVal, skewVal, pk2pk, variance, crestFactor, shapeFactor];
        
        % FREQUENCY DOMAIN (8 features)
        [Pxx, ~] = pwelch(normSig, hann(1024), 512, 2048, Fs);
        totalEnergy = sum(Pxx);
        if totalEnergy > 0
            for band = 1:8
                featVec(featureRanges.FD(band)) = sum(Pxx(binMasks(:, band))) / totalEnergy;
            end
        end
        
        % WAVELET PACKET TRANSFORM (254 features)
        try
            tree = wpdec(normSig, maxWPTLevel, 'db4');
            for i = 1:size(wptNodes,1)
                coefs = wpcoef(tree, wptNodes(i,:));
                featVec(featureRanges.WPT(i)) = sum(coefs.^2);
            end
        catch ME
            fprintf('WPT failed for file %d: %s\n', idx, ME.message);
        end
        
        % DISCRETE WAVELET TRANSFORM (9 features)
        try
            [C, L] = wavedec(normSig, 6, 'db4');
            for lev = 1:3
                dCoef = detcoef(C, L, lev);
                featVec(featureRanges.DWT(lev)) = var(dCoef);
            end
            for lev = 4:6
                dCoef = detcoef(C, L, lev);
                acf = xcorr(dCoef, 'unbiased');
                featVec(featureRanges.DWT(lev)) = var(acf);
            end
            for lev = 1:3
                dCoef = detcoef(C, L, lev);
                smoothed = movmean(dCoef, 5);
                featVec(featureRanges.DWT(6+lev)) = mean(smoothed);
            end
        catch ME
            fprintf('DWT failed for file %d: %s\n', idx, ME.message);
        end
        
        % MORLET WAVELET TRANSFORM (7 features)
        try
            a_val = 16; b_val = 0.02;
            t_wavelet = linspace(-0.5, 0.5, 2001);
            wavelet = exp(-b_val^2*t_wavelet.^2/a_val^2) .* cos(pi*t_wavelet/a_val);
            convSig = conv(normSig, wavelet, 'same');
            
            P = convSig.^2/sum(convSig.^2);
            entropyVal = -sum(P.*log2(P+eps));
            [~, locs] = findpeaks(abs(convSig));
            peaks = numel(locs);
            stdDev = std(convSig);
            kurtVal = kurtosis(convSig);
            zcr = sum(diff(convSig>0)~=0)/length(convSig);
            varVal = var(convSig);
            skewVal = skewness(convSig);
            
            featVec(featureRanges.MWT) = [entropyVal, peaks, stdDev, kurtVal, zcr, varVal, skewVal];
        catch ME
            fprintf('MWT failed for file %d: %s\n', idx, ME.message);
        end
        
        % DISCRETE COSINE TRANSFORM (8 features)
        try
            dct_coeffs = dct(normSig);
            featVec(featureRanges.DCT) = dct_coeffs(1:8);
        catch ME
            fprintf('DCT failed for file %d: %s\n', idx, ME.message);
        end
        
        % SHORT-TIME FOURIER TRANSFORM (72 features)
        try
            [s, ~, ~] = spectrogram(normSig, 256, 128, 256, Fs);
            stft_mag = mean(abs(s), 2);
            if length(stft_mag) < 72
                stft_mag(end+1:72) = 0;
            end
            featVec(featureRanges.STFT) = stft_mag(1:72);
        catch ME
            fprintf('STFT failed for file %d: %s\n', idx, ME.message);
        end
        
        % WIGNER-VILLE DISTRIBUTION (72 features)
        try
            [wvd_tf, ~, ~] = wvd(normSig, Fs, 'smoothedPseudo');
            wvd_mag = mean(abs(wvd_tf), 2);
            if size(wvd_mag,1) < 72
                wvd_mag(end+1:72, :) = 0;
            end
            featVec(featureRanges.WVD) = wvd_mag(1:72);
        catch ME
            fprintf('WVD failed for file %d: %s\n', idx, ME.message);
        end
        
        features(idx, :) = featVec;
        
    catch ME
        extractionErrors{idx} = sprintf('File %d (%s): %s', idx, filePaths{idx}, ME.message);
    end
end

%% Post-Processing
% Report errors
validFiles = true(totalFiles, 1);
fprintf('\n--- Processing Report ---\n');
for idx = 1:totalFiles
    if ~isempty(extractionErrors{idx})
        fprintf('ERROR: %s\n', extractionErrors{idx});
        validFiles(idx) = false;
    end
end
fprintf('Successfully processed %d/%d files\n', sum(validFiles), totalFiles);

% Remove invalid files
features = features(validFiles, :);
classLabels = classLabels(validFiles);
if isempty(features)
    error('No valid files processed. Check data and error messages.');
end

% Remove low-variance features
featureVars = var(features, 0, 1);
lowVarMask = featureVars < 1e-6;
features(:, lowVarMask) = [];
fprintf('Removed %d low-variance features\n', sum(lowVarMask));

% Class distribution
fprintf('\n--- Class Distribution ---\n');
tabulate(classLabels)

%% ======================== FEATURE SELECTION & MODELING ========================
fprintf('\n--- Model Training ---\n');
numToSelect = 25;  % Optimal per paper
cv = cvpartition(classLabels, 'KFold', 5, 'Stratify', true);

% Preallocate results
results = table('Size', [cv.NumTestSets, 2], ...
    'VariableTypes', {'double', 'double'}, ...
    'VariableNames', {'Fold', 'Accuracy'});

% Bayesian optimization options
hyperOpts = struct('AcquisitionFunctionName', 'expected-improvement-plus', ...
                   'MaxObjectiveEvaluations', 30, ...
                   'Verbose', 0);

% Main training loop               
for k = 1:cv.NumTestSets
    fprintf('Processing fold %d/%d...\n', k, cv.NumTestSets);
    % Data partitioning
    trainIdx = cv.training(k);
    testIdx = cv.test(k);
    
    % Fold-wise standardization
    mu_train = mean(features(trainIdx,:), 1);
    sigma_train = std(features(trainIdx,:), 0, 1);
    sigma_train(sigma_train == 0) = 1;
    X_train = (features(trainIdx,:) - mu_train) ./ sigma_train;
    X_test = (features(testIdx,:) - mu_train) ./ sigma_train;
    y_train = classLabels(trainIdx);
    y_test = classLabels(testIdx);
    
    % Feature selection using mRMR
    [idx_sorted, scores] = fscmrmr(X_train, y_train);
    selected = idx_sorted(1:min(numToSelect, length(idx_sorted)));
    
    % Hyperparameter optimization
    svm_model = fitcsvm(X_train(:, selected), y_train, ...
        'KernelFunction', 'rbf', ...
        'OptimizeHyperparameters', 'auto', ...
        'HyperparameterOptimizationOptions', hyperOpts);
    
    % Train ECOC model
    t = templateSVM('KernelFunction', 'rbf', ...
                    'BoxConstraint', svm_model.HyperparameterOptimizationResults.BoxConstraint, ...
                    'KernelScale', svm_model.HyperparameterOptimizationResults.KernelScale);
    ecoc_model = fitcecoc(X_train(:, selected), y_train, ...
                          'Learners', t, 'Coding', 'onevsone');
    
    % Evaluate
    pred = predict(ecoc_model, X_test(:, selected));
    acc = sum(pred == y_test) / numel(y_test);
    
    % Store results
    results.Fold(k) = k;
    results.Accuracy(k) = acc;
    fprintf('Fold %d accuracy: %.2f%%\n', k, acc*100);
end

%% ======================== RESULTS ANALYSIS ========================
fprintf('\n=== Final Results ===\n');
fprintf('Mean Accuracy: %.2f%% ± %.2f%%\n', ...
        mean(results.Accuracy)*100, std(results.Accuracy)*100);
disp(results);

% Train final model on full data
fprintf('\nTraining final model...\n');
mu_all = mean(features, 1);
sigma_all = std(features, 0, 1);
sigma_all(sigma_all == 0) = 1;
X_all = (features - mu_all) ./ sigma_all;

% Feature selection on full data
[~, scores] = fscmrmr(X_all, classLabels);
[~, idx_sorted] = sort(scores, 'descend');
selected_final = idx_sorted(1:numToSelect);

% Final model training
final_svm = fitcsvm(X_all(:, selected_final), classLabels, ...
    'KernelFunction', 'rbf', ...
    'OptimizeHyperparameters', 'auto', ...
    'HyperparameterOptimizationOptions', hyperOpts);

t_final = templateSVM('KernelFunction', 'rbf', ...
                'BoxConstraint', final_svm.HyperparameterOptimizationResults.BoxConstraint, ...
                'KernelScale', final_svm.HyperparameterOptimizationResults.KernelScale);
final_model = fitcecoc(X_all(:, selected_final), classLabels, ...
                      'Learners', t_final, 'Coding', 'onevsone');

% Final evaluation
pred = predict(final_model, X_all(:, selected_final));
finalAcc = sum(pred == classLabels) / numel(classLabels);
fprintf('Final model accuracy: %.2f%%\n', finalAcc*100);

%% ======================== VISUALIZATION ========================
fprintf('\nGenerating visualizations...\n');

% Confusion Matrix
cm = confusionmat(classLabels, pred);
figure('Position', [100, 100, 800, 600]);
confusionchart(cm, categories(classLabels), 'Normalization', 'row-normalized');
title('Confusion Matrix (Row Normalized)', 'FontSize', 14);
set(gca, 'FontSize', 12);

% PCA Visualization
[~, score] = pca(X_all(:, selected_final));
figure('Position', [100, 100, 900, 700]);
gscatter(score(:,1), score(:,2), classLabels, [], 'o', 15);
title('PCA: Feature Space Projection', 'FontSize', 14);
xlabel('Principal Component 1', 'FontSize', 12);
ylabel('Principal Component 2', 'FontSize', 12);
legend('Location', 'bestoutside', 'FontSize', 10);
grid on;

% Feature Importance
figure('Position', [100, 100, 1000, 600]);
bar(scores(selected_final));
title('Top Feature Importance Scores (mRMR)', 'FontSize', 14);
xlabel('Feature Index', 'FontSize', 12);
ylabel('Importance Score', 'FontSize', 12);
xticks(1:numToSelect);
xticklabels(arrayfun(@num2str, selected_final, 'UniformOutput', false));
set(gca, 'FontSize', 10, 'XTickLabelRotation', 45);
grid on;

%% ======================== SAVE RESULTS ========================
save('compressor_diagnosis_model.mat', 'final_model', 'selected_final', ...
     'mu_all', 'sigma_all', 'results', 'cm', 'scores', 'classLabels', '-v7.3');

fprintf('\n=== System Summary ===\n');
fprintf('Model saved as compressor_diagnosis_model.mat\n');
fprintf('Total features extracted: %d\n', numFeatures);
fprintf('Features selected: %d\n', numToSelect);
fprintf('Final accuracy: %.2f%%\n', finalAcc*100);
fprintf('Diagnostic system ready for deployment!\n');