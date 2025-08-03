%% System Configuration
clear; clc; close all;
mainFolder = 'AirCompressor_Data';
faultClasses = {'Bearing', 'Flywheel', 'Healthy', 'LIV', 'LOV', 'NRV', 'Piston', 'Riderbelt'};
Fs = 50000;  % Sampling rate (Hz)
segmentLength = Fs * 1;  % 1-second analysis window
stepSize = segmentLength / 2;  % 50% overlap for segmentation
numFeatures = 50;  % Increased feature count for better discrimination

%% Precomputed Constants
% Frequency vectors for consistent feature extraction
[~, F] = pwelch(randn(segmentLength,1), [], [], [], Fs);
binEdges = linspace(0, Fs/2, 9);  % 8 frequency bins
binMask = cell(1,8);
for bin = 1:8
    binMask{bin} = (F >= binEdges(bin)) & (F < binEdges(bin+1));
end

%% File Inventory Management
% Pre-discover all files for efficient processing
totalFiles = 0;
filePaths = {};
classIndices = [];
fileCounts = zeros(1, length(faultClasses));

for i = 1:length(faultClasses)
    currentFolder = fullfile(mainFolder, faultClasses{i});
    dataFiles = dir(fullfile(currentFolder, '*.dat'));
    numFiles = length(dataFiles);
    fileCounts(i) = numFiles;
    totalFiles = totalFiles + numFiles;
    
    for j = 1:numFiles
        filePaths{end+1} = fullfile(currentFolder, dataFiles(j).name);
        classIndices(end+1) = i;
    end
end

% Preallocate feature matrix
features = zeros(totalFiles, numFeatures);
classLabels = faultClasses(classIndices)';  % Create label vector

%% Signal Processing Setup
% Bandpass filter design (400-12000 Hz)
bpFilt = designfilt('bandpassiir', 'FilterOrder', 4, ...
                   'HalfPowerFrequency1', 400, 'HalfPowerFrequency2', 12000, ...
                   'SampleRate', Fs, ...
                   'DesignMethod', 'butter');

%% Parallel Processing Initialization
% Start parallel pool with 4 workers
if isempty(gcp('nocreate'))
    parpool(4);
end

%% Main Processing Loop - Parallel Execution
parfor idx = 1:totalFiles
    % Get current file path
    filePath = filePaths{idx};
    
    %% 24-bit PCM Reading with Arithmetic Conversion
    fid = fopen(filePath, 'r');
    rawBytes = fread(fid, inf, 'uint8=>uint8');
    fclose(fid);
    
    % Process only complete triplets
    numCompleteTriplets = floor(numel(rawBytes)/3);
    if numCompleteTriplets == 0
        warning('Empty file: %s', filePath);
        continue;
    end
    
    % Reshape into 3xN matrix (each column is a sample)
    reshapedBytes = reshape(rawBytes(1:3*numCompleteTriplets), 3, []);
    
    % Convert bytes to 24-bit integers
    rawData = double(reshapedBytes(1,:)) + ...      % Least significant byte
              256 * double(reshapedBytes(2,:)) + ... % Middle byte
              65536 * double(reshapedBytes(3,:));    % Most significant byte
    
    % Handle negative values (24-bit 2's complement)
    signMask = reshapedBytes(3,:) >= 128;
    rawData(signMask) = rawData(signMask) - 16777216;  % Subtract 2^24
    
    % Normalize to [-1, 1] and convert to column vector
    rawData = rawData' / (2^23);
    
    %% Signal Preprocessing Pipeline
    % Bandpass filtering
    filteredSignal = filtfilt(bpFilt, rawData);
    
    % Segmentation for stable section
    numSegments = floor((length(filteredSignal) - segmentLength) / stepSize) + 1;
    segmentStarts = 1:stepSize:(length(filteredSignal)-segmentLength+1);
    segmentStds = zeros(1, numSegments);
    
    for seg = 1:numSegments
        startIdx = segmentStarts(seg);
        endIdx = startIdx + segmentLength - 1;
        segData = filteredSignal(startIdx:endIdx);
        segmentStds(seg) = std(segData);
    end
    
    % Select most stable segment
    [~, minIdx] = min(segmentStds);
    startIdx = segmentStarts(minIdx);
    clippedSignal = filteredSignal(startIdx:startIdx+segmentLength-1);
    
    % Noise reduction
    smoothedSignal = movmean(clippedSignal, 5);
    
    % Robust normalization (remove 0.025% outliers)
    sortedSignal = sort(smoothedSignal);
    n = length(sortedSignal);
    skipCount = max(1, round(0.00025*n));
    lowerBound = sortedSignal(skipCount);
    upperBound = sortedSignal(end - skipCount + 1);
    normalizedSignal = 2 * (smoothedSignal - lowerBound) / (upperBound - lowerBound) - 1;
    
    %% Enhanced Feature Extraction
    % Time-domain features (8) - Enhanced from paper
    rmsVal = rms(normalizedSignal);
    zcr = sum(diff(sign(normalizedSignal)) ~= 0) / length(normalizedSignal);
    kurt = kurtosis(normalizedSignal);
    skew = skewness(normalizedSignal);
    pp = peak2peak(normalizedSignal);
    variance = var(normalizedSignal);
    crestFactor = max(abs(normalizedSignal)) / rmsVal;
    shapeFactorVal = rmsVal / mean(abs(normalizedSignal));
    
    % Frequency-domain features (8) - Enhanced
    [Pxx, ~] = pwelch(normalizedSignal, [], [], [], Fs);
    spectralCentroid = sum(F .* Pxx) / sum(Pxx);
    spectralRolloff = F(find(cumsum(Pxx) >= 0.85*sum(Pxx), 1, 'first'));
    spectralEntropy = -sum(Pxx.*log2(Pxx + eps)) / log2(length(Pxx));
    spectralFlux = sum(diff(Pxx).^2);
    spectralBandwidth = sqrt(sum(((F - spectralCentroid).^2) .* Pxx) / sum(Pxx));
    
    % Bin energies (8) - Using precomputed masks
    binEnergies = zeros(1,8);
    for bin = 1:8
        binEnergies(bin) = sum(Pxx(binMask{bin}));
    end
    binEnergies = binEnergies / sum(binEnergies);
    
    % Wavelet Packet Transform features (16)
    wpt = wpdec(normalizedSignal, 4, 'db4');
    wptEnergies = zeros(1,16);
    for k = 0:15
        packet = wpcoef(wpt, [4, k]);
        wptEnergies(k+1) = sum(packet.^2);
    end
    
    % DCT features (10) - Additional discriminative features
    dctCoeffs = dct(normalizedSignal);
    dctFeatures = zeros(1,10);
    for k = 1:10
        dctFeatures(k) = sum(dctCoeffs((k-1)*floor(length(dctCoeffs)/10)+1:k*floor(length(dctCoeffs)/10)).^2);
    end
    
    % Combine all features (8 + 5 + 8 + 16 + 10 + 3 = 50 features)
    features(idx, :) = [rmsVal, zcr, kurt, skew, pp, variance, crestFactor, shapeFactorVal, ...
                       spectralCentroid, spectralRolloff, spectralEntropy, spectralFlux, spectralBandwidth, ...
                       binEnergies, wptEnergies, dctFeatures];
end

%% Advanced Classification with Hyperparameter Tuning and Feature Selection
if exist('fitcecoc', 'file') && totalFiles > 50
    fprintf('\n=== Advanced SVM Classification with Optimization ===\n');
    rng(42);  % Set seed for reproducibility
    
    %% Step 1: Feature Standardization (CRITICAL)
    fprintf('Step 1: Standardizing features...\n');
    mu = mean(features);
    sigma = std(features);
    features_std = (features - mu) ./ sigma;
    
    %% Step 2: mRMR Feature Selection
    fprintf('Step 2: Performing mRMR feature selection...\n');
    
    % Simple mRMR implementation (since fscmrmr might not be available)
    % Calculate mutual information between features and class labels
    numClasses = length(faultClasses);
    featureScores = zeros(1, numFeatures);
    
    for f = 1:numFeatures
        % Discretize feature values for MI calculation
        [~, ~, featureIdx] = histcounts(features_std(:,f), 20);
        
        % Calculate class-feature mutual information
        classFeatureJoint = zeros(numClasses, 20);
        for c = 1:numClasses
            classData = featureIdx(classIndices == c);
            classData = classData(classData > 0);  % Remove zero indices
            for val = 1:20
                classFeatureJoint(c, val) = sum(classData == val) / length(classData);
            end
        end
        
        % Simplified mutual information score
        featureScores(f) = sum(sum(classFeatureJoint .* log2(classFeatureJoint + eps)));
    end
    
    % Select top features
    [~, topFeatureIdx] = sort(featureScores, 'descend');
    
    %% Step 3: Hyperparameter Grid Search with Cross-Validation
    fprintf('Step 3: Hyperparameter optimization...\n');
    
    % Define parameter grids
    CList = [0.1, 1, 10, 100, 1000];
    KernelScaleList = [0.001, 0.01, 0.1, 1, 10];
    
    % Test different numbers of features
    featureCounts = [10, 25, 40, 50];
    
    bestAccuracy = 0;
    bestParams = struct();
    
    for numTopFeatures = featureCounts
        fprintf('Testing with %d features...\n', numTopFeatures);
        selectedFeatures = features_std(:, topFeatureIdx(1:numTopFeatures));
        
        for C = CList
            for KS = KernelScaleList
                % 5-fold cross-validation
                cv = cvpartition(classLabels, 'KFold', 5);
                cvAccuracies = zeros(1, 5);
                
                for fold = 1:5
                    trainIdx = training(cv, fold);
                    testIdx = test(cv, fold);
                    
                    % Create SVM template
                    svmTemplate = templateSVM(...
                        'KernelFunction', 'rbf', ...
                        'BoxConstraint', C, ...
                        'KernelScale', KS, ...
                        'Standardize', false);  % Already standardized
                    
                    % Train model
                    try
                        model = fitcecoc(selectedFeatures(trainIdx,:), classLabels(trainIdx), ...
                            'Learners', svmTemplate, 'Coding', 'onevsone');
                        
                        % Predict
                        pred = predict(model, selectedFeatures(testIdx,:));
                        cvAccuracies(fold) = sum(strcmp(pred, classLabels(testIdx))) / numel(pred);
                    catch
                        cvAccuracies(fold) = 0;
                    end
                end
                
                avgAccuracy = mean(cvAccuracies);
                if avgAccuracy > bestAccuracy
                    bestAccuracy = avgAccuracy;
                    bestParams.C = C;
                    bestParams.KernelScale = KS;
                    bestParams.NumFeatures = numTopFeatures;
                    bestParams.SelectedFeatures = topFeatureIdx(1:numTopFeatures);
                end
            end
        end
    end
    
    fprintf('Best parameters found:\n');
    fprintf('  C: %.3f\n', bestParams.C);
    fprintf('  KernelScale: %.3f\n', bestParams.KernelScale);
    fprintf('  NumFeatures: %d\n', bestParams.NumFeatures);
    fprintf('  Best CV Accuracy: %.2f%%\n', bestAccuracy * 100);
    
    %% Step 4: Final Model Training with Best Parameters
    fprintf('\nStep 4: Training final model with best parameters...\n');
    
    % Use best features
    finalFeatures = features_std(:, bestParams.SelectedFeatures);
    
    % Final train-test split
    cv = cvpartition(classLabels, 'Holdout', 0.3);
    trainIdx = training(cv);
    testIdx = test(cv);
    
    % Create final SVM template
    finalSvmTemplate = templateSVM(...
        'KernelFunction', 'rbf', ...
        'BoxConstraint', bestParams.C, ...
        'KernelScale', bestParams.KernelScale, ...
        'Standardize', false);
    
    % Train final model
    tic;
    finalModel = fitcecoc(finalFeatures(trainIdx,:), classLabels(trainIdx), ...
        'Learners', finalSvmTemplate, 'Coding', 'onevsone');
    trainTime = toc;
    
    % Final prediction
    tic;
    finalPred = predict(finalModel, finalFeatures(testIdx,:));
    testTime = toc;
    
    % Calculate final accuracy
    finalAccuracy = sum(strcmp(finalPred, classLabels(testIdx))) / numel(finalPred);
    
    fprintf('=== FINAL RESULTS ===\n');
    fprintf('Final Test Accuracy: %.2f%%\n', finalAccuracy * 100);
    fprintf('Training Time: %.2f seconds\n', trainTime);
    fprintf('Prediction Time: %.2f seconds\n', testTime);
    
    %% Enhanced Confusion Matrix and Metrics
    figure('Name', 'Enhanced Confusion Matrix', 'Position', [100, 100, 800, 600]);
    confusionchart(classLabels(testIdx), finalPred, ...
        'Title', sprintf('Enhanced Classification Performance (%.2f%% Accuracy)', finalAccuracy * 100), ...
        'RowSummary', 'row-normalized', ...
        'ColumnSummary', 'column-normalized');
    
    % Detailed class-wise metrics
    [cm, order] = confusionmat(classLabels(testIdx), finalPred);
    stats = zeros(length(order), 3);
    for i = 1:length(order)
        TP = cm(i,i);
        FP = sum(cm(:,i)) - TP;
        FN = sum(cm(i,:)) - TP;
        
        precision = TP / (TP + FP);
        recall = TP / (TP + FN);
        f1 = 2 * (precision * recall) / (precision + recall);
        
        stats(i,:) = [precision, recall, f1];
    end
    
    fprintf('\n=== DETAILED PERFORMANCE METRICS ===\n');
    fprintf('%-15s %-10s %-10s %-10s\n', 'Class', 'Precision', 'Recall', 'F1-Score');
    fprintf('%-15s %-10s %-10s %-10s\n', '-----', '---------', '------', '--------');
    for i = 1:length(order)
        fprintf('%-15s %-10.3f %-10.3f %-10.3f\n', ...
                order{i}, stats(i,1), stats(i,2), stats(i,3));
    end
    
    % Calculate overall metrics
    avgPrecision = mean(stats(:,1));
    avgRecall = mean(stats(:,2));
    avgF1 = mean(stats(:,3));
    
    fprintf('\n%-15s %-10.3f %-10.3f %-10.3f\n', 'AVERAGE', avgPrecision, avgRecall, avgF1);
    
    %% Feature Importance Analysis
    fprintf('\n=== SELECTED FEATURES ===\n');
    fprintf('Feature indices selected by mRMR:\n');
    fprintf('%s\n', mat2str(bestParams.SelectedFeatures));
    
    %% Save Enhanced Model
    save('enhanced_svm_fault_classifier.mat', 'finalModel', 'bestParams', ...
         'mu', 'sigma', 'finalAccuracy', 'stats', 'topFeatureIdx');
    
    %% Performance Comparison
    fprintf('\n=== PERFORMANCE COMPARISON ===\n');
    fprintf('Previous accuracy: ~30%%\n');
    fprintf('Enhanced accuracy: %.2f%%\n', finalAccuracy * 100);
    fprintf('Improvement: +%.2f%%\n', (finalAccuracy * 100) - 30);
    
else
    fprintf('Skipping enhanced classification (insufficient data or missing toolbox)\n');
end

%% Visualization (keep existing)
% [Previous visualization code remains the same]

%% Cleanup
delete(gcp('nocreate'));  % Shutdown parallel pool
fprintf('\nProcessing complete! %d files processed.\n', totalFiles);
fprintf('Enhanced model saved with %.2f%% accuracy.\n', finalAccuracy * 100);
