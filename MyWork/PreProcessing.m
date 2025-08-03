%% System Configuration
clear; clc; close all;
mainFolder = 'AirCompressor_Data';
faultClasses = {'Bearing', 'Flywheel', 'Healthy', 'LIV', 'LOV', 'NRV', 'Piston', 'Riderbelt'};
Fs = 50000;  % Sampling rate (Hz)
segmentLength = Fs * 1;  % 1-second analysis window
stepSize = segmentLength / 2;  % 50% overlap for segmentation
numFeatures = 32;  % Feature count: 5 TD + 3 FD + 8 bins + 16 WPT

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
    
    %% Feature Extraction
    % Time-domain features (5)
    rmsVal = rms(normalizedSignal);
    zcr = sum(diff(sign(normalizedSignal)) ~= 0) / length(normalizedSignal);
    kurt = kurtosis(normalizedSignal);
    skew = skewness(normalizedSignal);
    pp = peak2peak(normalizedSignal);
    
    % Frequency-domain features (3)
    [Pxx, ~] = pwelch(normalizedSignal, [], [], [], Fs);
    spectralCentroid = sum(F .* Pxx) / sum(Pxx);
    spectralRolloff = F(find(cumsum(Pxx) >= 0.85*sum(Pxx), 1, 'first'));
    spectralEntropy = -sum(Pxx.*log2(Pxx + eps)) / log2(length(Pxx));
    
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
    
    % Combine all features
    features(idx, :) = [rmsVal, zcr, kurt, skew, pp, ...
                       spectralCentroid, spectralRolloff, spectralEntropy, ...
                       binEnergies, wptEnergies];
end

%% Get Example Signals for Visualization
% Process first file of each class for representative signals
exampleSignals = cell(length(faultClasses), 1);
t = (0:segmentLength-1)/Fs;  % Time vector

for i = 1:length(faultClasses)
    firstFileIdx = find(classIndices == i, 1, 'first');
    if ~isempty(firstFileIdx)
        filePath = filePaths{firstFileIdx};
        
        % Read and process file (same as parallel loop)
        fid = fopen(filePath, 'r');
        rawBytes = fread(fid, inf, 'uint8=>uint8');
        fclose(fid);
        
        numCompleteTriplets = floor(numel(rawBytes)/3);
        reshapedBytes = reshape(rawBytes(1:3*numCompleteTriplets), 3, []);
        rawData = double(reshapedBytes(1,:)) + 256*double(reshapedBytes(2,:)) + 65536*double(reshapedBytes(3,:));
        signMask = reshapedBytes(3,:) >= 128;
        rawData(signMask) = rawData(signMask) - 16777216;
        rawData = rawData' / (2^23);
        
        filteredSignal = filtfilt(bpFilt, rawData);
        numSegments = floor((length(filteredSignal) - segmentLength) / stepSize) + 1;
        segmentStarts = 1:stepSize:(length(filteredSignal)-segmentLength+1);
        segmentStds = zeros(1, numSegments);
        for seg = 1:numSegments
            segData = filteredSignal(segmentStarts(seg):segmentStarts(seg)+segmentLength-1);
            segmentStds(seg) = std(segData);
        end
        [~, minIdx] = min(segmentStds);
        clippedSignal = filteredSignal(segmentStarts(minIdx):segmentStarts(minIdx)+segmentLength-1);
        smoothedSignal = movmean(clippedSignal, 5);
        sortedSignal = sort(smoothedSignal);
        n = length(sortedSignal);
        skipCount = max(1, round(0.00025*n));
        lowerBound = sortedSignal(skipCount);
        upperBound = sortedSignal(end - skipCount + 1);
        exampleSignals{i} = 2 * (smoothedSignal - lowerBound) / (upperBound - lowerBound) - 1;
    end
end

%% Visualization Module
% Time-domain waveforms
figure('Position', [100, 100, 1200, 800], 'Name', 'Time Domain Signals');
for i = 1:min(length(faultClasses), 8)
    if ~isempty(exampleSignals{i})
        subplot(3, 3, i);
        plot(t, exampleSignals{i});
        title([faultClasses{i} ' - Time Domain']);
        xlabel('Time (s)'); ylabel('Amplitude');
        xlim([0, 0.1]);  % First 100ms
        grid on;
    end
end

% Spectrograms
figure('Position', [100, 100, 1200, 800], 'Name', 'Spectrograms');
for i = 1:min(length(faultClasses), 8)
    if ~isempty(exampleSignals{i})
        subplot(3, 3, i);
        spectrogram(exampleSignals{i}, 1024, 512, 1024, Fs, 'yaxis');
        title([faultClasses{i} ' - Spectrogram']);
        clim([-80, 0]);
    end
end

% Feature distribution
figure('Name', 'Feature Distribution', 'Position', [100, 100, 800, 600]);
boxplot(features(:,1:5), 'Labels', {'RMS', 'ZCR', 'Kurtosis', 'Skewness', 'Peak2Peak'});
title('Time-Domain Feature Distribution');
grid on;

%% Data Export for Python Integration
% Features and labels
writematrix(features, 'compressor_features.csv');
writetable(table(classLabels), 'compressor_labels.csv');

% Example signals and metadata
save('example_signals.mat', 'exampleSignals', 'faultClasses', 'Fs', 't');

%% SVM Classification (Corrected Implementation)
if exist('fitcecoc', 'file') && totalFiles > 50
    fprintf('\n--- Starting SVM Classification ---\n');
    rng(42);  % Set seed for reproducibility
    cv = cvpartition(classLabels, 'Holdout', 0.3);  % 70% training, 30% testing
    trainIdx = training(cv);
    testIdx = test(cv);
    
    % Create SVM template with RBF kernel
    svmTemplate = templateSVM(...
        'KernelFunction', 'rbf', ...
        'BoxConstraint', 1, ...
        'KernelScale', 'auto');
    
    % Train ECOC model with SVM template
    tic;
    svmModel = fitcecoc(features(trainIdx,:), classLabels(trainIdx), ...
        'Learners', svmTemplate, ...
        'Coding', 'onevsone', ...
        'Verbose', 1);  % Display training progress
    
    trainTime = toc;
    fprintf('SVM training completed in %.2f seconds\n', trainTime);
    
    % Predict on test set
    tic;
    pred = predict(svmModel, features(testIdx,:));
    testTime = toc;
    fprintf('Prediction completed in %.2f seconds\n', testTime);
    
    % Calculate accuracy
    accuracy = sum(strcmp(pred, classLabels(testIdx))) / numel(pred);
    fprintf('SVM Holdout Accuracy: %.2f%%\n', accuracy*100);
    
    % Confusion matrix
    figure('Name', 'Confusion Matrix', 'Position', [100, 100, 800, 600]);
    confusionchart(classLabels(testIdx), pred, ...
        'Title', 'Classification Performance', ...
        'RowSummary', 'row-normalized', ...
        'ColumnSummary', 'column-normalized');
    
    % Calculate class-wise precision and recall
    [cm, order] = confusionmat(classLabels(testIdx), pred);
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
    
    % Display class-wise metrics
    fprintf('\nClass-wise Performance Metrics:\n');
    fprintf('%-15s %-10s %-10s %-10s\n', 'Class', 'Precision', 'Recall', 'F1-Score');
    for i = 1:length(order)
        fprintf('%-15s %-10.2f %-10.2f %-10.2f\n', ...
                order{i}, stats(i,1), stats(i,2), stats(i,3));
    end
    
    % Feature importance analysis (for model interpretation)
    if exist('predictorImportance', 'file')
        imp = predictorImportance(svmModel);
        [~, idx] = sort(imp, 'descend');
        
        figure('Name', 'Feature Importance', 'Position', [100, 100, 800, 600]);
        bar(imp(idx(1:10)));
        set(gca, 'XTick', 1:10, 'XTickLabel', idx(1:10));
        xlabel('Feature Index');
        ylabel('Importance Score');
        title('Top 10 Important Features');
    end
    
    % Save model for future use
    save('svm_fault_classifier.mat', 'svmModel', 'trainIdx', 'testIdx', 'accuracy', 'stats');
else
    fprintf('Skipping SVM classification (insufficient data or missing toolbox)\n');
end

%% Synthetic Signal Generation (Bonus)
t_synth = 0:1/Fs:1;
chirpSignal = chirp(t_synth, 0, 1, 2000);
impulseSignal = [zeros(1, 1000), 1, zeros(1, length(t_synth)-1001)];
noise = 0.1 * randn(size(t_synth));
syntheticSignal = chirpSignal + impulseSignal + noise;

figure('Position', [100, 100, 800, 600]);
subplot(2,1,1);
plot(t_synth, syntheticSignal);
title('Synthetic Compressor Signal');
xlabel('Time (s)'); ylabel('Amplitude');
grid on;

subplot(2,1,2);
spectrogram(syntheticSignal, 1024, 512, 1024, Fs, 'yaxis');
title('Synthetic Signal Spectrogram');
clim([-80, 0]);

%% Cleanup
delete(gcp('nocreate'));  % Shutdown parallel pool
fprintf('Processing complete! %d files processed.\n', totalFiles);