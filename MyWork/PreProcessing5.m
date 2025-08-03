%% Enhanced Air Compressor Fault Classification System with Robust End-to-End Implementation
% Implements EMD-based SPA, optimized preprocessing, dynamic Fchoices, 
% advanced feature extraction & selection, and ensemble classification
clear; clc; close all;
warning('off','all');

%% System Configuration
mainFolder    = 'AirCompressor_Data';
faultClasses  = {'Bearing','Flywheel','Healthy','LIV','LOV','NRV','Piston','Riderbelt'};
Fs            = 50000;                              % Sampling rate (Hz)
segmentLength = Fs * 2;                             % 2-second window
stepSize      = segmentLength/4;                    % 75% overlap
initNumFeat   = 85;                                 % Initial feature count

%% Advanced Filter Bank Design
filters.hp  = designfilt('highpassiir','FilterOrder',6,'HalfPowerFrequency',100,'SampleRate',Fs);
filters.bp1 = designfilt('bandpassiir','FilterOrder',6,'HalfPowerFrequency1',400,'HalfPowerFrequency2',3000,'SampleRate',Fs);
filters.bp2 = designfilt('bandpassiir','FilterOrder',6,'HalfPowerFrequency1',3000,'HalfPowerFrequency2',8000,'SampleRate',Fs);
filters.bp3 = designfilt('bandpassiir','FilterOrder',6,'HalfPowerFrequency1',8000,'HalfPowerFrequency2',15000,'SampleRate',Fs);

[~,F] = pwelch(randn(segmentLength,1),[],[],[],Fs);
binEdges = linspace(0,Fs/2,17);
binMask = cell(1,16);
for b=1:16
    binMask{b} = F>=binEdges(b) & F<binEdges(b+1);
end
bearingFreqs   = [157,314,471,628,785,942];
harmonicOrders = 1:5;

%% SPA: Find Most Sensitive Position via EMD-based Ranking
allPositions = 24;
recsPerPos   = 15;
statsRMS     = zeros(allPositions,1);
statsMean    = zeros(allPositions,1);
for p = 1:allPositions
    envRMS = zeros(recsPerPos,1);
    envMean= zeros(recsPerPos,1);
    for r = 1:recsPerPos
        file = sprintf('Position_%02d_Rec_%02d.dat',p,r);
        sig  = preprocessRaw(file,Fs);
        imfs = emd(sig);
        corrThresh = 0.2;
        relIMFs = selectIMFs(imfs, sig, corrThresh);
        recSig  = sum(imfs(:,relIMFs),2);
        env     = abs(hilbert(recSig));
        envRMS(r)  = rms(env);
        envMean(r) = mean(abs(env));
    end
    statsRMS(p)  = mean(envRMS);
    statsMean(p) = mean(envMean);
end
[~,rR] = sort(statsRMS,'descend');
[~,rM] = sort(statsMean,'descend');
sumR   = rR + rM;
[~,order] = sort(sumR);
bestPos = order(1);

%% Load & Preallocate Files & Labels
files = dir(fullfile(mainFolder,faultClasses{1},'*.dat'));
Ntot  = numel(faultClasses)*numel(files);
filePaths = cell(Ntot,1); labels = cell(Ntot,1);
k=0;
for c=1:numel(faultClasses)
    D = dir(fullfile(mainFolder,faultClasses{c},'*.dat'));
    for i=1:numel(D)
        k=k+1; 
        filePaths{k} = fullfile(D(i).folder,D(i).name);
        labels{k}    = faultClasses{c};
    end
end

%% Parallel Feature Extraction
features = zeros(Ntot, initNumFeat);
parfor i=1:Ntot
    features(i,:) = extractAllFeatures( preprocessRaw( filePaths{i}, Fs ), Fs, filters, F, binMask, bearingFreqs, harmonicOrders, initNumFeat );
end
classLabels = labels';

%% Clean & Standardize
features(:, any(~isfinite(features),1)) = [];
features(:, std(features,0,1)<1e-10) = [];
numFeatures = size(features,2);
mu    = mean(features,1);
sigma = std(features,0,1); sigma(sigma<eps)=1;
X     = (features - mu)./sigma;

%% Feature Selection: MI + Relief-F + Variance → feature_order
numClasses = numel(faultClasses);
MI_scores = zeros(1,numFeatures);
try relief_scores = relieff(X,classLabels,10); catch, relief_scores=zeros(1,numFeatures); end
var_scores = var(X,[],1);
for f=1:numFeatures
    [~,~,bins] = histcounts(X(:,f),20);
    P=zeros(numClasses,20);
    for c=1:numClasses
        idx = strcmp(classLabels,faultClasses{c});
        if any(idx)
            cb = bins(idx);
            P(c,:) = histcounts(cb,1:21,'Normalization','probability');
        end
    end
    MI_scores(f)=sum(P(:).*log2(P(:)+eps));
end
combined = 0.4*MI_scores + 0.4*relief_scores + 0.2*var_scores;
[~,feature_order] = sort(combined,'descend');

%% Recompute Fchoices AFTER feature_order known
Fchoices = [20,35,50,65,min(80,numFeatures)];
Fchoices = Fchoices(Fchoices<=numFeatures);

%% Hyperparameter Optimization (OAO SVM RBF)
Cgrid = [0.01,0.1,1,10,100,1000]; 
Sgrid = [0.001,0.01,0.1,1,10];
bestPerf = struct('acc',0,'f1',0,'C',1,'S',1,'nf',Fchoices(1));
cv_results=[];
for nf=Fchoices
    Xf = X(:,feature_order(1:nf));
    cvp= cvpartition(classLabels,'KFold',5,'Stratify',true);
    for C=Cgrid
        for S=Sgrid
            accs = zeros(cvp.NumTestSets,1);
            f1s  = zeros(cvp.NumTestSets,1);
            for k=1:cvp.NumTestSets
                tr = training(cvp,k); te=test(cvp,k);
                t=templateSVM('KernelFunction','rbf','BoxConstraint',C,'KernelScale',S,'Standardize',true);
                mdl=fitcecoc(Xf(tr,:),classLabels(tr),'Learners',t,'Coding','onevsone');
                pred=predict(mdl,Xf(te,:));
                accs(k)=mean(strcmp(pred,classLabels(te)));
                f1s(k)=mean(computeF1(pred,classLabels(te)));
            end
            avg_acc=mean(accs); avg_f1=mean(f1s);
            score=0.6*avg_acc+0.4*avg_f1;
            cv_results(end+1,:)=[nf,C,S,avg_acc,avg_f1,score]; 
            if score> (0.6*bestPerf.acc+0.4*bestPerf.f1)
                bestPerf=struct('acc',avg_acc,'f1',avg_f1,'C',C,'S',S,'nf',nf);
            end
        end
    end
end

%% Final Ensemble Training & Testing
Xf = X(:,feature_order(1:bestPerf.nf));
cvp= cvpartition(classLabels,'Holdout',0.2,'Stratify',true);
tr=training(cvp); te=test(cvp);
t1=templateSVM('KernelFunction','rbf','BoxConstraint',bestPerf.C,'KernelScale',bestPerf.S,'Standardize',true);
M{1}=fitcecoc(Xf(tr,:),classLabels(tr),'Learners',t1,'Coding','onevsone');
try M{2}=TreeBagger(200,Xf(tr,:),classLabels(tr),'Method','classification'); catch, M{2}=M{1}; end
t3=templateSVM('KernelFunction','polynomial','PolynomialOrder',3,'BoxConstraint',bestPerf.C,'KernelScale',bestPerf.S,'Standardize',true);
try M{3}=fitcecoc(Xf(tr,:),classLabels(tr),'Learners',t3,'Coding','onevsone'); catch, M{3}=M{1}; end
for i=1:3, P{i}=predict(M{i},Xf(te,:)); end
ensemble = majorityVote(P);
finalAcc = mean(strcmp(ensemble,classLabels(te)));
fprintf('Final Ensemble Test Accuracy: %.2f%%\n',finalAcc*100);

%% Helper Functions

function sig = preprocessRaw(file, Fs)
    fid = fopen(file, 'r');
if fid < 0
    error('Could not open file: %s', file);
end
raw = fread(fid, inf, 'uint8=>uint8');
fclose(fid);

    N = floor(numel(raw)/3);
    bytes = reshape(raw(1:3*N),3,[]);
    data = double(bytes(1,:)) + 256*double(bytes(2,:)) + 65536*double(bytes(3,:));
    neg = bytes(3,:)>=128; data(neg)=data(neg)-2^24;
    sig = data'/2^23; sig = sig - mean(sig);
    % Filtering
    sig = filter(designfilt('highpassiir','FilterOrder',6,'HalfPowerFrequency',400,'SampleRate',Fs),sig);
    sig = filter(designfilt('lowpassiir','FilterOrder',18,'HalfPowerFrequency',12000,'SampleRate',Fs),sig);
    % Clipping
    w=Fs; ov=w/2; starts=1:ov:(numel(sig)-w+1);
    if isempty(starts), idx=1; else
        stds = arrayfun(@(s) std(sig(s:s+w-1)),starts);
        [~,idx]=min(stds);
    end
    seg = sig(starts(idx):starts(idx)+w-1);
    % Smoothing
    seg = movmean(seg,7);
    % Normalization via fast histogram-based method
    L=prctile(seg,0.25); U=prctile(seg,99.75);
    sig = 2*(seg - L)/(U-L) - 1;
end

function feats = extractAllFeatures(sig, Fs, filters, F, binMask, bf, ho, nF)
    % Combine preprocessing + Advanced preprocessing
    sig2 = advancedPreprocessing(sig,Fs);
    feats = extractEnhancedFeatures(sig2,Fs,filters,F,binMask,bf,ho,nF);
end

function imfs = emd(sig)
    imfs = emd(sig,'MaxNumIMF',10,'Display',0);
end

function rel = selectIMFs(imfs,sig,th)
    C = corrcoef([sig imfs]); c = C(1,2:end);
    rel = find(c>th);
end

function score = computeF1(pred,truth)
    classes = unique(truth); K=numel(classes);
    f1s = zeros(K,1);
    for i=1:K
        tp=sum(strcmp(pred,classes{i})&strcmp(truth,classes{i}));
        fp=sum(strcmp(pred,classes{i})&~strcmp(truth,classes{i}));
        fn=sum(~strcmp(pred,classes{i})&strcmp(truth,classes{i}));
        p=tp/(tp+fp+eps); r=tp/(tp+fn+eps);
        f1s(i)=2*p*r/(p+r+eps);
    end
    score = f1s;
end

function vm = majorityVote(P)
    N = numel(P{1}); vm=cell(N,1);
    for i=1:N
        votes = {P{1}{i},P{2}{i},P{3}{i}};
        [u,~,idx]=unique(votes); vc=accumarray(idx,1);
        [~,mi]=max(vc); vm{i}=u{mi};
    end
end

function out = advancedPreprocessing(sig,Fs)
    try
        if exist('emd','file')
            [imfs,~]=emd(sig);
            if size(imfs,2)>3, out=sum(imfs(:,3:end),2);
            else out=sig; end
        else out=sig; end
        if exist('dsp.LMSFilter','class')
            ha=dsp.LMSFilter(32,0.1); [~,out]=ha(out,out);
        end
        if exist('wdenoise','file')
            out=wdenoise(out,'Wavelet','db8','DenoisingMethod','BlockJS','ThresholdRule','Soft');
        end
    catch
        out=sig;
    end
end
