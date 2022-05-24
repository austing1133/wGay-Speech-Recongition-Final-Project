clear all; close all;

%dataDir = 'C:\Users\austi\OneDrive\Documents\Speech Recognition\Final Project\data';
dataDir = 'data';

%----------------------------------------------------------------
%Put audio data into an audio datastore

ads = audioDatastore(dataDir, ...
        IncludeSubfolders=true, ...
        FileExtensions=".wav", ...
        LabelSource="foldernames")

%----------------------------------------------------------------
%Split data into train and test

[adsTrain, adsTest] = splitEachLabel(ads,0.7);

%----------------------------------------------------------------
%Train the model(s)

model = trainModel(adsTrain, adsTest);

%----------------------------------------------------------------
%Read in audio from live input

%deviceReader = audioDeviceReader(Device='Microphone (USB2.0 Microphone)');
deviceReader = audioDeviceReader();
fileWriter = dsp.AudioFileWriter('SampleRate',deviceReader.SampleRate);

disp('Begin Signal Input...')
tic
while toc<3
    mySignal = deviceReader();
    fileWriter(mySignal);
end
disp('End Signal Input')

release(deviceReader)
release(fileWriter)

[audioIn, fs] = audioread('output.wav');

%----------------------------------------------------------------
%Process live audio and extract features

input_features = process_audio(audioIn, fs);

%----------------------------------------------------------------
%Predict speaker/word of audio using K-Nearest Neighbor

[prediction,gc,grps] = predict_speaker_knn(input_features, model);
disp('KNN Model')
prediction
percent = max(gc)/sum(gc)
gc
grps

%----------------------------------------------------------------
%Predict speaker/word using Gaussian Mixture Model

disp('GMM Model')
[speaker, gc3, grps3] = predict_speaker_gmm(input_features, obj1, obj2, obj3, obj4, obj5, obj6, obj7, obj8);
mode(categorical(speaker))
percent = max(gc3)/sum(gc3)
gc3
grps3

%----------------------------------------------------------------
%Predict speaker/word using Naive Bayes Classifier
disp('Naive Bayes Model')
[predict, posterior, cost] = nBayes(bayesModel,input_features);

[gc2,grps2] = groupcounts(predict);
mode(predict)
percent2 = max(gc2)/sum(gc2)
gc2
grps2

%----------------------------------------------------------------

function m = trainModel(adsTrain, adsTest)

    %Process Audio from data

    [sampleTrain, dsInfo] = read(adsTrain);
    
    reset(adsTrain);

    fs = dsInfo.SampleRate;

    
    windowLength = round(0.03*fs);
    overlapLength = round(0.025*fs);
    afe = audioFeatureExtractor(SampleRate=fs, ...
        Window=hamming(windowLength,"periodic"), OverlapLength=overlapLength, ...
        zerocrossrate=true, shortTimeEnergy=true, pitch=true, mfcc = true);
    
    featureMap = info(afe)
    
    features = [];
    labels = [];
    lengths = [];
    energyThreshold = 0.005;
    zcrThreshold = 0.2;
    while hasdata(adsTrain)
        [audioIn, dsInfo] = read(adsTrain);
    
        feature = extract(afe, audioIn);
        isSpeech = feature(:, featureMap.shortTimeEnergy) > energyThreshold;
        isVoiced = feature(:, featureMap.zerocrossrate) < zcrThreshold;
    
        voicedSpeech = isSpeech & isVoiced;
    
        feature(~voicedSpeech,:) = [];
        feature(:,[featureMap.zerocrossrate,featureMap.shortTimeEnergy]) = [];
        label = repelem(dsInfo.Label,size(feature,1));
        
        lengths = [lengths;size(feature,1)];

        features = [features;feature];
        assignin('base', "test", features);
        labels = [labels,label];
        assignin('base', 'features', features);
    end
    global M; global S;
    M = mean(features,1);
    S = std(features, [], 1);
    features = (features-M)./S;
    
    %----------------------------------------------------------------

    %Train Model(s)

    %Naiive Bayes
    bayesModel = fitcnb(features,labels,'ClassNames',unique(labels));
    assignin('base','bayesModel',bayesModel);
    
    %GMM
    
    a = sum(lengths(1:7));
    obj1 = gmdistribution.fit(features(1:a,:), 14, 'CovType','diagonal','SharedCov',true);
    a = sum(lengths(1:7));
    b = a + sum(lengths(8:14));
    obj2 = gmdistribution.fit(features(a+1:b,:), 14, 'CovType','diagonal','SharedCov',true);
    a = a + sum(lengths(8:14));
    b = b + sum(lengths(15:21));
    obj3 = gmdistribution.fit(features(a+1:b,:), 14, 'CovType','diagonal','SharedCov',true);
    a = a + sum(lengths(15:21));
    b = b + sum(lengths(22:28));
    obj4 = gmdistribution.fit(features(a+1:b,:), 14, 'CovType','diagonal','SharedCov',true);
    a = a + sum(lengths(22:28));
    b = b + sum(lengths(29:35));
    obj5 = gmdistribution.fit(features(a+1:b,:), 14, 'CovType','diagonal','SharedCov',true);
    a = a + sum(lengths(29:35));
    b = b + sum(lengths(36:42));
    obj6 = gmdistribution.fit(features(a+1:b,:), 14, 'CovType','diagonal','SharedCov',true);
    a = a + sum(lengths(36:42));
    b = b + sum(lengths(43:49));
    obj7 = gmdistribution.fit(features(a+1:b,:), 14, 'CovType','diagonal','SharedCov',true);
    a = a + sum(lengths(43:49));
    b = b + sum(lengths(50:56));
    obj8 = gmdistribution.fit(features(a+1:b,:), 14, 'CovType','diagonal','SharedCov',true);

    assignin('base',"obj1",obj1);
    assignin('base',"obj2",obj2);
    assignin('base',"obj3",obj3);
    assignin('base',"obj4",obj4);
    assignin('base',"obj5",obj5);
    assignin('base',"obj6",obj6);
    assignin('base',"obj7",obj7);
    assignin('base',"obj8",obj8);


%     options = statset('maxIter',100);
%     trainedModel = fitgmdist(features, 100, 'RegularizationValue', 0.01, 'Start','plus', 'Options',options);
%     assignin('base', 'trainedModel', trainedModel);

    %KNN Classifier
    trainedClassifier = fitcknn(features, labels, ...
        Distance="euclidean", ...
        NumNeighbors=5, ...
        DistanceWeight="squaredinverse", ...
        Standardize=false, ...
        ClassNames=unique(labels));
    
    %----------------------------------------------------------------

    k = 5;
    group = labels;
    c = cvpartition(group, KFold=k);
    partitionedModel = crossval(trainedClassifier, CVPartition=c);
    %partitionedModel = crossval(bayesModel, CVPartition=c);
    
    validationAccuracy = 1 - kfoldLoss(partitionedModel, LossFun="ClassifError");
    fprintf('\nValidation accuracy = %.2f%%\n', validationAccuracy*100);
    
    %----------------------------------------------------------------

    %Testing Classifier
    
    features = [];
    labels = [];
    numVectorsPerFile = [];
    while hasdata(adsTest)
        [audioIn, dsInfo] = read(adsTest);
    
        feature = extract(afe, audioIn);
    
        isSpeech = feature(:,featureMap.shortTimeEnergy) > energyThreshold;
        isVoiced = feature(:,featureMap.zerocrossrate) < zcrThreshold;
    
        voicedSpeech = isSpeech & isVoiced;
    
        feature(~voicedSpeech,:) = [];
        numVec = size(feature,1);
        feature(:,[featureMap.zerocrossrate,featureMap.shortTimeEnergy]) = [];
    
        label = repelem(dsInfo.Label, numVec);
    
        numVectorsPerFile = [numVectorsPerFile, numVec];
    
        features = [features;feature];
        labels = [labels, label];
    end
    features = (features-M)./S;
    
    prediction = predict(trainedClassifier,features);
    prediction = categorical(string(prediction));
    
    r2 = prediction(1:numel(adsTest.Files));
    idx = 1;
    for ii = 1:numel(adsTest.Files)
        r2(ii) = mode(prediction(idx:idx+numVectorsPerFile(ii)-1));
        idx = idx + numVectorsPerFile(ii);
    end
    
    figure(Units="normalized",Position=[0.4 0.4 0.4 0.4])
    confusionchart(adsTest.Labels,r2,title="Test Accuracy (Per File)", ...
        ColumnSummary="column-normalized",RowSummary="row-normalized");
    
    m = trainedClassifier;

%----------------------------------------------------------------

    predlabels = predict(bayesModel,features);

    r3 = predlabels(1:numel(adsTest.Files));
    idx = 1;
    for ii = 1:numel(adsTest.Files)
        r3(ii) = mode(predlabels(idx:idx+numVectorsPerFile(ii)-1));
        idx = idx + numVectorsPerFile(ii);
    end

    %table(adsTest.Labels,r3,'VariableNames',...
    %{'TrueLabel','PredictedLabel'})
    figure
    cm = confusionchart(adsTest.Labels,r3);
    
  

end

%----------------------------------------------------------------
%Extract audio features

function features = process_audio(audioIn, fs)

    windowLength = round(0.03*fs);
    overlapLength = round(0.025*fs);
    energyThreshold = 0.005;
    zcrThreshold = 0.2;

    afe = audioFeatureExtractor(SampleRate=fs, ...
        Window=hamming(windowLength,"periodic"), OverlapLength=overlapLength, ...
        zerocrossrate=true, shortTimeEnergy=true, pitch=true, mfcc = true);

    featureMap = info(afe);

    features = extract(afe, audioIn);

    isSpeech = features(:, featureMap.shortTimeEnergy) > energyThreshold;
    isVoiced = features(:, featureMap.zerocrossrate) < zcrThreshold;

    voicedSpeech = isSpeech & isVoiced;

    features(~voicedSpeech,:) = [];
    features(:,[featureMap.zerocrossrate,featureMap.shortTimeEnergy]) = [];
    
    global M;
    global S;

    features = (features - M)./S;
end

%----------------------------------------------------------------
%K-Nearest Neighbor Prediction

function [predicted,gc,grps] = predict_speaker_knn(features, model)

    prediction = predict(model,features);
    prediction = categorical(string(prediction));
    predicted = mode(prediction);
    [gc, grps] = groupcounts(prediction);

end

%----------------------------------------------------------------
%Gaussian Mixture Model Prediction

function [final_label, gc, grps] = predict_speaker_gmm(features, obj1, obj2, obj3, obj4, obj5, obj6, obj7, obj8)

    pdf1 = pdf(obj1,features);
    pdf2 = pdf(obj2,features);
    pdf3 = pdf(obj3,features);
    pdf4 = pdf(obj4,features);
    pdf5 = pdf(obj5,features);
    pdf6 = pdf(obj6,features);
    pdf7 = pdf(obj7,features);
    pdf8 = pdf(obj8,features);

    mod = [pdf1 pdf2 pdf3 pdf4 pdf5 pdf6 pdf7 pdf8];

    [~,ymax] = max(mod,[],2);

    real_labels = ["austin_bicycle" "austin_enter" "austin_guitar" "austin_hello" "austin_music" "austin_water" "nigel_enter" "seth_enter"];
    
    [gc,grps] = groupcounts(ymax);

    final_label = real_labels(ymax);

end

%----------------------------------------------------------------
%Additional recording and classification

function record(trainedModel)
    
    deviceReader = audioDeviceReader(Device='Microphone (USB2.0 Microphone)');
    fileWriter = dsp.AudioFileWriter('SampleRate',deviceReader.SampleRate);
    
    disp('Begin Signal Input...')
    tic
    while toc<3
        mySignal = deviceReader();
        fileWriter(mySignal);
    end
    disp('End Signal Input')
    
    release(deviceReader)
    release(fileWriter)
    
    [audioIn, fs] = audioread('output.wav');
    input_features = process_audio(audioIn, fs);

    clusters = predict_speaker_gmm(input_features, trainedModel);
    tabulate(clusters)

end

%----------------------------------------------------------------
%Naive Bayes Classification Prediction

function [prediction,Posterior, Cost] = nBayes(model, features)
    
    [prediction, Posterior, Cost] = predict(model,features);
    
end
