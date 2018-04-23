counter = 0;
for i=1:100
    detection = load(sprintf('../data/Detection/img%i/img%i_detection.mat', i, i));
    detection = detection.detection;
    counter = counter + size(detection, 1);
    
    for k = 1:5
        detection = load(sprintf('../data/Detection/img%i/img%i_%i_detection.mat', i, i, k));
        detection = detection.detection;
        counter = counter + size(detection, 1);
    end
    
end
fprintf('Total cells: # %i\n',counter);

traningFeature = zeros(1,900);
counter = 1;
for i=1:100
    fprintf('Extracting feature of image # %i\n',i);
    img = imread(sprintf('../data/Detection/img%i/img%i.bmp', i, i));
    detection = load(sprintf('../data/Detection/img%i/img%i_detection.mat', i, i));
    detection = detection.detection;
    for j = 1:size(detection, 1)
        c = detection(j, :);
        x = round(c(1));
        y = round(c(2));
        x1 = max(1, x-6);
        x2 = min(500, x+6);
        y1 = max(1, y-6);
        y2 = min(500, y+6);
        if x1 == 1
            x2 = 13;
        end
        if x2 == 500
            x1 = 488;
        end
        if y1 == 1
            y2 = 13;
        end
        if y2 == 500
            y1 = 488;
        end
        patch = img(y1:y2, x1:x2,:);
        [featureVector,hogVisualization] = extractHOGFeatures(patch, 'CellSize', [2 2]);
        traningFeature(counter,:) = featureVector;
        counter = counter + 1;
    end
    
    for k = 1:5
        img = imread(sprintf('../data/Detection/img%i/img%i_%i.bmp', i, i, k));
        detection = load(sprintf('../data/Detection/img%i/img%i_%i_detection.mat', i, i, k));
        detection = detection.detection;
        for j = 1:size(detection, 1)
            c = detection(j, :);
            x = round(c(1));
            y = round(c(2));
            x1 = max(1, x-6);
            x2 = min(500, x+6);
            y1 = max(1, y-6);
            y2 = min(500, y+6);
            if x1 == 1
                x2 = 13;
            end
            if x2 == 500
                x1 = 488;
            end
            if y1 == 1
                y2 = 13;
            end
            if y2 == 500
                y1 = 488;
            end
            patch = img(y1:y2, x1:x2,:);
            [featureVector,hogVisualization] = extractHOGFeatures(patch, 'CellSize', [2 2]);
            traningFeature(counter,:) = featureVector;
            counter = counter + 1;
        end
    end
    
end

trainingLabels = zeros(counter-1, 1);

for i = 1:counter-1
    trainingLabels(i) = 1;
end


% Here I train a linear support vector machine (SVM) classifier.
svmmdl = fitcsvm(trainingFeatures ,trainingLabels);

save('SVMmodel1.mat', 'svmmdl');

% Perform cross-validation and check accuracy
cvmdl = crossval(svmmdl,'KFold',10);
fprintf('kFold CV accuracy: %2.2f\n',1-cvmdl.kfoldLoss)
