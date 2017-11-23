counter = 1;
for i=1:100
    fprintf('Cropping cells of image # %i\n',i);
    img = imread(sprintf('../data/Detection/img%i/img%i.bmp', i, i));
    detection = load(sprintf('../data/Detection/img%i/img%i_detection.mat', i, i));
    detection = detection.detection;
    for j = 1:size(detection, 1)
        c = detection(j, :);
        x = round(c(1));
        y = round(c(2));
        x1 = max(1, x-13);
        x2 = min(500, x+13);
        y1 = max(1, y-13);
        y2 = min(500, y+13);
        if x1 == 1
            x2 = 27;
        end
        if x2 == 500
            x1 = 474;
        end
        if y1 == 1
            y2 = 27;
        end
        if y2 == 500
            y1 = 474;
        end
        patch = img(y1:y2, x1:x2,:);
        imwrite(patch, sprintf('../cell/%i.bmp', counter));
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
            x1 = max(1, x-13);
            x2 = min(500, x+13);
            y1 = max(1, y-13);
            y2 = min(500, y+13);
            if x1 == 1
                x2 = 27;
            end
            if x2 == 500
                x1 = 474;
            end
            if y1 == 1
                y2 = 27;
            end
            if y2 == 500
                y1 = 474;
            end
            patch = img(y1:y2, x1:x2,:);
            imwrite(patch, sprintf('../cell/%i.bmp', counter));
            counter = counter + 1;
        end
    end
    
end