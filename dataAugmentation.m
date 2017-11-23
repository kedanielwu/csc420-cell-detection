clear all;
close all;

for i = 1:100
    old_detection = load(sprintf('../data/Detection/img%i/img%i_detection.mat', i, i));
    im = imread(sprintf('../data/Detection/img%i/img%i.bmp', i, i));
    old_detection = old_detection.detection;

    im1 = im((500:-1:1),:,:);
    detection = old_detection;
    detection(:,2) = 500 - detection(:,2);
    imwrite(im1, sprintf('../data/Detection/img%i/img%i_1.bmp', i, i));
    save(sprintf('../data/Detection/img%i/img%i_1_detection.mat', i, i), 'detection');

    im1 = im(:,(500:-1:1),:);
    detection = old_detection;
    detection(:,1) = 500 - detection(:,1);
    imwrite(im1, sprintf('../data/Detection/img%i/img%i_2.bmp', i, i));
    save(sprintf('../data/Detection/img%i/img%i_2_detection.mat', i, i), 'detection');

    for j = 1:3
        theta = 90 * j; % to rotate 90 counterclockwise
        im1 = imrotate(im, 360-theta);
        R = [cosd(theta) -sind(theta); sind(theta) cosd(theta)];
        detection = (R*(old_detection-250)')' + 250;
        imwrite(im1, sprintf('../data/Detection/img%i/img%i_%i.bmp', i, i, 2+j));
        save(sprintf('../data/Detection/img%i/img%i_%i_detection.mat', i, i, 2+j), 'detection');
    end
end
