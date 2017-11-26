ec = 1;
fc = 1;
ic = 1;
oc = 1;
nonCounter = 1;
for i=1:100
    fprintf('Cropping cells of image # %i\n',i);
    img = imread(sprintf('../data/Classification/img%i/img%i.bmp', i, i));
    epithelial = load(sprintf('../data/Classification/img%i/img%i_epithelial.mat', i, i));
    epithelial = epithelial.detection;
    fibroblast = load(sprintf('../data/Classification/img%i/img%i_fibroblast.mat', i, i));
    fibroblast = fibroblast.detection;
    inflammatory = load(sprintf('../data/Classification/img%i/img%i_inflammatory.mat', i, i));
    inflammatory = inflammatory.detection;
    others = load(sprintf('../data/Classification/img%i/img%i_others.mat', i, i));
    others = others.detection;

    for j = 1:size(epithelial, 1)
        c = epithelial(j, :);
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
        imwrite(patch, sprintf('../epithelial/%i.bmp', ec));
        ec = ec + 1;
    end
    
    for j = 1:size(fibroblast, 1)
        c = fibroblast(j, :);
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
        imwrite(patch, sprintf('../fibroblast/%i.bmp', fc));
        fc = fc + 1;
    end
    
    for j = 1:size(inflammatory, 1)
        c = inflammatory(j, :);
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
        imwrite(patch, sprintf('../inflammatory/%i.bmp', ic));
        ic = ic + 1;
    end
    
    for j = 1:size(others, 1)
        c = others(j, :);
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
        imwrite(patch, sprintf('../others/%i.bmp', oc));
        oc = oc + 1;
    end
end