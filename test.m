function test(index, data)
    figure = data(:,index);
    count = 1;
    for i = 1:28
        for j = 1:28
            img(j, i) = figure(count);
            count = count + 1;
        end
    end
    imshow(img)
end