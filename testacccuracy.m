load('NN.mat');
load('testSet.mat');

count = 0;
for i = 1:10000
   if (detect(testData(:,i)) == testLabels(i))
      count = count + 1;
   end
end

count / 10000