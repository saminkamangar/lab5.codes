function num = detect(fig)
    load('NN.mat');
    hidden1 = finalW1L1 * fig + finalB1L1;
    hidden1 = 1./(1+exp(-hidden1));
    
    hidden2 = finalW1L2 * hidden1 + finalB1L2;
    hidden2 = 1./(1+exp(-hidden2));
    
    final = finalSoftmaxTheta * hidden2;
    final = 1./(1+exp(-final));
    
    [v, i] = max(final);
    num = i;
end