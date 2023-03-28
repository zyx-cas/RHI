clear;clc;

load('Coding.mat')
N = size(Coding,2);
for k = 1:N
    disp(k);
    DealyTime = 0;
    Train_Batch(k, Coding, DealyTime);
end
