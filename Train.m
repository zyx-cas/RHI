clear;clc;

load('Coding.mat')
N = size(Coding,2);

TrickID = 1;
if TrickID == 0
    for k = 1:N
        disp(k);
        DealyTime = 0;
        Train_Batch(k, Coding, DealyTime);
    end
else
    k = 1;
    DealyTime = 0;
    Train_Batch(k, Coding, DealyTime);
    load('Train.mat')
    W_M1_S1 = repmat(W_M1_S1(1),N,1);  
    W_V_EBA = repmat(W_V_EBA(1),N,1);  
    W_S1_TPJ = repmat(W_S1_TPJ(1),N,1); 
    W_EBA_TPJ = repmat(W_EBA_TPJ(1),N,1);  
    W_TPJ_AI = repmat(W_TPJ_AI(1),N,1);  
    W_S1_AI = repmat(W_S1_AI(1),N,1);  
    W_EBA_AI = repmat(W_EBA_AI(1),N,1); 
    matname = 'Train.mat';                
    save(matname, 'W_M1_S1', 'W_V_EBA', 'W_S1_TPJ', 'W_EBA_TPJ', 'W_TPJ_AI', 'W_S1_AI', 'W_EBA_AI')
end
