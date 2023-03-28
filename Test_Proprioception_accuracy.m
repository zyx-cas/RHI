clear;clc;
AccuracyResult = [];
for s = 2:6
    ISI = 1;
    JMax = 30;
    listJ = [-JMax:ISI:JMax]; 
    list = [];
    for i = 1:size(listJ,2)
        e = listJ(i);
        listY = [];
        for j = 1:size(listJ,2)   
            x = listJ(j);
            y = exp(-(x-e)^2/(2*s^2));
            listY = [listY,y];
        end   
        list = [list;listY];
    end
    Coding = list;
    
    
    N = size(Coding,2);
    Trick = 1;    
    if Trick == 0
        for k = 1:N 
            disp(k);
            DealyTime = 0;
            Train_Batch(k, Coding, DealyTime);
        end
    else
        for k = 1:1
            disp(k);
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
    end

    Result = Test_Proprioception_accuracy_batch(Coding);
    AccuracyResult = [AccuracyResult, Result];
end

D1 = AccuracyResult(22:40,1:2);
D2 = AccuracyResult(17:45,3:4);
D3 = AccuracyResult(12:50,5:6);
D4 = AccuracyResult(7:55,7:8);
D5 = AccuracyResult(2:60,9:10);

% Coding2
P1 = polyfit(D1(:,1),D1(:,2),3);
X1 = -9:0.1:9;
Y1 = polyval(P1,X1);
figure;
plot(D1(:,1),D1(:,2),'.','color','g','markersize',8)
hold on;
L1 = plot(X1,Y1,'linewidth',2,'color','g')

% Coding3
P2 = polyfit(D2(:,1),D2(:,2),3);
X2 = -14:0.1:14;
Y2 = polyval(P2,X2);
hold on;
plot(D2(:,1),D2(:,2),'.','color','m','markersize',8)
hold on;
L2 = plot(X2,Y2,'linewidth',2,'color','m')

% Coding4
P3 = polyfit(D3(:,1),D3(:,2),3);
X3 = -19:0.1:19;
Y3 = polyval(P3,X3);
hold on;
plot(D3(:,1),D3(:,2),'.','color','b','markersize',8)
hold on;
L3 = plot(X3,Y3,'linewidth',2,'color','b')

% Coding5
P4 = polyfit(D4(:,1),D4(:,2),3);
X4 = -24:0.1:24;
Y4 = polyval(P4,X4);
hold on;
plot(D4(:,1),D4(:,2),'.','color','r','markersize',8)
hold on;
L4 = plot(X4,Y4,'linewidth',2,'color','r')

% Coding6
P5 = polyfit(D5(:,1),D5(:,2),3);
X5 = -29:0.1:29;
Y5 = polyval(P5,X5);
hold on;
plot(D5(:,1),D5(:,2),'.','color','k','markersize',8)
hold on;
L5 = plot(X5,Y5,'linewidth',2,'color','k')

xlabel('Disparity','fontsize',14);
ylabel('Drift','fontsize',14);
set(gca,'fontsize',13)
H = legend([L1,L2,L3,L4,L5],'{\sigma=2}','{\sigma=3}','{\sigma=4}','{\sigma=5}','{\sigma=6}','Location', 'SouthEast')
set(H,'FontSize',13,'FontWeight','normal')

