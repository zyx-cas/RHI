clear;clc;

load('Train.mat')
load('Coding.mat')

setGlobalOpts()
global Opts
alpha = Opts.alpha;
beta = Opts.beta;
gamma = Opts.gamma; 
C_M1 = Opts.C_M1;
C_V = Opts.C_V;
C_S1 = Opts.C_S1;
C_EBA = Opts.C_EBA;
C_TPJ = Opts.C_TPJ;
C_AI = Opts.C_AI;
N = size(Coding,2);
Time = Opts.Time; 
ContinueTime = Opts.ContinueTime ; 


DealyTimeList = [0,100,500];
DealyTimeResult = [];
for DealyTimeID = 1:3
    DealyTime = DealyTimeList(DealyTimeID);
    Motion_Start = 1;
    Motion_End = Motion_Start + ContinueTime;
    Vision_Start = Motion_End + DealyTime;
    Vision_End = Vision_Start + ContinueTime;

    Veridical_hand = 16;
    Result = [];
    for Disparity = -15:15
        V_M1 = zeros(N,1);
        V_V = zeros(N,1);
        V_S1 = zeros(N,1);
        V_EBA = zeros(N,1);
        V_TPJ = zeros(N,1);
        V_AI = zeros(N,1);  
        List_V_AI = [];

        for t= 1:Time  
            S_M = zeros(N,1);
            S_V = zeros(N,1);
            if t>Motion_Start && t<Motion_End
                S_M = Coding(Veridical_hand,:)';
            end
            if t>Vision_Start && t<Vision_End
               S_V = Coding(Veridical_hand+Disparity,:)';
            end

            V_M1_n = Neuron_Pre(S_M, V_M1, C_M1); 
            V_V_n = Neuron_Pre(S_V, V_V, C_V); 

            % M1 -> S1
            V_S1_n = Neuron_Post(V_S1, V_M1, W_M1_S1, C_S1);     

            % V -> EBA
            V_EBA_n = Neuron_Post(V_EBA, V_V, W_V_EBA, C_EBA);

            % S1 & EBA -> TPJ
            V_TPJ_Input = [V_S1_n, V_EBA_n];
            W_TPJ_Input = [W_S1_TPJ, W_EBA_TPJ];
            V_TPJ_n = Neuron_Post(V_TPJ, V_TPJ_Input, W_TPJ_Input, C_TPJ);         

            % S1 & TPJ & EBA -> AI
            V_AI_Input = [V_S1_n, V_TPJ_n, V_EBA_n];
            W_AI_Input = [W_S1_AI, W_TPJ_AI, W_EBA_AI]; 
            V_AI_n = Neuron_Post(V_AI, V_AI_Input, W_AI_Input, C_AI);        
            List_V_AI = [List_V_AI, V_AI_n];

            V_M1 = V_M1_n;
            V_V = V_V_n;
            V_S1 = V_S1_n;
            V_EBA = V_EBA_n;
            V_TPJ = V_TPJ_n;
            V_AI = V_AI_n;       
        end    

        maxValue = max(List_V_AI');
        [~,Estimated_hand] = max(maxValue);
        Proprioceptive_drift = Estimated_hand - Veridical_hand;
        R = [Disparity,Proprioceptive_drift];
        Result = [Result; R];
    end
DealyTimeResult = [DealyTimeResult, Result];
end

%matname = 'Result.mat';                
%save(matname, 'Result')

figure;
P2 = polyfit(DealyTimeResult(2:30,1)*3,DealyTimeResult(2:30,2)*3,3);
X2 = -14*3:0.1:14*3;
Y2 = polyval(P2,X2);
hold on;
plot(DealyTimeResult(:,1)*3,DealyTimeResult(:,2)*3,'.','color','k','markersize',8)
hold on;
L2 = plot(X2,Y2,'linewidth',2,'color','k')
set(gca,'XLim',[-45 45]);
set(gca,'XTick',[-45:15:45]);

P3 = polyfit(DealyTimeResult(:,3)*3,DealyTimeResult(:,4)*3,3);
X3 = -15*3:0.1:15*3;
Y3 = polyval(P3,X3);
hold on;
plot(DealyTimeResult(:,3)*3,DealyTimeResult(:,4)*3,'.','color','b','markersize',8)
hold on;
L3 = plot(X3,Y3,'linewidth',2,'color','b')

P4 = polyfit(DealyTimeResult(:,5)*3,DealyTimeResult(:,6)*3,3);
X4 = -15*3:0.1:15*3;
Y4 = polyval(P4,X4);
hold on;
plot(DealyTimeResult(:,5)*3,DealyTimeResult(:,6)*3,'.','color','r','markersize',8)
hold on;
L4 = plot(X4,Y4,'linewidth',2,'color','r')

xlabel('Disparity (deg)','fontsize',14);
ylabel('Drift (deg)','fontsize',14);
legend([L2,L3,L4],'DelayTime=0','DelayTime=100','DelayTime=500','Location', 'SouthEast')
set(gca,'fontsize',13)
