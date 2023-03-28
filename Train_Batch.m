function Train_Batch(k,Coding,DealyTime)
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
MoveNum = Opts.MoveNum; 
Time = Opts.Time; 
ContinueTime = Opts.ContinueTime ; 
Weight = Opts.Weight;

if k == 1
    W_M1_S1 = zeros(N,1) + Weight; 
    W_V_EBA = zeros(N,1) + Weight;
    W_S1_TPJ = zeros(N,1) + Weight;
    W_EBA_TPJ = zeros(N,1) + Weight;
    W_TPJ_AI = zeros(N,1) + Weight;
    W_S1_AI = zeros(N,1) + Weight;
    W_EBA_AI = zeros(N,1) + Weight;
else 
    load('Train.mat')
end


W_LatInh_Init = zeros(N,1) + 1;

fire_threshold = 0.7;

Motion_Start = 1;
Motion_End = Motion_Start + ContinueTime;
Vision_Start = Motion_End + DealyTime;
Vision_End = Vision_Start + ContinueTime;

List_W_S1_TPJ = [];
List_W_EBA_TPJ = [];

for i = 1:MoveNum
    V_M1 = zeros(N,1);
    V_V = zeros(N,1);
    V_S1 = zeros(N,1);
    V_EBA = zeros(N,1);
    V_TPJ = zeros(N,1);
    V_AI = zeros(N,1);  
  
    dW_S1_TPJ = 0;
    dW_EBA_TPJ = 0;  
    dW_S1_AI = 0;
    dW_EBA_AI = 0;

    W_LatInh_S1_TPJ = W_LatInh_Init; 
    W_LatInh_EBA_TPJ = W_LatInh_Init; 
    W_LatInh_S1_AI = W_LatInh_Init; 
    W_LatInh_EBA_AI = W_LatInh_Init; 
    
    Fired_S1 = zeros(N,1);
    Fired_EBA = zeros(N,1);
    for t= 1:Time  
        S_M = zeros(N,1);
        S_V = zeros(N,1);
        if t>Motion_Start && t<Motion_End
            S_M = Coding(k,:)';
        end
        if t>Vision_Start && t<Vision_End
           S_V = Coding(k,:)';
        end
        
        V_M1_n = Neuron_Pre(S_M, V_M1, C_M1); 
        V_V_n = Neuron_Pre(S_V, V_V, C_V); 
        
        % M1 -> S1
        V_S1_n = Neuron_Post(V_S1, V_M1, W_M1_S1, C_S1);     
        S_S1_n = V_S1_n;
        S_S1_n(V_S1_n >= fire_threshold) = 1;
        S_S1_n(V_S1_n < fire_threshold) = 0; 
        if sum(S_S1_n) > 0       
            Fired_S1 = Fired_S1 + 1; 
            W_LatInh_S1_TPJ = tanh(W_LatInh_S1_TPJ - 2 * acos(S_S1_n) / pi .* exp(Fired_S1) - 1) + 1;  
            W_LatInh_S1_AI = tanh(W_LatInh_S1_AI - 2 * acos(S_S1_n) .* exp(Fired_S1) - 1) + 1;
        end        
        
        % V -> EBA
        V_EBA_n = Neuron_Post(V_EBA, V_V, W_V_EBA, C_EBA);
        S_EBA_n = V_EBA_n;
        S_EBA_n(V_EBA_n >= fire_threshold) = 1;
        S_EBA_n(V_EBA_n < fire_threshold) = 0; 
        if sum(S_EBA_n) > 0  
            Fired_EBA = Fired_EBA + 1;
            W_LatInh_EBA_TPJ = tanh(W_LatInh_EBA_TPJ - 2 * acos(S_EBA_n) / pi .* exp(Fired_EBA) - 1) + 1;
            W_LatInh_EBA_AI = tanh(W_LatInh_EBA_AI - 2 * acos(S_EBA_n) / pi .* exp(Fired_EBA) - 1) + 1;
        end
      
        % S1 & EBA -> TPJ
        V_TPJ_Input = [V_S1_n, V_EBA_n];
        W_TPJ_Input = [W_S1_TPJ, W_EBA_TPJ];
        V_TPJ_n = Neuron_Post(V_TPJ, V_TPJ_Input, W_TPJ_Input, C_TPJ);         
        S_TPJ_n = V_TPJ_n;
        S_TPJ_n(V_TPJ_n >= fire_threshold) = 1; 
        S_TPJ_n(V_TPJ_n < fire_threshold) = 0;
        
        % Update weights
        % S1 -> TPJ    dW_S1_TPJ
        ddW_S1_TPJ = DeltaWeight(V_S1, V_S1_n, V_TPJ, V_TPJ_n, alpha, beta ,gamma);
        dW_S1_TPJ = dW_S1_TPJ + ddW_S1_TPJ; 
        % EBA -> TPJ   dW_EBA_TPJ  
        ddW_EBA_TPJ = DeltaWeight(V_S1, V_S1_n, V_TPJ, V_TPJ_n, alpha, beta ,gamma);
        dW_EBA_TPJ = dW_EBA_TPJ + ddW_EBA_TPJ;
            
        % S1 & TPJ & EBA -> AI
        V_AI_Input = [V_S1_n, V_TPJ_n, V_EBA_n];
        W_AI_Input = [W_S1_AI, W_TPJ_AI, W_EBA_AI]; 
        V_AI_n = Neuron_Post(V_AI, V_AI_Input, W_AI_Input, C_AI); 
        S_AI_n = V_AI_n;
        S_AI_n(V_AI_n >= fire_threshold) = 1; 
        S_AI_n(V_AI_n < fire_threshold) = 0;    
        
        % Update weights
        % S1 -> AI   dW_S1_AI
        ddW_S1_AI = DeltaWeight(V_S1, V_S1_n, V_AI, V_AI_n, alpha, beta ,gamma);
        dW_S1_AI = dW_S1_AI + ddW_S1_AI;
        % EBA -> AI  dW_EBA_AI
        ddW_EBA_AI = DeltaWeight(V_EBA, V_EBA_n, V_AI, V_AI_n, alpha, beta ,gamma);
        dW_EBA_AI = dW_EBA_AI + ddW_EBA_AI;

        V_M1 = V_M1_n;
        V_V = V_V_n;
        V_S1 = V_S1_n;
        V_EBA = V_EBA_n;
        V_TPJ = V_TPJ_n;
        V_AI = V_AI_n;       
    end    

    W_S1_TPJ = W_S1_TPJ + dW_S1_TPJ.*W_LatInh_S1_TPJ ; % S1 -> TPJ
    W_EBA_TPJ = W_EBA_TPJ + dW_EBA_TPJ.*W_LatInh_EBA_TPJ ; % EBA -> TPJ 
    W_S1_AI = W_S1_AI + dW_S1_AI.*W_LatInh_S1_AI ; % S1 -> AI
    W_EBA_AI = W_EBA_AI + dW_EBA_AI.*W_LatInh_EBA_AI ; % EBA -> AI 
      
    List_W_S1_TPJ = [List_W_S1_TPJ, W_S1_AI];
    List_W_EBA_TPJ = [List_W_EBA_TPJ, W_EBA_AI];
end

if k == 1
    figure;
    L1 = plot((1:MoveNum),List_W_S1_TPJ(k,:),'linewidth',2,'color','k')
    hold on;
    L2 = plot((1:MoveNum),List_W_EBA_TPJ(k,:),'linewidth',2,'color','r')
    xlabel('Training times','fontsize',14);
    ylabel('Weight','fontsize',14);
    set(gca,'fontsize',13)
    H = legend([L1,L2],'Weight_{S1-AI}','Weight_{EBA-AI}','Location', 'SouthEast')
    set(H,'FontSize',13,'FontWeight','normal')
    imgname = 'Train.png';   
    f = getframe(gcf);
    imwrite(f.cdata, imgname);
end

matname = 'Train.mat';                
save(matname, 'W_M1_S1', 'W_V_EBA', 'W_S1_TPJ', 'W_EBA_TPJ', 'W_TPJ_AI', 'W_S1_AI', 'W_EBA_AI')
disp('Training Finish');
end




