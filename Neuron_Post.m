function [V_Rn] = Neuron_Post(V_R,V_S, W, C)
Sf = tanh(sum(V_S.*W,2)); 
Sf(Sf<0) = 0;
V_Rn = -C .* (V_R - Sf) + V_R;
V_Rn(V_Rn<0) = 0;
end

