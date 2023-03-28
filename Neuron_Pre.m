function [V_Sn] = Neuron_Pre(S,V_S, C)
V_Sn =  -C .* (V_S - S) + V_S;
end

