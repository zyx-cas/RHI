function [dW] = DeltaWeight(V_S, V_Sn, V_R, V_Rn, alpha, beta ,gamma)
T1 = alpha .* (V_Sn.*V_Rn);
T2 = beta .* (V_Sn.*(V_Rn-V_R));
T3 = gamma .* ((V_Sn-V_S).*V_Rn);
dW = T1 + T2 + T3;
end

