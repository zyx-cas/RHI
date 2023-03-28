function setGlobalOpts()
global Opts

Opts.alpha = -0.0035;
Opts.beta = 0.35;
Opts.gamma = -0.55;

C = 0.04;    
Opts.C_M1 = C;
Opts.C_V = C;
Opts.C_S1 = C;
Opts.C_EBA = C;
Opts.C_TPJ = 0.01;
Opts.C_AI = 0.15;


Opts.ContinueTime = 100; 

Opts.MoveNum = 1000; 
Opts.Time = 2000; 
Opts.Weight = 1;

end

