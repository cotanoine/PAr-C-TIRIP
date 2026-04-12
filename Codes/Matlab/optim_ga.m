options = optimoptions('ga','PopulationSize',10,'MaxGeneration',100, ...
                       'Display','iter','PlotFcn', {@gaplotbestf,@gaplotbestindiv});

nvars=2;

fun=@optimarap;
A = [];
b = [];
Aeq = [];
beq = [];
lb = [0.01,0.01];
ub = [0.2,0.2];
nonlcon = [];

[x,fval] = ga(fun,nvars,A,b,Aeq,beq,lb,ub,nonlcon,options);

