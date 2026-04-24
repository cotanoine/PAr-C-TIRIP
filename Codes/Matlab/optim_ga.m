options = optimoptions('ga','PopulationSize',1,'MaxGeneration',100, ...
                       'Display','iter','PlotFcn', {@gaplotbestf,@gaplotbestindiv});

nvars=2;

fun=@optimarap;
A = [];
b = [];
Aeq = [];
beq = [];
lb = [0,0];
ub = [4,1];
nonlcon = [];
%intcon = [1,2,3,4];

[x,fval] = ga(fun,nvars,A,b,Aeq,beq,lb,ub,nonlcon,options);

