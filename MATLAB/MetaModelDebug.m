%{
        Script for Debugging Metamodel

        Jashua Luna
        November 2022
%}

WindowSize = 80;
nSubsamples = 30;
learnrate = 1e-3;

meta = metamodel(learnrate,WindowSize,nSubsamples); % init metamodel

inputsize = [WindowSize nSubsamples 1];

