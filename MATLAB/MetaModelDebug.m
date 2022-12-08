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


mparams.RDL.IE.nblocks = 3;

mparams.RDL.IE.conv1.params = [8  64  64;4 128 128; 4 512 512];
mparams.RDL.IE.conv1.stride = [1;1;1];
mparams.RDL.IE.conv1.nd = 1;

mparams.RDL.IE.conv2.params = [8 64 128;4 256 512;4 512 512];
mparams.RDL.IE.conv2.stride = [1;1;1];
mparams.RDL.IE.conv2.nd = 1;
