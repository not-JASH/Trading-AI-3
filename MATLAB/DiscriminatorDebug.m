%{
        Script for Debugging Discriminator

        Jashua Luna
        October 2022
%}


layersizes.PL = [               % discriminator prediction layer sizes
        
    4   9   2   16;
    4   9   16  32

    4   9   32  64;
    4   9   64  128;
    
    4   9   128 256;    
    4   9   256 1;
    ];

windowsize  = 80;       % windowsize
batchsize   = 32;       % batchsize

Disc = discriminator(layersizes,windowsize,1e-3);    % init discriminator
Disc.Debug = true;                              % enable debug

[~,total_params] = DeepNetwork.struct_tree2cell(Disc.weights);
    
sample = rand(windowsize,batchsize);            % generate random sample
sample = gpudl(sample,'');                      % load sample onto gpu

output = Disc.predict(sample);                  % predict output
