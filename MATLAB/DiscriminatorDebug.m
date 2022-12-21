%{
        Script for Debugging Discriminator

        Jashua Luna
        October 2022
%}


layersizes.PL = [               % discriminator prediction layer sizes
        
    16   36   2   32;
    12   27   32  32

    4   9   32  128;
    4   9   128  128;
    
    4   9   128 512;    
    4   9   512 1;     
    ];

windowsize  = 80;       % windowsize
batchsize   = 32;       % batchsize

Disc = discriminator(layersizes,windowsize,1e-3);    % init discriminator
Disc.Debug = true;                              % enable debug

[~,total_params] = DeepNetwork.struct_tree2cell(Disc.weights);
    
sample = rand(windowsize,batchsize);            % generate random sample
sample = gpudl(sample,'');                      % load sample onto gpu

output = Disc.predict(sample);                  % predict output
