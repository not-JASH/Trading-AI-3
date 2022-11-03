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
    4   9   256 512;

    4   9   512 1024;
    4   9   1024 1;

    ];

windowsize  = 80;       % windowsize
batchsize   = 32;       % batchsize

Disc = discriminator(layersizes,windowsize);    % init discriminator
Disc.Debug = true;                              % enable debug
    
sample = rand(windowsize,batchsize);            % generate random sample
sample = gpudl(sample,'');                      % load sample onto gpu

output = Disc.predict(sample);                  % predict output
