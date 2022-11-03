%{
        Script for Debugging Generator

        Jashua Luna
        October 2022
%}


% Establish sizes for each layer

% layersizes.SL   = [];   % Scaling Layer

layersizes.RDL.IE   = [     % Reweight-Detrend Layer -> Input Encoder
    8   64  64;
    8   64  128;

    4   128 256;
    4   256  512;

    4   512 512;
    4   512 512; 
    ];   
layersizes.RDL.RD   = [     % Reweight-Detrend Layer -> Reweight Decoder
    12   512     256;
    12   256     128;
    12   128     64;
    11   64      64;
    8    64      64;
    ];  
layersizes.RDL.DD   = [     % Reweight-Detrend Layer -> Detrend Decoder
    12   512     256;
    12   256     128;
    12   128     64;
    11   64      64;    
    8    64      64;
    ];   

layersizes.WTL  = [];       % Wavelet Transform Layer
layersizes.SEL  = [         % Set Encoding Layer
    4   9   7   2   16;
    4   9   7   16  32;

    4   9   7   32  64;
    4   9   7   64  128;

    4   7   5   128  256;
    4   7   5   256  512;

    3   5   3   512 512;
    2   5   3   512 512;
    ];   

layersizes.EEL  = [         % Estimate Encoding Layer
    3   16   2   32;
    3   16   32  64;

    3   9   64  128;
    3   9   128  256;
    ];   

layersizes.PL.w2D   = [         % Prediction layer 2D component
    3   3   512    512;
    3   3   512     256;
    ];

layersizes.PL.w1D   = [         % Prediction Layer 1D component
    8   1280    1024;
    8   1024   512;
    8   512     256;     
    4   256     128;
    4   128     64;
    4   64      32;
    2   32      16;
    2   16      8;
    2   8       4;
    1   4       1;
    ];   
layersizes.IDL  = [     % Inverse-Detrend Layer
    8   512 256;
    8   256 128;
    8   128 64;
    8   64  32;
    8   32  16;
    8   16  8;
    6   8   4;
    4   4   1;
    ];   
% layersizes.ISL  = [];   % Inverse-Scaling Layer

% Set Training Variables

WindowSize              = 80;
ExtrapolationLength     = 15;
Overlap                 = 20;
nSubsamples             = 64;
nSamples                = 1e3;

BatchSize               = 48;


InputSize = nSubsamples*(WindowSize-Overlap) + WindowSize;

% Initialize Networks

Generator = generator(layersizes,WindowSize,Overlap,nSubsamples,BatchSize);
[~,total_params] = DeepNetwork.struct_tree2cell(Generator.weights);
display_number("Total Params:",total_params);
Generator.Debug = true;

sample_x        = rand(InputSize,BatchSize);
sample_xhat     = rand(WindowSize,BatchSize);

Generator.predict(sample_x,sample_xhat);