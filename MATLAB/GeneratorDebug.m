%{
        Script for Debugging Generator

        Jashua Luna
        October 2022
%}

% Set Training Variables

WindowSize              = 80;
ExtrapolationLength     = 15;
Overlap                 = 20;
nSubsamples             = 30;
nSamples                = 1e3;

BatchSize               = 48;

ns = nSubsamples;

% Establish sizes for each layer

% layersizes.SL   = [];   % Scaling Layer

layersizes.RDL.IE   = [     % Reweight-Detrend Layer -> Input Encoder
    4   ns  64;
    4   64  128;

    4   128 256;
    4   256  512;

    4   512 512;
    4   512 512; 
    ];   

layersizes.RDL.RD   = [     % Reweight-Detrend Layer -> Reweight Decoder
    4   512     256;
    4   256     256;
    4   256     128;
    4   128     64;
    4   64      64;
    4   64      30;
    ];  

layersizes.RDL.DD   = [     % Reweight-Detrend Layer -> Detrend Decoder
    4   512     256;
    4   256     256;
    4   256     128;
    4   128     64;
    4   64      64;    
    4   64      30;
    ];   

layersizes.WTL  = [];       % Wavelet Transform Layer
layersizes.SEL  = [         % Set Encoding Layer
    3   6   30   32  2;
    3   6   32   64  2;

    3   6   32  64  4;
    3   6   64  128 4;

    3   3   64  128 8;
    3   3   128 256 8;

    3   3   256 128 8;  
    3   3   128 64  8;

    ];   

layersizes.EEL  = [         % Estimate Encoding Layer
    4   4   2   32;
    3   4   32  128;

    3   4   128  64;
    3   4   64  32;
    ];   

layersizes.PL.w2D   = [         % Prediction layer 2D component
    3   5   512    256;
    3   5   256    128;
    3   5   128    64;
    2   5   64     32;
    ];

layersizes.PL.w1D   = [         % Prediction Layer 1D component
    4   680    512;
    4   512     256;     
    4   256     128;
    4   128     64;
    4   64      32;
    4   32      16;
    4   16      8;
    4   8       4;
    4   4       1;
    ];   

layersizes.IDL  = [     % Inverse-Detrend Layer
    4   512 256;
    4   256 128;
    4   128 64;
    4   64  32;
    3   32  16;
    3   16  8;
    2   8   4;
    2   4   1;
    ];   

return
% layersizes.ISL  = [];   % Inverse-Scaling Layer

InputSize = nSubsamples*(WindowSize-Overlap) + WindowSize;

% Initialize Networks

Generator = generator(layersizes,WindowSize,ExtrapolationLength,Overlap,nSubsamples,BatchSize,1e-3);
[~,total_params] = DeepNetwork.struct_tree2cell(Generator.weights);
display_number("Total Params:",total_params);
Generator.Debug = true;

sample_x        = rand(InputSize,BatchSize);
sample_xhat     = rand(WindowSize,BatchSize);

Generator.predict(sample_x,sample_xhat);