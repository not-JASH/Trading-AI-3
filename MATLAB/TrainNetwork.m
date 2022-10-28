%{
        Script for Training Network

        Jashua Luna
        October 2022
%}


% Establish sizes for each layer

% layersizes.SL   = [];   % Scaling Layer

layersizes.RDL.IE   = [
    8   25  64;
    6   64  128;
    4   128 512;
    2   512  1024;
    ];   % Reweight-Detrend Layer -> Input Encoder
layersizes.RDL.RD   = [];   % Reweight-Detrend Layer -> Reweight Decoder
layersizes.RDL.DD   = [];   % Reweight-Detrend Layer -> Detrend Decoder

layersizes.WTL  = [];   % Wavelet Transform Layer
layersizes.SEL  = [];   % Set Encoding Layer
layersizes.EEL  = [];   % Estimate Encoding Layer
layersizes.PL   = [];   % Prediction Layer
layersizes.IDL  = [];   % Inverse-Detrend Layer
layersizes.ISL  = [];   % Inverse-Scaling Layer


% Set Training Variables

WindowSize              = 480;
ExtrapolationLength     = 45;
Overlap                 = 60;
nSubsamples             = 25;
nSamples                = 1e3;

BatchSize               = 32;


InputSize = nSubsamples*(WindowSize-Overlap) + WindowSize;



% Initialize Networks

Generator = generator(layersizes,WindowSize,Overlap,nSubsamples);
Generator.Debug = true;

sample_x        = rand(InputSize,BatchSize);
sample_xhat     = rand(WindowSize,BatchSize);

Generator.predict(sample_x,sample_xhat);