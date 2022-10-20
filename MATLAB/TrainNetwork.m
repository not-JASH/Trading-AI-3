%{
        Script for Training Network

        Jashua Luna
        October 2022
%}


% Establish sizes for each layer

layersizes.SL   = [];   % Scaling Layer
layersizes.RDL  = [];   % Reweight-Detrend Layer
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

nSamples = 1e3;

% Initialize Networks

Generator = generator(layersizes,WindowSize,Overlap,ExtrapolationLength);
