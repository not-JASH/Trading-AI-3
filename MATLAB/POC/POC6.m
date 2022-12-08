%{
        Proof of Concept 6

        Jashua Luna
        November 2022

        Generating Layersizes for Network

        Generating layersizes for network based on some metaparameters. 
        
        This script assumes stride = 1 for all convolutions/transposed convolutions in all dimensions.
        When stride=1 for a particular dimension, the convolution output for that dimension changes by filtersize-1
        
        At this point filtersizes are uniform throughout a layer

%}

%% Global Parameters

ws  = 80;       % windowsize
nss = 30;       % number of subsamples

TBD = NaN;

%% Reweight Detrend Layer

rdl_inputsize = [ws nss];

% Input Encoder

ie_fs   = [4];
ie_nci  = nss;

ie_nlayers      = TBD;
% ie_nco          = TBD;
ie_outputsize   = TBD;

% Reweight Decoder

rd_inputsize = ie_outputsize;

rd_fs   = [4];
rd_nci  = ie_nco;

rd_nlayers      = TBD;
% rd_nco          = TBD;
rd_outputsize   = rdl_inputsize;

% Detrend Decoder
% has the same sizes as reweight decoder

rdl_outputsize = rdl_inputsize;


%% Wavelet Layer

wl_inputsize = rdl_inputsize;
wl_outputsize = [size(cwt(rand(ws,1))) wl_inputsize(2:end)];

%% Set Encoder

se_inputsize = wl_outputsize;

se_fs   = [4 4 4];
se_nci  = 2;

se_nlayers      = TBD;
se_nco          = TBD;
se_outputsize   = TBD;

%% Estimate Encoder

% se_inputsize = [ws 1]; -> input is wavelet transformed
ee_inputsize = size(cwt(rand(ws,1)));

ee_fs   = [4 4];
ee_nci  = 2;

ee_nlayers      = TBD;
% ee_nco          = TBD;
ee_outputsize   = TBD;

%% Prediction Layer

pl_inputsize1 = se_outputsize;
pl_inputsize2 = ee_outputsize;

% Set decoder

sd_inputsize = pl_inputsize1;

sd_fs   = [4 4];
sd_nci  = se_nco;

sd_nlayers  = TBD;
% sd_nco      = ee_nco;   
se_outputsize = pl_inputsize2;


% Prediction Decoder

pd_inputsize = se_outputsize;

pd_fs = [4];
pd_nci = sd_nco;

pd_nlayers  = TBD;
% pd_nco      = 1;
pd_outputsize = ws;

%% Inverse Detrend Layer

idl_inputsize1 = pd_outputsize;
idl_inputsize2 = ie_outputsize;

% detrend decoder

dd_inputsize = idl_inputsize2;

dd_fs = [4];
dd_nci = ie_nco;

dd_nlayers = TBD;
% dd_nco  = 1;
dd_outputsize = idl_inputsize1;


function outputsize = determine_outputsize(metainfo,ws,nss)
    % function which determines output size based on metainfo -> does not take into account nchannels
    






end






















