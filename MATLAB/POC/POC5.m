%{
        Proof of Concept 5

        Jashua Luna
        November 2022

        Change in layersize
%}


%% Convolutional Layers

layers = [
    3   4   4   4   16  32  1   1   1;

    3   4   4   8   32  32  1   1   1;

    3   4   4   8   32  32  1   1   1;
    ];

inputsize = [256 256 256 16];

outputsize = sizechange(inputsize,layers(1,:))
outputsize = sizechange(outputsize,layers(2,:))



function outputsize = sizechange(inputsize,layerinfo)
    % function for calculating dimension change from filtersize and stride
    % layerinfo [ndims filtersize nchannelsin nchannelsout stride];

    nd = layerinfo(1);                  % number of dimensions
    fs = layerinfo(2:nd+1);             % filtersizes
    stride = layerinfo(end-nd+1:end);   % stride

    assert(length(layerinfo)==2*nd+3,"layerinfo must contain info for stride, filtersize, ndims, nchannelsin and nchannelsout");
    assert(layerinfo(nd+2)==inputsize(end),"nchannelsin must match inputsize's channel dimension");

    outputsize = inputsize;
    
    outputsize(1:nd) = 1 + floor((inputsize(1:nd)-fs)./stride);
    outputsize(end) = layerinfo(end-nd);
end
