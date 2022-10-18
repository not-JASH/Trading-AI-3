function [layer] = init_grouped2dconvlayer(fs,ncpg,nfpg,ng)
%     initialize weights for two dimensional grouped convolution layer
%     fs = filtersize
%     ncpg = number of channels per group
%     nfpg = number of filters per group
%     ng = number of groups
    layer.w = gpudl(init_gauss([fs,ncpg,nfpg,ng]),'');   % weights
    layer.b = gpudl(init_gauss([nfpg*ng 1]),'');         % bias
end
