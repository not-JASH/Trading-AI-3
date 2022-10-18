function [weights] = init_fullyconnectedlayer(dimensions,labels)
%   initialize fully connected layer
%   dimensionst are [output features, ...]

    weights.w = gpudl(init_gauss(dimensions),labels);         %weights
    weights.b = gpudl(init_gauss([dimensions(1),1]),labels);    %bias
end