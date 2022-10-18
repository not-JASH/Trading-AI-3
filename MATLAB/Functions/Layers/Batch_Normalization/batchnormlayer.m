function [dly] = batchnormlayer(dlx,weights,varargin)
%   batch normalization layer
    [dly] = batchnorm(dlx,weights.o,weights.sf,varargin{:});
end