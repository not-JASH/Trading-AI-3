function dly = groupedtransposedconvolutionlayer(dlx,layer,varargin)
% function wrapper for transposed convolution
    dly = dltranspconv(dlx,layer.w,layer.b,varargin{:});
end