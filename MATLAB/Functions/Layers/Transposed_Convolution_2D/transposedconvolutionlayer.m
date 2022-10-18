function dly = transposedconvolutionlayer(dlx,layer,varargin)
% function wrapper for transposed convolution
    dly = dltranspconv(dlx,layer.w,layer.b,varargin{:});
end