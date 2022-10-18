function dly = grouped2dconvolutionlayer(dlx,layer,varargin)
%     grouped two dimensional convolution layer
    dly = dlconv(dlx,layer.w,layer.b,varargin{:});
end