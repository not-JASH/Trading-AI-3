function dly = convlayer_1D(dlx,layer,varargin)
    dly = dlconv(dlx,layer.w,varargin{:});
end