function dly = fullyconnectedlayer(dlx,weights,varargin)
% fully connected layer
    dly = fullyconnect(dlx,weights.w,weights.b,varargin{:});
end