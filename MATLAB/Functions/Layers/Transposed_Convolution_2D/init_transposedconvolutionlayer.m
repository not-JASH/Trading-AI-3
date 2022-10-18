function layer = init_transposedconvolutionlayer(fs,nc,nf)
% function for initializing transposed convolution
    layer.w = gpudl(init_gauss([fs,nf,nc]),'');
    layer.b = gpudl(init_gauss([nf 1]),'');
end