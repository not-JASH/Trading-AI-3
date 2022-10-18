function layer = init_groupedtransposedconvolutionlayer(fs,nc,nf,ng)
% function for initializing transposed convolution
    layer.w = gpudl(init_gauss([fs,nf,nc,ng]),'');
    layer.b = gpudl(zeros([nf*ng 1],'single'),'');
end