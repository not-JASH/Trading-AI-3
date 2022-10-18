function layer = init_convlayer3d(fs,nc,nf)
    % filter size, number of channels, number of filters
    assert(length(fs)==3,"filter must be three dimensional\n");

    layer.w = gpudl(init_gauss([fs, nc, nf]),'');
    layer.b = gpudl(zeros([nf,1],'single'),'');
end