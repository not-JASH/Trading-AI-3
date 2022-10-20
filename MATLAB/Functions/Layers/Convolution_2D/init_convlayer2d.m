function layer = init_convlayer_2D(fs,nc,nf)
    % filter size, number of channels, number of filters
    assert(length(fs)==2,"filter must be two dimensional\n");

    layer.w = gpudl(init_gauss([fs, nc, nf]),'');
    layer.b = gpudl(zeros([nf,1],'single'),'');
end