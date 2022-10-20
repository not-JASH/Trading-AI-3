function layer = init_convlayer_1D(fs,nc,nf,datatype)
    assert(length(fs)==1,'Filter must be one dimensional\n');

    layer.w = gpudl(init_gauss([fs nc nf],datatype),'');
    layer.b = gpudl(zeros([nf 1],datatype));
end