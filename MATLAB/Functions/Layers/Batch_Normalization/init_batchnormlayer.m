function layer = init_batchnormlayer(nchannels)
%   initialize batch normalization layer
    layer.o = gpudl(zeros([nchannels 1],'single'),''); %offset
    layer.sf = gpudl(ones([nchannels 1],'single'),''); %scale factor
end