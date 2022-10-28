function data = gpudl(data,labels)
    % function for creating taced deep learning arrays on gpu
    data = gpuArray(dlarray(data,labels));
end 