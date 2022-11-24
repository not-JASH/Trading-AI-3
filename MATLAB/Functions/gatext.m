function data = gatext(data)
% function for converting dl arrays on gpu into normal arrays in heap
    data = gather(extractdata(data));
end