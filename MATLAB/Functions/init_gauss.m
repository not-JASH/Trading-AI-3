function mat = init_gauss(size,datatype,sigma)
    if nargin < 3;sigma = 0.05;end
    mat = sigma.*randn(size,datatype);
end