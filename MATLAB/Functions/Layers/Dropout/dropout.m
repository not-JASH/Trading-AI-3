function dly = dropout(dly,rate)
    dly(randperm(numel(dly),floor(rate*numel(dly)))) = 0;   
end