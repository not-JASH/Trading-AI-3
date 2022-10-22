function data = binance_textLoad(path)
    file        = fopen(path);
    data        = textscan(file,'%f %f %f %f %f %f %f %f %f %f %f %f ');
    data        = cell2mat(data);
    fclose('all');
end