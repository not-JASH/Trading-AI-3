function data = get_data
    files = dir("Data");
    files(1:2) = [];

    data = {};
    for i = 1:length(files)
        if endsWith(files(i).name,".txt")
            data{end+1} = binance_textLoad(append("Data/",files(i).name));
        end
    end
    data = cell2mat(data');
end