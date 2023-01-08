function data = get_data(varargin)
    if nargin > 0
        datapath = varargin{1};
    else
        datapath = "Data";
    end

    files = dir(datapath);
    files(1:2) = [];

    data = {};
    for i = 1:length(files)
        if endsWith(files(i).name,".txt")
            data{end+1} = binance_textLoad(append(datapath,"/",files(i).name));
        end
    end
    data = cell2mat(data');
end