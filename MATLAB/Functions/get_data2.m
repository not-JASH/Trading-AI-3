function data = get_data2(varargin)
    % function that generates datasets

    nobservations = 60*24*365*2;

    ndof = 30;
    
    elmts = cell(ndof,1);
    
    for i = 1:ndof
        elmts{i}.mu = 5*rand(1,1);
        elmts{i}.sig = 5*rand(1,1);
    end
    
    observations = zeros(nobservations,1);
    
    for i = 1:nobservations
        observations(i) = normrnd(elmts{1}.mu,elmts{1}.sig) - elmts{1}.mu;
        for j = 2:ndof
            observations(i) = observations(i) + normrnd(elmts{j}.mu,elmts{j}.sig) - elmts{j}.mu;
        end
    end
    
    observations = cumsum(observations);
    
    data = zeros(nobservations-1,12);
    data(:,2) = observations(1:end-1);
    data(:,5) = observations(2:end);
    
    % mimic the structure of binance's data




end