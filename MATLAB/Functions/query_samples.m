function samples = query_samples(nSamples,ws)       
    samples = cell(nSamples,1);
    data = get_data;

    x = 1:ws;

    for i = 1:nSamples
        dt = [0 0 1];

        while ~all(dt == mean(dt))
            x0 = randi(length(data)-1-ws,1);
            dt = diff(data(x0+x,1));
        end
    
        samples{i} = data(x0+x,:);
    end
end