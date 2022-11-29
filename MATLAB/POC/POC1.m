%{
        Proof of Concept 1

        Jashua Luna
        November 2022
        
        Constant distribution between samples        
%}

data = get_data;            % load data from file
data = data(:,5)-data(:,2); % close - open = change in price / candle

[n,edges] = histcounts(data,1e3,'Normalization','probability'); % create histogram for entire dataset
cutoff = 0.001; % specify cutoff probablity

while n(1) < cutoff % remove datapoints with probablility less than cutoff
    n(1) = [];edges(1) = [];
end

while n(end) < cutoff % remove datapoints with probablility less than cutoff
    n(end) = [];edges(end) = [];
end

nsamples = 1e2;         % number of samples
windowsize = 2e3;       % windowsize

%% generate figure
samples = cell(nsamples,1);     % init cell array for samples
n = cell(nsamples,1);           % init cell array for histogram data

x = [-windowsize+1:0];          % template locs for sample window 
locs = randi([windowsize size(data,1)],[nsamples 1]);   % sample locs

for i = 1:nsamples 
    samples{i} = data(locs(i)+x);
    n{i} = histcounts(samples{i},edges,'Normalization','probability');
    mu(i) = mean(samples{i});
    sig(i) = std(samples{i});
end

%% generate mean and variance stats for subsamples

nsamples = 1e4;

samples = cell(nsamples,1);     % init cell array for samples
[mu,sig] = deal(zeros(nsamples,1)); % init arrays for mean and standard dev

x = [-windowsize+1:0];          % template locs for sample window 
locs = randi([windowsize size(data,1)],[nsamples 1]);   % sample locs

for i = 1:nsamples 
    samples{i} = data(locs(i)+x);
    mu(i) = mean(samples{i});
    sig(i) = std(samples{i});
end

































